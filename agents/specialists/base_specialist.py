"""Base class for demo-mode LLM specialists."""

from __future__ import annotations

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.specialist_report import SpecialistReport
from runtime.env import load_runtime_env

try:
    from huggingface_hub import InferenceClient
except Exception:  # pragma: no cover - optional runtime dependency
    InferenceClient = None  # type: ignore[assignment]


@dataclass
class SpecialistExecutionResult:
    report: SpecialistReport
    execution: Any


class BaseSpecialist(ABC):
    """Common LLM wrapper with deterministic fallback and execution tracing."""

    model_id: str
    provider: str | None = None
    prompt_file: str
    max_retries: int = 3
    model_env_var: str | None = None
    provider_env_var: str | None = None

    def __init__(self) -> None:
        load_runtime_env()
        token = os.getenv("SPECIALIST_HF_TOKEN") or os.getenv("HF_TOKEN")
        self._model_id = os.getenv(self.model_env_var, self.model_id) if self.model_env_var else self.model_id
        self._provider = os.getenv(self.provider_env_var, self.provider) if self.provider_env_var else self.provider
        self._client = None
        if InferenceClient is not None and token:
            try:
                self._client = InferenceClient(
                    model=self._model_id,
                    token=token,
                    provider=self._provider,
                )
            except Exception:
                self._client = None

    @abstractmethod
    def fallback_report(self, state: Any) -> SpecialistReport:
        raise NotImplementedError

    def generate_report(self, state: Any) -> SpecialistReport:
        return self.generate_execution(state).report

    def generate_execution(self, state: Any, *, specialist_mode: str = "deterministic") -> SpecialistExecutionResult:
        start = time.perf_counter()
        normalized_mode = specialist_mode.lower()
        use_llm = normalized_mode in {"llm", "hybrid"}
        if self._client is None or not use_llm:
            report = self.fallback_report(state)
            latency_ms = (time.perf_counter() - start) * 1000.0
            raw_response = json.dumps(report.model_dump(), ensure_ascii=True)
            execution = report.to_execution(
                mode="rule",
                raw_response=raw_response,
                latency_ms=latency_ms,
                followup_applied=bool(getattr(state, "followup_requested", False)),
            )
            return SpecialistExecutionResult(report=report, execution=execution)

        try:
            content = self._request_llm_json(state)
            report = self._parse_report(content, state)
            latency_ms = (time.perf_counter() - start) * 1000.0
            execution = report.to_execution(
                mode="llm",
                raw_response=content,
                latency_ms=latency_ms,
                followup_applied=bool(getattr(state, "followup_requested", False)),
            )
            return SpecialistExecutionResult(report=report, execution=execution)
        except Exception:
            if normalized_mode == "llm":
                raise
            report = self.fallback_report(state)
            latency_ms = (time.perf_counter() - start) * 1000.0
            raw_response = json.dumps(report.model_dump(), ensure_ascii=True)
            execution = report.to_execution(
                mode="rule",
                raw_response=raw_response,
                latency_ms=latency_ms,
                followup_applied=bool(getattr(state, "followup_requested", False)),
            )
            return SpecialistExecutionResult(report=report, execution=execution)

    async def generate_execution_async(self, state: Any, *, specialist_mode: str = "deterministic") -> SpecialistExecutionResult:
        return await asyncio.to_thread(self.generate_execution, state, specialist_mode=specialist_mode)

    def _system_prompt(self) -> str:
        prompt_path = Path(__file__).resolve().parent / "prompts" / self.prompt_file
        return prompt_path.read_text(encoding="utf-8")

    def _request_llm_json(self, state: Any) -> str:
        assert self._client is not None
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat_completion(
                    messages=[
                        {"role": "system", "content": self._system_prompt()},
                        {
                            "role": "user",
                            "content": (
                                "Return JSON only, matching the SpecialistReport schema.\n\n"
                                f"Observation:\n{json.dumps(state.model_dump(), indent=2)}"
                            ),
                        },
                    ],
                    max_tokens=500,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(float(attempt))
        assert last_error is not None
        raise last_error

    def _parse_report(self, content: str, state: Any) -> SpecialistReport:
        try:
            return SpecialistReport.model_validate_json(content)
        except Exception:
            candidate = content.strip()
            if "```" in candidate:
                candidate = candidate.replace("```json", "```").split("```", 2)[1].strip()
            if "{" in candidate and "}" in candidate:
                candidate = candidate[candidate.find("{") : candidate.rfind("}") + 1]
            try:
                payload = json.loads(candidate)
            except Exception:
                repaired = self._repair_report_json(candidate, state)
                payload = json.loads(repaired)
            if not isinstance(payload, dict):
                raise ValueError("Specialist output did not parse to an object.")

            fallback = self.fallback_report(state)
            normalized = {
                "agent_id": payload.get("agent_id", fallback.agent_id.value),
                "observation_digest": payload.get("observation_digest", fallback.observation_digest),
                "confidence": self._coerce_confidence(payload.get("confidence"), fallback.confidence),
                "top_hypothesis_service": (
                    payload.get("top_hypothesis_service")
                    or payload.get("target_service")
                    or payload.get("likely_culprit_service")
                    or fallback.top_hypothesis_service
                ),
                "top_hypothesis_cause": (
                    payload.get("top_hypothesis_cause")
                    or payload.get("cause_description")
                    or payload.get("earliest_failure_signature")
                    or payload.get("attack_type")
                    or fallback.top_hypothesis_cause
                ),
                "supporting_evidence": self._coerce_list(
                    payload.get("supporting_evidence") or payload.get("evidence"),
                    fallback.supporting_evidence,
                ),
                "recommended_action": (
                    payload.get("recommended_action")
                    or payload.get("immediate_action")
                    or payload.get("recommended_immediate_action")
                    or fallback.recommended_action
                ),
                "uncertainty_flags": self._coerce_list(
                    payload.get("uncertainty_flags") or payload.get("cannot_determine"),
                    fallback.uncertainty_flags,
                ),
                "severity": self._coerce_severity(payload.get("severity"), fallback.severity),
                "timestamp": payload.get("timestamp", fallback.timestamp),
                "followup_brief": payload.get("followup_brief") or payload.get("note"),
            }
            return SpecialistReport(**normalized)

    def _repair_report_json(self, raw_output: str, state: Any) -> str:
        if self._client is None:
            raise ValueError("No client available for report repair.")
        fallback = self.fallback_report(state)
        response = self._client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the given output as one valid JSON object matching this schema only: "
                        "{\"agent_id\": str, \"observation_digest\": str, \"confidence\": float, "
                        "\"top_hypothesis_service\": str, \"top_hypothesis_cause\": str, "
                        "\"supporting_evidence\": list[str], \"recommended_action\": str, "
                        "\"uncertainty_flags\": list[str], \"severity\": str, \"timestamp\": int, "
                        "\"followup_brief\": str|null}. Return JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Broken output:\n{raw_output}\n\n"
                        f"Use this as a safe field reference if needed:\n{json.dumps(fallback.model_dump(), ensure_ascii=True)}"
                    ),
                },
            ],
            max_tokens=500,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _coerce_confidence(value: Any, fallback: float) -> float:
        if value is None:
            return fallback
        try:
            parsed = float(value)
        except Exception:
            return fallback
        return max(0.0, min(1.0, round(parsed, 3)))

    @staticmethod
    def _coerce_list(value: Any, fallback: list[str]) -> list[str]:
        if value is None:
            return list(fallback)
        if isinstance(value, list):
            return [str(item) for item in value][:4]
        if isinstance(value, str):
            return [value]
        return list(fallback)

    @staticmethod
    def _coerce_severity(value: Any, fallback: str) -> str:
        allowed = {"critical", "high", "medium", "low"}
        if isinstance(value, str) and value.lower() in allowed:
            return value.lower()
        return fallback
