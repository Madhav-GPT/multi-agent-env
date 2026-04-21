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

    def __init__(self) -> None:
        token = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
        self._client = None
        if InferenceClient is not None and token:
            try:
                self._client = InferenceClient(
                    model=self.model_id,
                    token=token,
                    provider=self.provider,
                )
            except Exception:
                self._client = None

    @abstractmethod
    def fallback_report(self, state: Any) -> SpecialistReport:
        raise NotImplementedError

    def generate_report(self, state: Any) -> SpecialistReport:
        return self.generate_execution(state).report

    def generate_execution(self, state: Any) -> SpecialistExecutionResult:
        start = time.perf_counter()
        if self._client is None or os.getenv("SPECTRA_MODE", "training") != "demo":
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
            content = response.choices[0].message.content
            report = SpecialistReport.model_validate_json(content)
            latency_ms = (time.perf_counter() - start) * 1000.0
            execution = report.to_execution(
                mode="llm",
                raw_response=content,
                latency_ms=latency_ms,
                followup_applied=bool(getattr(state, "followup_requested", False)),
            )
            return SpecialistExecutionResult(report=report, execution=execution)
        except Exception:
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

    async def generate_execution_async(self, state: Any) -> SpecialistExecutionResult:
        return await asyncio.to_thread(self.generate_execution, state)

    def _system_prompt(self) -> str:
        prompt_path = Path(__file__).resolve().parent / "prompts" / self.prompt_file
        return prompt_path.read_text(encoding="utf-8")
