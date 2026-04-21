"""Commander policy wrapper."""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from openai import OpenAI

from agents.observation_builder import build_commander_observation
from agents.specialist_report import AgentID, SpecialistReport
from environments.shared.state import CommanderAction, CommanderExecution, MasterSREState
from runtime.env import load_runtime_env

from .action_parser import parse_action

try:
    from huggingface_hub import InferenceClient
except Exception:  # pragma: no cover - optional runtime dependency
    InferenceClient = None  # type: ignore[assignment]


@dataclass
class CommanderDecision:
    action: CommanderAction
    execution: CommanderExecution


class Commander:
    """Heuristic commander with optional OpenAI-compatible inference backend."""

    def __init__(self) -> None:
        load_runtime_env()
        self._client = None
        self._provider = os.getenv("COMMANDER_PROVIDER", "openai")
        base_url = os.getenv("API_BASE_URL")
        if os.getenv("COMMANDER_MODE", "heuristic") == "llm":
            try:
                if self._provider == "hf":
                    token = os.getenv("HF_TOKEN")
                    if token and InferenceClient is not None:
                        self._client = InferenceClient(
                            model=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
                            token=token,
                            provider=os.getenv("COMMANDER_HF_PROVIDER"),
                        )
                elif base_url:
                    api_key = os.getenv("OPENAI_API_KEY", "local")
                    self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=45.0)
            except Exception:
                self._client = None
        self._system_prompt = (Path(__file__).resolve().parent / "system_prompt.txt").read_text(encoding="utf-8")
        self.reset()

    def reset(self) -> None:
        self.isolated_targets: set[str] = set()
        self.rolled_back_targets: set[str] = set()
        self.recovered_targets: set[str] = set()
        self.last_trust_weights = {"infra": 0.0, "log": 0.0, "security": 0.0}
        self.primary_target: str | None = None

    def decide(
        self,
        reports: list[SpecialistReport],
        state: MasterSREState,
        *,
        allowed_actions: list[str],
    ) -> CommanderDecision:
        if self._client is not None:
            try:
                start = time.perf_counter()
                observation = build_commander_observation(reports, state, allowed_actions=allowed_actions)
                if self._provider == "hf":
                    response = self._client.chat_completion(  # type: ignore[union-attr]
                        messages=[
                            {"role": "system", "content": self._system_prompt},
                            {"role": "user", "content": observation},
                        ],
                        max_tokens=220,
                        temperature=0.2,
                        response_format={"type": "json_object"},
                    )
                    content = response.choices[0].message.content or ""
                else:
                    response = self._client.chat.completions.create(  # type: ignore[union-attr]
                        model=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
                        messages=[
                            {"role": "system", "content": self._system_prompt},
                            {"role": "user", "content": observation},
                        ],
                        max_tokens=220,
                        temperature=0.2,
                    )
                    content = response.choices[0].message.content or ""
                action = parse_action(content)
                action.reasoning = content
                self._remember(action)
                weights = self._update_trust_weights(reports, state)
                return CommanderDecision(
                    action=action,
                    execution=CommanderExecution(
                        mode="llm",
                        raw_response=content,
                        latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
                        trust_weights=weights,
                        action=action.model_dump(),
                    ),
                )
            except Exception:
                pass

        return self._heuristic_decision(reports, state, allowed_actions=allowed_actions)

    def _heuristic_decision(
        self,
        reports: list[SpecialistReport],
        state: MasterSREState,
        *,
        allowed_actions: list[str],
    ) -> CommanderDecision:
        start = time.perf_counter()
        allowed_action_names = {action.split("(", 1)[0] for action in allowed_actions}
        weights = self._update_trust_weights(reports, state)
        ranked_reports = sorted(reports, key=lambda report: weights[report.agent_id.value], reverse=True)
        service_votes = Counter(report.top_hypothesis_service for report in reports)
        weighted_scores: dict[str, float] = defaultdict(float)
        for report in reports:
            weighted_scores[report.top_hypothesis_service] += weights[report.agent_id.value]
        consensus_service, consensus_count = service_votes.most_common(1)[0]
        target = self.primary_target or max(weighted_scores, key=weighted_scores.get)
        has_conflict = len(service_votes) > 1
        best_followup = state.best_followup_agent or ranked_reports[0].agent_id.value

        if (
            "request_followup" in allowed_action_names
            and state.follow_up_count == 0
            and has_conflict
            and state.best_followup_agent
        ):
            action = CommanderAction(
                action_type="request_followup",
                target_agent=best_followup,  # type: ignore[arg-type]
                reasoning="Specialists disagree on the origin service; request the most trusted follow-up before containment.",
            )
        elif (
            "investigate_service" in allowed_action_names
            and state.workflow_stage == "triage"
            and consensus_count < 2
        ):
            action = CommanderAction(
                action_type="investigate_service",
                target_service=target,
                reasoning="No majority service target yet; validate the strongest weighted hypothesis.",
            )
        elif "isolate_service" in allowed_action_names and target not in self.isolated_targets:
            action = CommanderAction(
                action_type="isolate_service",
                target_service=target,
                reasoning="Contain the service with the strongest combined specialist signal.",
            )
        elif "rollback_config" in allowed_action_names and target not in self.rolled_back_targets:
            action = CommanderAction(
                action_type="rollback_config",
                target_service=target,
                reasoning="Containment is done; remove the risky config or release now.",
            )
        elif {"restart_service", "scale_service"} & allowed_action_names:
            recovery_action = self._infer_recovery_action(target, allowed_action_names)
            action = CommanderAction(
                action_type=recovery_action,  # type: ignore[arg-type]
                target_service=target,
                reasoning="Recovery stage: apply the scenario-specific stabilizing action.",
            )
        elif "submit_resolution" in allowed_action_names:
            dominant_causes = ", ".join(
                report.top_hypothesis_cause for report in ranked_reports[:2]
            )
            action = CommanderAction(
                action_type="submit_resolution",
                resolution_summary=f"Contained and recovered {target} after confirming: {dominant_causes}.",
                reasoning="The containment, remediation, and recovery chain is complete.",
            )
        elif "investigate_service" in allowed_action_names:
            action = CommanderAction(
                action_type="investigate_service",
                target_service=target,
                reasoning="Investigate the service with the best current consensus.",
            )
        else:
            action = CommanderAction(
                action_type="request_followup",
                target_agent=best_followup,  # type: ignore[arg-type]
                reasoning="No stronger action is currently allowed; ask for clarification.",
            )

        self._remember(action)
        return CommanderDecision(
            action=action,
            execution=CommanderExecution(
                mode="heuristic",
                raw_response=json.dumps(action.model_dump(), ensure_ascii=True),
                latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
                trust_weights=weights,
                action=action.model_dump(),
            ),
        )

    def _update_trust_weights(self, reports: Iterable[SpecialistReport], state: MasterSREState) -> dict[str, float]:
        weights: dict[str, float] = {}
        report_list = list(reports)
        for report in report_list:
            bias = 1.0
            if report.agent_id == AgentID.SECURITY:
                bias = 1.08
            elif report.agent_id == AgentID.LOG:
                bias = 1.03
            elif report.agent_id == AgentID.INFRA:
                bias = 0.96
            weights[report.agent_id.value] = round(report.confidence * bias, 3)

        if state.decoy_agent and state.decoy_service:
            for report in report_list:
                if report.agent_id.value == state.decoy_agent and report.top_hypothesis_service == state.decoy_service:
                    weights[report.agent_id.value] = round(weights[report.agent_id.value] * 0.72, 3)
            if state.best_followup_agent:
                weights[state.best_followup_agent] = round(weights[state.best_followup_agent] * 1.1, 3)

        self.last_trust_weights = weights
        return weights

    def _infer_recovery_action(self, target: str, allowed_action_names: set[str]) -> str:
        if target == "api-gateway" and "scale_service" in allowed_action_names:
            return "scale_service"
        if "restart_service" in allowed_action_names:
            return "restart_service"
        return "scale_service"

    def _remember(self, action: CommanderAction) -> None:
        if action.action_type == "isolate_service" and action.target_service is not None:
            self.isolated_targets.add(action.target_service)
            self.primary_target = action.target_service
        elif action.action_type == "rollback_config" and action.target_service is not None:
            self.rolled_back_targets.add(action.target_service)
            self.primary_target = action.target_service
        elif action.action_type in {"restart_service", "scale_service"} and action.target_service is not None:
            self.recovered_targets.add(action.target_service)
            self.primary_target = action.target_service
        elif action.action_type == "investigate_service" and action.target_service is not None:
            self.primary_target = action.target_service
