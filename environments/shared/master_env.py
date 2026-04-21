"""Shared full-state environment for SPECTRA."""

from __future__ import annotations

from typing import Any

from environments.infra_env.state import InfraState
from environments.log_env.state import LogState
from environments.sec_env.state import SecState

from .judge import DeterministicJudge
from .scenarios import build_master_state
from .state import CommanderAction, MasterSREState


class MasterSREEnv:
    """Owns the hidden state and applies commander actions."""

    def __init__(self) -> None:
        self._judge = DeterministicJudge()
        self._state = build_master_state(difficulty="easy")

    def reset(
        self,
        *,
        difficulty: str | None = None,
        scenario_id: str | None = None,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> MasterSREState:
        self._state = build_master_state(
            difficulty=difficulty,
            scenario_id=scenario_id,
            seed=seed,
            episode_id=episode_id,
        )
        return self.state

    def step(self, action: CommanderAction) -> tuple[MasterSREState, dict[str, Any]]:
        if self._state.step_budget <= 0 or self._state.incident_resolved:
            return self.state, {
                "message": "Episode already complete.",
                "done": True,
                "resolved": self._state.incident_resolved,
                "workflow_stage": self._state.workflow_stage,
            }

        self._state.tick += 1
        self._state.step_count = self._state.tick
        self._state.step_budget -= 1
        result = self._judge.evaluate(self._state, action)
        done = self._state.incident_resolved or self._state.step_budget <= 0
        info = result.model_dump()
        info["done"] = done
        info["steps_remaining"] = self._state.step_budget
        info["workflow_stage"] = self._state.workflow_stage
        info["progress_flags"] = dict(self._state.progress_flags)
        return self.state, info

    @property
    def state(self) -> MasterSREState:
        return self._state.model_copy(deep=True)

    def append_round_trace(self, round_payload: dict[str, Any]) -> None:
        self._state.round_history.append(round_payload)

    def clear_pending_followup(self) -> None:
        self._state.pending_followup_agent = None

    def mark_followup_seen(self, agent: str) -> None:
        self._state.followup_answers_seen[agent] = self._state.followup_answers_seen.get(agent, 0) + 1

    def get_partial_observation(self, agent_type: str) -> InfraState | LogState | SecState:
        payload = self._state
        followup_requested = payload.pending_followup_agent == agent_type
        if agent_type == "infra":
            return InfraState(
                scenario_id=payload.scenario_id,
                scenario_name=payload.objective.name,
                workflow_stage=payload.workflow_stage,
                cpu_pct=payload.cpu_pct,
                mem_pct=payload.mem_pct,
                latency_ms=payload.latency_ms,
                error_rate=payload.error_rate,
                health=payload.health,
                service_graph=payload.service_graph,
                tick=payload.tick,
                step_budget=payload.step_budget,
                difficulty=payload.difficulty,
                episode_id=payload.episode_id,
                followup_requested=followup_requested,
            )
        if agent_type == "log":
            return LogState(
                scenario_id=payload.scenario_id,
                scenario_name=payload.objective.name,
                workflow_stage=payload.workflow_stage,
                log_lines=payload.log_lines,
                error_types=payload.error_types,
                stack_traces=payload.stack_traces,
                query_patterns=payload.query_patterns,
                event_sequence=payload.event_sequence,
                tick=payload.tick,
                step_budget=payload.step_budget,
                difficulty=payload.difficulty,
                episode_id=payload.episode_id,
                followup_requested=followup_requested,
            )
        if agent_type == "security":
            return SecState(
                scenario_id=payload.scenario_id,
                scenario_name=payload.objective.name,
                workflow_stage=payload.workflow_stage,
                alert_strings=payload.alert_strings,
                auth_fail_count=payload.auth_fail_count,
                suspicious_ips=payload.suspicious_ips,
                injection_patterns=payload.injection_patterns,
                cve_flags=payload.cve_flags,
                tick=payload.tick,
                step_budget=payload.step_budget,
                difficulty=payload.difficulty,
                episode_id=payload.episode_id,
                followup_requested=followup_requested,
            )
        raise ValueError(f"Unknown agent_type: {agent_type}")
