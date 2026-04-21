"""Composite reward aggregation."""

from __future__ import annotations

from dataclasses import dataclass

from agents.specialist_report import SpecialistReport
from environments.shared.state import CommanderAction, MasterSREState

from .coordination import coordination_reward
from .efficiency import efficiency_reward
from .resolution import resolution_reward
from .root_cause import root_cause_reward
from .trust_calibration import trust_calibration_reward


@dataclass
class CompositeRewardResult:
    r1_resolution: float
    r2_root_cause: float
    r3_coordination: float
    r4_efficiency: float
    r5_trust: float
    penalty_wrong_target: float
    total: float
    episode_id: str
    difficulty: str
    step: int

    def as_dict(self) -> dict[str, float | str | int]:
        return {
            "r1_resolution": self.r1_resolution,
            "r2_root_cause": self.r2_root_cause,
            "r3_coordination": self.r3_coordination,
            "r4_efficiency": self.r4_efficiency,
            "r5_trust": self.r5_trust,
            "penalty_wrong_target": self.penalty_wrong_target,
            "total": self.total,
            "episode_id": self.episode_id,
            "difficulty": self.difficulty,
            "step": self.step,
        }


class CompositeRewardCalculator:
    def compute(
        self,
        *,
        action: CommanderAction,
        reports: list[SpecialistReport],
        state: MasterSREState,
        prior_actions: list[CommanderAction],
        done: bool,
        first_action: CommanderAction | None,
        observation_mode: str = "multi_agent",
    ) -> CompositeRewardResult:
        r1 = resolution_reward(state, done)
        r2 = root_cause_reward(first_action, state) if len(prior_actions) == 0 else 0.0
        r3 = coordination_reward(action, reports, state, observation_mode=observation_mode)
        r4 = efficiency_reward(state.step_budget, state.total_step_budget, state.incident_resolved)
        r5 = trust_calibration_reward(
            action,
            reports,
            state,
            prior_actions=prior_actions,
            observation_mode=observation_mode,
        )
        penalty = 0.0
        if action.target_service is not None and action.target_service != state.root_cause_service:
            penalty = -0.1
        total = round(max(0.0, min(1.0, r1 + r2 + r3 + r4 + r5 + penalty)), 3)
        return CompositeRewardResult(
            r1_resolution=r1,
            r2_root_cause=r2,
            r3_coordination=r3,
            r4_efficiency=r4,
            r5_trust=r5,
            penalty_wrong_target=penalty,
            total=total,
            episode_id=state.episode_id,
            difficulty=state.difficulty,
            step=state.tick,
        )
