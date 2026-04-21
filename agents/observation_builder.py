"""Build the commander's observation string."""

from __future__ import annotations

from agents.specialist_report import SpecialistReport, serialize_reports
from environments.shared.state import MasterSREState


def build_commander_observation(
    reports: list[SpecialistReport],
    state: MasterSREState,
    *,
    allowed_actions: list[str],
) -> str:
    return serialize_reports(
        reports,
        tick=state.tick,
        step_budget=state.step_budget,
        difficulty=state.difficulty,
        episode_id=state.episode_id,
        workflow_stage=state.workflow_stage,
        common_trap=state.common_trap,
        allowed_actions=allowed_actions,
    )
