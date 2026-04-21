"""R3 coordination reward."""

from __future__ import annotations

from agents.specialist_report import SpecialistReport
from environments.shared.state import CommanderAction, MasterSREState


def coordination_reward(
    action: CommanderAction,
    reports: list[SpecialistReport],
    state: MasterSREState,
    *,
    observation_mode: str,
) -> float:
    if observation_mode != "multi_agent" or not reports:
        return 0.0

    if action.action_type == "request_followup":
        return 0.15 if action.target_agent == state.best_followup_agent else 0.0

    agreeing = [report for report in reports if report.top_hypothesis_service == action.target_service]
    if len(agreeing) >= 2 and action.target_service == state.root_cause_service:
        return 0.15
    if len(agreeing) >= 2:
        return 0.05
    return 0.0
