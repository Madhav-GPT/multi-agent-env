"""R5 trust calibration reward."""

from __future__ import annotations

from agents.specialist_report import AgentID, SpecialistReport
from environments.shared.state import CommanderAction, MasterSREState


def trust_calibration_reward(
    action: CommanderAction,
    reports: list[SpecialistReport],
    state: MasterSREState,
    *,
    prior_actions: list[CommanderAction],
    observation_mode: str,
) -> float:
    if observation_mode != "multi_agent":
        return 0.0
    if state.decoy_agent is None or state.decoy_service is None or not state.incident_resolved:
        return 0.0
    decoy_report = next((report for report in reports if report.agent_id.value == state.decoy_agent), None)
    if decoy_report is None:
        return 0.0
    episode_actions = [*prior_actions, action]
    if any(
        candidate.target_service == state.root_cause_service and candidate.target_service != state.decoy_service
        for candidate in episode_actions
    ):
        return 0.05
    return 0.0
