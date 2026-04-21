"""R5 trust calibration reward."""

from __future__ import annotations

from agents.specialist_report import AgentID, SpecialistReport
from environments.shared.state import CommanderAction, MasterSREState


def trust_calibration_reward(
    action: CommanderAction,
    reports: list[SpecialistReport],
    state: MasterSREState,
    prior_actions: list[CommanderAction],
    episode_succeeded: bool,
) -> float:
    if state.decoy_agent is None or state.decoy_service is None or not episode_succeeded:
        return 0.0
    decoy_report = next((report for report in reports if report.agent_id.value == state.decoy_agent), None)
    episode_actions = [*prior_actions, action]
    score = 0.0
    if any(
        candidate.action_type != "request_followup"
        and candidate.target_service == state.root_cause_service
        and candidate.target_service != state.decoy_service
        for candidate in episode_actions
    ):
        score += 0.2
    if any(
        candidate.action_type == "request_followup"
        and candidate.target_agent == state.best_followup_agent
        for candidate in episode_actions
    ):
        score += 0.1
    if decoy_report is None:
        return round(score, 3)
    return round(score, 3)
