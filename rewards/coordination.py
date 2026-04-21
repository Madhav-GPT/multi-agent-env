"""R3 coordination reward."""

from __future__ import annotations

from agents.specialist_report import SpecialistReport
from environments.shared.state import CommanderAction, MasterSREState


def coordination_reward(
    action: CommanderAction,
    reports: list[SpecialistReport],
    state: MasterSREState,
    prior_actions: list[CommanderAction],
    episode_succeeded: bool,
) -> float:
    if action.action_type == "request_followup":
        if action.target_agent == state.best_followup_agent:
            return 0.2
        return -0.05

    agreeing = [report for report in reports if report.top_hypothesis_service == action.target_service]
    score = 0.0
    if len(agreeing) >= 2:
        score += 0.3
    non_root_actions = [
        item
        for item in prior_actions
        if item.action_type != "request_followup" and item.target_service != state.root_cause_service
    ]
    if action.target_service == state.root_cause_service and not non_root_actions:
        score += 0.2
    if len(agreeing) == 0 and not episode_succeeded:
        score -= 0.2
    return round(score, 3)
