"""R2 root-cause targeting reward."""

from __future__ import annotations

from environments.shared.state import CommanderAction, MasterSREState


def root_cause_reward(first_action: CommanderAction | None, state: MasterSREState) -> float:
    if first_action is None:
        return 0.0
    if first_action.action_type == "request_followup":
        return 0.2 if first_action.target_agent == state.best_followup_agent else 0.0
    return 0.2 if first_action.target_service == state.root_cause_service else 0.0
