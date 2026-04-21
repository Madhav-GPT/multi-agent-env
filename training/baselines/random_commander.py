"""Condition A baseline: random commander."""

from __future__ import annotations

import random

from environments.shared.state import CommanderAction, SERVICE_NAMES


class RandomCommander:
    ACTIONS = (
        "investigate_service",
        "request_followup",
        "isolate_service",
        "rollback_config",
        "scale_service",
        "restart_service",
        "submit_resolution",
    )

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def reset(self) -> None:
        return None

    def act(self, allowed_actions: list[str] | None = None) -> CommanderAction:
        action_names = [action.split("(", 1)[0] for action in allowed_actions] if allowed_actions else list(self.ACTIONS)
        action_type = self._rng.choice(action_names)
        if action_type == "request_followup":
            return CommanderAction(
                action_type=action_type,
                target_agent=self._rng.choice(("infra", "log", "security")),
                reasoning="Random baseline action.",
            )
        if action_type == "submit_resolution":
            return CommanderAction(
                action_type=action_type,
                resolution_summary="Randomly assuming resolution.",
                reasoning="Random baseline action.",
            )
        return CommanderAction(
            action_type=action_type,
            target_service=self._rng.choice(SERVICE_NAMES),
            reasoning="Random baseline action.",
        )
