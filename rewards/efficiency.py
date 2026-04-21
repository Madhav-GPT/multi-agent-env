"""R4 efficiency reward."""

from __future__ import annotations


def efficiency_reward(steps_remaining: int, total_step_budget: int, resolved: bool) -> float:
    if not resolved or total_step_budget <= 0:
        return 0.0
    return round(0.1 * (steps_remaining / total_step_budget), 3)
