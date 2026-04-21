"""R1 resolution reward."""

from __future__ import annotations

from environments.shared.state import MasterSREState


def resolution_reward(state: MasterSREState, done: bool) -> float:
    if done and state.incident_resolved and state.resolution_submitted:
        return 1.0
    if done and state.service_recovered:
        return 0.5
    return 0.0
