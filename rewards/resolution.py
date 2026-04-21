"""R1 resolution reward."""

from __future__ import annotations

from environments.shared.state import MasterSREState


def resolution_reward(state: MasterSREState, done: bool) -> float:
    if done and state.incident_resolved and state.resolution_submitted:
        return 0.5
    if state.service_recovered:
        return 0.2
    return 0.0
