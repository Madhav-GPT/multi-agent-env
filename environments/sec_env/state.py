"""Security-only observation schema."""

from __future__ import annotations

from pydantic import ConfigDict, Field

from environments.shared.openenv_compat import State
from environments.shared.state import Difficulty, WorkflowStage


class SecState(State):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    scenario_name: str
    workflow_stage: WorkflowStage
    alert_strings: list[str] = Field(default_factory=list)
    auth_fail_count: int = 0
    suspicious_ips: list[str] = Field(default_factory=list)
    injection_patterns: list[str] = Field(default_factory=list)
    cve_flags: list[str] = Field(default_factory=list)
    tick: int
    step_budget: int
    difficulty: Difficulty
    episode_id: str
    followup_requested: bool = False

