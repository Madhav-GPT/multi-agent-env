"""Log-only observation schema."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from environments.shared.openenv_compat import State
from environments.shared.state import Difficulty, WorkflowStage


class LogState(State):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    scenario_name: str
    workflow_stage: WorkflowStage
    log_lines: list[str] = Field(default_factory=list)
    error_types: dict[str, int] = Field(default_factory=dict)
    stack_traces: list[str] = Field(default_factory=list)
    query_patterns: list[str] = Field(default_factory=list)
    event_sequence: list[dict[str, Any]] = Field(default_factory=list)
    tick: int
    step_budget: int
    difficulty: Difficulty
    episode_id: str
    followup_requested: bool = False

