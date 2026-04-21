"""Infrastructure-only observation schema."""

from __future__ import annotations

from pydantic import ConfigDict

from environments.shared.openenv_compat import State
from environments.shared.state import Difficulty, HealthStatus, ServiceName, WorkflowStage


class InfraState(State):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    scenario_name: str
    workflow_stage: WorkflowStage
    cpu_pct: dict[ServiceName, float]
    mem_pct: dict[ServiceName, float]
    latency_ms: dict[ServiceName, float]
    error_rate: dict[ServiceName, float]
    health: dict[ServiceName, HealthStatus]
    service_graph: dict[ServiceName, list[ServiceName]]
    tick: int
    step_budget: int
    difficulty: Difficulty
    episode_id: str
    followup_requested: bool = False

