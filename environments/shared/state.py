"""Shared typed state for the rebuilt SPECTRA benchmark."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .openenv_compat import Action, State

ServiceName = Literal["api-gateway", "database", "cache", "worker", "auth_service"]
Difficulty = Literal["easy", "medium", "hard"]
HealthStatus = Literal["healthy", "degraded", "critical"]
WorkflowStage = Literal["triage", "containment", "remediation", "recovery", "retrospective", "done"]
CommanderActionType = Literal[
    "investigate_service",
    "request_followup",
    "isolate_service",
    "rollback_config",
    "scale_service",
    "restart_service",
    "submit_resolution",
]
AgentName = Literal["infra", "log", "security"]

SERVICE_NAMES: tuple[ServiceName, ...] = (
    "api-gateway",
    "database",
    "cache",
    "worker",
    "auth_service",
)
AGENT_NAMES: tuple[AgentName, ...] = ("infra", "log", "security")


class ScenarioObjective(BaseModel):
    """Scenario-specific hidden objective."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    name: str
    domain: str
    description: str
    difficulty: Difficulty
    root_cause_service: ServiceName
    root_cause_vector: str
    decoy_agent: AgentName | None = None
    decoy_service: ServiceName | None = None
    best_followup_agent: AgentName | None = None
    required_plan: list[tuple[CommanderActionType, ServiceName | None]] = Field(default_factory=list)
    common_trap: str
    specialist_signal_map: dict[AgentName, str] = Field(default_factory=dict)


class CommanderAction(Action):
    """Structured action taken by the commander."""

    model_config = ConfigDict(extra="forbid")

    action_type: CommanderActionType
    target_service: ServiceName | None = None
    target_agent: AgentName | None = None
    reasoning: str | None = None
    resolution_summary: str | None = None

    @model_validator(mode="after")
    def _validate_payload(self) -> "CommanderAction":
        if self.action_type in {
            "investigate_service",
            "isolate_service",
            "rollback_config",
            "scale_service",
            "restart_service",
        } and self.target_service is None:
            raise ValueError(f"target_service is required for {self.action_type}")
        if self.action_type == "request_followup" and self.target_agent is None:
            raise ValueError("target_agent is required for request_followup")
        if self.action_type == "submit_resolution" and not self.resolution_summary:
            raise ValueError("resolution_summary is required for submit_resolution")
        return self

    @property
    def rendered(self) -> str:
        if self.action_type == "request_followup":
            return f"{self.action_type}({self.target_agent})"
        if self.action_type == "submit_resolution":
            return f"{self.action_type}(summary)"
        return f"{self.action_type}({self.target_service})"


class ActionResult(State):
    """Judge output for one commander action."""

    model_config = ConfigDict(extra="forbid")

    message: str
    changed: bool
    resolved: bool
    targeted_root_cause: bool
    phase_advanced: bool = False
    trap: bool = False
    follow_up_used: bool = False


class SpecialistRoundExecution(BaseModel):
    """One specialist's execution artifact for a round."""

    model_config = ConfigDict(extra="forbid")

    agent: AgentName
    channel: Literal["metrics", "logs", "security"]
    mode: Literal["rule", "llm"]
    observation_digest: str
    raw_response: str
    latency_ms: float
    followup_applied: bool = False
    report: dict[str, Any]


class CommanderExecution(State):
    """One commander's decision artifact."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["heuristic", "llm"]
    raw_response: str
    latency_ms: float
    trust_weights: dict[str, float] = Field(default_factory=dict)
    action: dict[str, Any]


class MultiAgentRound(State):
    """Full parallel analysis round for one step."""

    model_config = ConfigDict(extra="forbid")

    round_index: int
    workflow_stage: WorkflowStage
    specialist_executions: list[SpecialistRoundExecution] = Field(default_factory=list)
    commander_execution: CommanderExecution | None = None
    environment_feedback: str = ""
    reward_breakdown: dict[str, float | str | int] = Field(default_factory=dict)


class MasterSREState(State):
    """The full hidden benchmark state."""

    model_config = ConfigDict(extra="forbid")

    objective: ScenarioObjective
    episode_id: str
    tick: int = 0
    step_budget: int = 0
    total_step_budget: int = 0
    workflow_stage: WorkflowStage = "triage"

    cpu_pct: dict[ServiceName, float]
    mem_pct: dict[ServiceName, float]
    latency_ms: dict[ServiceName, float]
    error_rate: dict[ServiceName, float]
    health: dict[ServiceName, HealthStatus]

    log_lines: list[str] = Field(default_factory=list)
    error_types: dict[str, int] = Field(default_factory=dict)
    stack_traces: list[str] = Field(default_factory=list)
    query_patterns: list[str] = Field(default_factory=list)
    event_sequence: list[dict[str, Any]] = Field(default_factory=list)

    alert_strings: list[str] = Field(default_factory=list)
    auth_fail_count: int = 0
    suspicious_ips: list[str] = Field(default_factory=list)
    injection_patterns: list[str] = Field(default_factory=list)
    cve_flags: list[str] = Field(default_factory=list)

    service_graph: dict[ServiceName, list[ServiceName]] = Field(default_factory=dict)
    causal_chain: list[str] = Field(default_factory=list)

    follow_up_count: int = 0
    pending_followup_agent: AgentName | None = None
    followup_history: list[AgentName] = Field(default_factory=list)
    followup_answers_seen: dict[AgentName, int] = Field(default_factory=dict)

    root_cause_confirmed: bool = False
    attack_isolated: bool = False
    config_remediated: bool = False
    service_recovered: bool = False
    resolution_submitted: bool = False
    incident_resolved: bool = False

    action_history: list[str] = Field(default_factory=list)
    last_action_result: str = ""
    resolution_history: list[str] = Field(default_factory=list)
    progress_flags: dict[str, bool] = Field(default_factory=dict)
    round_history: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def scenario_id(self) -> str:
        return self.objective.scenario_id

    @property
    def difficulty(self) -> Difficulty:
        return self.objective.difficulty

    @property
    def root_cause_service(self) -> ServiceName:
        return self.objective.root_cause_service

    @property
    def root_cause_vector(self) -> str:
        return self.objective.root_cause_vector

    @property
    def common_trap(self) -> str:
        return self.objective.common_trap

    @property
    def best_followup_agent(self) -> AgentName | None:
        return self.objective.best_followup_agent

    @property
    def decoy_agent(self) -> AgentName | None:
        return self.objective.decoy_agent

    @property
    def decoy_service(self) -> ServiceName | None:
        return self.objective.decoy_service

    @model_validator(mode="after")
    def _validate_service_keys(self) -> "MasterSREState":
        for mapping_name in ("cpu_pct", "mem_pct", "latency_ms", "error_rate", "health"):
            mapping = getattr(self, mapping_name)
            if set(mapping) != set(SERVICE_NAMES):
                raise ValueError(f"{mapping_name} must contain all services")
        return self

    def refresh_progress_flags(self) -> None:
        self.progress_flags = {
            "root_cause_confirmed": self.root_cause_confirmed,
            "attack_isolated": self.attack_isolated,
            "config_remediated": self.config_remediated,
            "service_recovered": self.service_recovered,
            "resolution_submitted": self.resolution_submitted,
            "incident_resolved": self.incident_resolved,
        }

