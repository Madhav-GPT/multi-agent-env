"""Specialist report protocol and execution artifacts."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from environments.shared.state import AgentName, ServiceName, SpecialistRoundExecution


class AgentID(str, Enum):
    INFRA = "infra"
    LOG = "log"
    SECURITY = "security"

    @property
    def channel(self) -> str:
        return {
            AgentID.INFRA: "metrics",
            AgentID.LOG: "logs",
            AgentID.SECURITY: "security",
        }[self]


INFRA_UNCERTAINTY_FLAGS = [
    "Cannot determine exploit type from metrics alone",
    "Cannot verify application-layer evidence without logs",
    "Cannot confirm whether degraded services are victims or origin services",
]

LOG_UNCERTAINTY_FLAGS = [
    "Cannot confirm system-wide saturation without metrics",
    "Cannot verify external threat signatures without security alerts",
    "Cannot see whether the suspected service is the primary blast-radius source",
]

SEC_UNCERTAINTY_FLAGS = [
    "Cannot confirm the worst runtime bottleneck without metrics",
    "Cannot verify application-layer exception chains without logs",
    "Cannot prove recovery progress from security telemetry alone",
]


class SpecialistReport(BaseModel):
    """Typed report emitted by a specialist."""

    model_config = ConfigDict(extra="forbid")

    agent_id: AgentID
    observation_digest: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_hypothesis_service: ServiceName
    top_hypothesis_cause: str
    supporting_evidence: list[str] = Field(default_factory=list, max_length=4)
    recommended_action: str
    uncertainty_flags: list[str] = Field(default_factory=list)
    severity: Literal["critical", "high", "medium", "low"]
    timestamp: int
    followup_brief: str | None = None

    def to_execution(
        self,
        *,
        mode: Literal["rule", "llm"],
        raw_response: str,
        latency_ms: float,
        followup_applied: bool,
    ) -> SpecialistRoundExecution:
        return SpecialistRoundExecution(
            agent=self.agent_id.value,  # type: ignore[arg-type]
            channel=self.agent_id.channel,  # type: ignore[arg-type]
            mode=mode,
            observation_digest=self.observation_digest,
            raw_response=raw_response,
            latency_ms=round(latency_ms, 2),
            followup_applied=followup_applied,
            report=self.model_dump(),
        )


def serialize_reports(
    reports: list[SpecialistReport],
    *,
    tick: int,
    step_budget: int,
    difficulty: str,
    episode_id: str,
    workflow_stage: str,
    common_trap: str,
    allowed_actions: list[str],
) -> str:
    sections: list[str] = [
        f"=== SPECTRA INCIDENT REPORT — STEP {tick} / BUDGET {step_budget} ===",
        f"DIFFICULTY: {difficulty.upper()} | STAGE: {workflow_stage.upper()} | EPISODE: {episode_id}",
        f"Known trap: {common_trap}",
        "",
    ]
    ordered = {
        AgentID.INFRA: "INFRA SPECIALIST",
        AgentID.LOG: "LOG SPECIALIST",
        AgentID.SECURITY: "SECURITY SPECIALIST",
    }
    for report in reports:
        sections.extend(
            [
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"[{ordered[report.agent_id]}] confidence: {report.confidence:.0%} | severity: {report.severity}",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"Observation: {report.observation_digest}",
                f"Hypothesis: {report.top_hypothesis_cause}",
                f"Target service: {report.top_hypothesis_service}",
                "Evidence:",
            ]
        )
        for evidence in report.supporting_evidence:
            sections.append(f"  - {evidence}")
        if report.followup_brief:
            sections.append(f"Follow-up detail: {report.followup_brief}")
        sections.append(f"Recommended: {report.recommended_action}")
        sections.append(f"Cannot determine: {' | '.join(report.uncertainty_flags)}")
        sections.append("")

    sections.extend(
        [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "AVAILABLE ACTIONS:",
            *[f"  {action}" for action in allowed_actions],
            "SERVICES: api-gateway | database | cache | worker | auth_service",
            "FOLLOW-UP AGENTS: infra | log | security",
        ]
    )
    return "\n".join(sections).strip()


def agent_name(value: AgentID | AgentName) -> AgentName:
    return value.value if isinstance(value, AgentID) else value

