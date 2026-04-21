"""Rule-based infrastructure specialist."""

from __future__ import annotations

from agents.specialist_report import AgentID, INFRA_UNCERTAINTY_FLAGS, SpecialistReport
from environments.shared.scenarios import get_scenario

from .state import InfraState


def _classify_severity(score: float) -> str:
    if score >= 220:
        return "critical"
    if score >= 150:
        return "high"
    if score >= 95:
        return "medium"
    return "low"


class InfraExtractor:
    """Infer the most degraded service from metrics only."""

    def extract(self, state: InfraState) -> SpecialistReport:
        scores = {
            service: (
                state.cpu_pct[service]
                + (state.mem_pct[service] * 0.6)
                + (state.latency_ms[service] / 8.0)
                + (state.error_rate[service] * 10.0)
            )
            for service in state.cpu_pct
        }
        worst_service = max(scores, key=scores.get)
        worst_score = scores[worst_service]
        scenario = get_scenario(state.scenario_id)
        followup_brief = scenario.followup_notes["infra"] if state.followup_requested else None

        return SpecialistReport(
            agent_id=AgentID.INFRA,
            observation_digest=(
                f"Metrics focus on {worst_service}: CPU {state.cpu_pct[worst_service]:.0f}%, "
                f"latency {state.latency_ms[worst_service]:.0f}ms, error {state.error_rate[worst_service]:.2f}."
            ),
            confidence=round(min(0.93, max(0.55, worst_score / 250.0)), 2),
            top_hypothesis_service=worst_service,
            top_hypothesis_cause=f"{worst_service} is the most degraded service by infrastructure telemetry",
            supporting_evidence=[
                f"CPU utilization: {state.cpu_pct[worst_service]:.1f}%",
                f"Memory pressure: {state.mem_pct[worst_service]:.1f}%",
                f"Request latency: {state.latency_ms[worst_service]:.0f}ms",
                f"Error rate: {state.error_rate[worst_service]:.2f}",
            ],
            recommended_action=f"isolate_service({worst_service})",
            uncertainty_flags=INFRA_UNCERTAINTY_FLAGS,
            severity=_classify_severity(worst_score),
            timestamp=state.tick,
            followup_brief=followup_brief,
        )

