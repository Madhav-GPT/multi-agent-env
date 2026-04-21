"""Rule-based log specialist."""

from __future__ import annotations

from agents.specialist_report import AgentID, LOG_UNCERTAINTY_FLAGS, SpecialistReport
from environments.shared.scenarios import get_scenario

from .state import LogState


class LogExtractor:
    ATTACK_PATTERNS = {
        "SQLSyntaxError": ("database", "SQL injection attack on database query parser", 0.91),
        "XSSPayloadDetected": ("api-gateway", "Reflected XSS entering through gateway rendering", 0.89),
        "JWTVerificationFailed": ("auth_service", "JWT token forgery causing authentication failure", 0.93),
        "RetryLoop": ("worker", "Compromised worker release is replaying poisoned jobs", 0.9),
        "CacheDeserializeFallback": ("api-gateway", "Gateway appears unstable because poisoned cache variants are surfacing", 0.74),
    }

    def extract(self, state: LogState) -> SpecialistReport:
        dominant_error = max(state.error_types, key=state.error_types.get)
        service, cause, confidence = self.ATTACK_PATTERNS.get(
            dominant_error,
            ("database", "Unclassified but severe application error", 0.45),
        )
        scenario = get_scenario(state.scenario_id)
        if scenario.decoy_agent == "log" and scenario.decoy_service is not None:
            service = scenario.decoy_service
            cause = f"Log evidence makes {service} look like the origin service"
            confidence = max(confidence, 0.78)
        followup_brief = scenario.followup_notes["log"] if state.followup_requested else None
        return SpecialistReport(
            agent_id=AgentID.LOG,
            observation_digest=(
                f"Dominant log pattern is {dominant_error} with {state.error_types[dominant_error]} events "
                f"in scenario {state.scenario_name}."
            ),
            confidence=confidence,
            top_hypothesis_service=service,
            top_hypothesis_cause=cause,
            supporting_evidence=[
                f"{dominant_error}: {state.error_types[dominant_error]} events in the latest window",
                f"Top query pattern: {state.query_patterns[0] if state.query_patterns else 'N/A'}",
                f"Stack trace: {state.stack_traces[0] if state.stack_traces else 'N/A'}",
                f"Most recent event: {state.event_sequence[-1]['message'] if state.event_sequence else 'N/A'}",
            ],
            recommended_action=f"rollback_config({service})",
            uncertainty_flags=LOG_UNCERTAINTY_FLAGS,
            severity="critical" if state.error_types[dominant_error] > 500 else "high",
            timestamp=state.tick,
            followup_brief=followup_brief,
        )

