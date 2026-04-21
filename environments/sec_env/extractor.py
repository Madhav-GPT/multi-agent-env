"""Rule-based security specialist."""

from __future__ import annotations

from agents.specialist_report import AgentID, SEC_UNCERTAINTY_FLAGS, SpecialistReport
from environments.shared.scenarios import get_scenario

from .state import SecState


class SecExtractor:
    CVE_MAP = {
        "SQL_INJECTION": ("database", "SQL injection — CWE-89", "CWE-89", 0.95),
        "XSS_ATTEMPT": ("api-gateway", "Cross-site scripting — CWE-79", "CWE-79", 0.88),
        "BROKEN_AUTH": ("auth_service", "Broken authentication — CVE-2023-45812", "CVE-2023-45812", 0.96),
        "UNSIGNED_PACKAGE": ("worker", "Supply-chain compromise in worker release", "SLSA provenance violation", 0.94),
        "CACHE_POISONING": ("cache", "Cache poisoning via signed-header confusion", "CWE-349", 0.9),
    }

    def extract(self, state: SecState) -> SpecialistReport:
        scenario = get_scenario(state.scenario_id)
        followup_brief = scenario.followup_notes["security"] if state.followup_requested else None
        for alert in state.alert_strings:
            for signature, (service, cause, cve, confidence) in self.CVE_MAP.items():
                if signature in alert:
                    return SpecialistReport(
                        agent_id=AgentID.SECURITY,
                        observation_digest=(
                            f"Security telemetry is dominated by {signature} with {len(state.suspicious_ips)} suspicious sources."
                        ),
                        confidence=confidence,
                        top_hypothesis_service=service,
                        top_hypothesis_cause=cause,
                        supporting_evidence=[
                            f"Alert: {alert}",
                            f"Auth failures: {state.auth_fail_count}",
                            f"CVE/CWE: {cve}",
                            f"Patterns: {', '.join(state.injection_patterns[:2]) or 'N/A'}",
                        ],
                        recommended_action=f"isolate_service({service})",
                        uncertainty_flags=SEC_UNCERTAINTY_FLAGS,
                        severity="critical",
                        timestamp=state.tick,
                        followup_brief=followup_brief,
                    )
        return SpecialistReport(
            agent_id=AgentID.SECURITY,
            observation_digest=f"Security telemetry is noisy but not classifiable in scenario {state.scenario_name}.",
            confidence=0.45,
            top_hypothesis_service="database",
            top_hypothesis_cause="Generic security anomaly without a mapped signature",
            supporting_evidence=[
                f"Alerts observed: {len(state.alert_strings)}",
                f"Suspicious IPs: {len(state.suspicious_ips)}",
                "No mapped CVE/CWE signature detected",
            ],
            recommended_action="request_followup(security)",
            uncertainty_flags=SEC_UNCERTAINTY_FLAGS,
            severity="medium",
            timestamp=state.tick,
            followup_brief=followup_brief,
        )

