"""Build a full-state prompt for single-agent commander runs."""

from __future__ import annotations

from environments.shared.state import MasterSREState


def _render_mapping(title: str, values: dict[str, float | str]) -> list[str]:
    lines = [title]
    for service, value in values.items():
        lines.append(f"  - {service}: {value}")
    return lines


def _render_flags(title: str, values: dict[str, bool]) -> list[str]:
    lines = [title]
    for name, value in values.items():
        lines.append(f"  - {name}: {str(value).lower()}")
    return lines


def build_single_agent_observation(
    state: MasterSREState,
    *,
    allowed_actions: list[str],
    hint_prefix: str | None = None,
) -> str:
    """Serialize the hidden world state for single-agent experiments."""

    sections: list[str] = []
    if hint_prefix:
        sections.extend(
            [
                "=== OPTIONAL CHEAT SHEET ===",
                hint_prefix.strip(),
                "",
            ]
        )

    sections.extend(
        [
            f"=== SPECTRA FULL-STATE INCIDENT VIEW — STEP {state.tick} / BUDGET {state.step_budget} ===",
            f"DIFFICULTY: {state.difficulty.upper()} | STAGE: {state.workflow_stage.upper()} | EPISODE: {state.episode_id}",
            f"Known trap: {state.common_trap}",
            f"Incident resolved: {str(state.incident_resolved).lower()}",
            f"Last action result: {state.last_action_result or 'none yet'}",
            f"Recent actions: {', '.join(state.action_history[-4:]) or 'none yet'}",
            "",
        ]
    )

    sections.extend(_render_flags("PROGRESS FLAGS:", state.progress_flags))
    sections.append("")

    unhealthy = [service for service, status in state.health.items() if status != "healthy"]
    sections.extend(
        [
            "ACTIVE HEALTH ALERTS:",
            *[f"  - {service}: {state.health[service]}" for service in unhealthy],
        ]
        if unhealthy
        else ["ACTIVE HEALTH ALERTS:", "  - none"]
    )
    sections.append("")

    sections.extend(_render_mapping("SERVICE HEALTH:", state.health))
    sections.append("")
    sections.extend(_render_mapping("CPU %:", {service: f"{value:.1f}" for service, value in state.cpu_pct.items()}))
    sections.append("")
    sections.extend(_render_mapping("MEM %:", {service: f"{value:.1f}" for service, value in state.mem_pct.items()}))
    sections.append("")
    sections.extend(
        _render_mapping("LATENCY MS:", {service: f"{value:.1f}" for service, value in state.latency_ms.items()})
    )
    sections.append("")
    sections.extend(
        _render_mapping("ERROR RATE:", {service: f"{value:.3f}" for service, value in state.error_rate.items()})
    )
    sections.append("")

    sections.extend(
        [
            "LOG SNAPSHOT:",
            *[f"  - {line}" for line in state.log_lines[:5]],
            "",
            "SECURITY SNAPSHOT:",
            *[f"  - {line}" for line in state.alert_strings[:5]],
            "",
            f"Auth failures: {state.auth_fail_count}",
            f"Suspicious sources: {', '.join(state.suspicious_ips[:5]) or 'none'}",
            f"Injection patterns: {', '.join(state.injection_patterns[:4]) or 'none'}",
            f"CVE/CWE flags: {', '.join(state.cve_flags[:4]) or 'none'}",
            "",
            "DECISION RULES:",
            "  - Do not submit_resolution until progress_flags.service_recovered is true.",
            "  - In recovery, choose restart_service or scale_service for the still-unhealthy origin service.",
            "  - If the last action was rejected or gave no progress, change the action family or target.",
            "",
            "AVAILABLE ACTIONS:",
            *[f"  - {action}" for action in allowed_actions],
            "SERVICES: api-gateway | database | cache | worker | auth_service",
        ]
    )
    return "\n".join(sections).strip()
