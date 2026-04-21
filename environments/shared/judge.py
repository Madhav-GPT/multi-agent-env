"""Deterministic judge for staged multi-agent SPECTRA incidents."""

from __future__ import annotations

from .causal_graph import impacted_services
from .state import ActionResult, CommanderAction, MasterSREState, ServiceName


class DeterministicJudge:
    """Apply commander actions against the hidden world state."""

    def evaluate(self, state: MasterSREState, action: CommanderAction) -> ActionResult:
        target = action.target_service
        root = state.root_cause_service
        targeted_root = target == root if target is not None else False
        trap = False
        changed = False
        resolved = state.incident_resolved
        phase_advanced = False

        if action.action_type == "request_followup":
            state.pending_followup_agent = action.target_agent
            state.follow_up_count += 1
            if action.target_agent is not None:
                state.followup_history.append(action.target_agent)
            changed = True
            message = f"Commander requested a deeper brief from {action.target_agent}."
        elif action.action_type == "investigate_service":
            changed = True
            if targeted_root:
                state.root_cause_confirmed = True
                message = f"Investigation confirms that {target} carries the primary failure signature."
            else:
                trap = True
                message = f"{target} is degraded, but the evidence still suggests it is a victim rather than the origin."
        elif action.action_type == "isolate_service":
            if state.workflow_stage not in {"triage", "containment"}:
                trap = True
                message = f"Containment is no longer the highest-value move at stage {state.workflow_stage}."
            elif targeted_root:
                state.attack_isolated = True
                changed = True
                message = f"Traffic to {target} isolated. The exploit path is now constrained."
                self._apply_containment_projection(state)
            else:
                trap = True
                message = f"Isolating {target} cuts a victim service, not the root cause."
        elif action.action_type == "rollback_config":
            if not state.attack_isolated:
                trap = True
                message = f"Rollback on {target} is premature before containment."
            elif targeted_root:
                state.config_remediated = True
                changed = True
                message = f"Rolled back the risky configuration on {target}."
                self._apply_remediation_projection(state)
            else:
                trap = True
                message = f"Rolling back {target} treats a symptom, not the origin service."
        elif action.action_type in {"restart_service", "scale_service"}:
            if not state.config_remediated:
                trap = True
                message = f"{action.action_type} on {target} is too early before remediation."
            elif targeted_root:
                required_recovery = self._required_recovery_action(state)
                if action.action_type == required_recovery:
                    state.service_recovered = True
                    changed = True
                    message = f"{action.action_type} completed on {target}; service stability is returning."
                    self._apply_recovery_projection(state)
                else:
                    trap = True
                    message = (
                        f"{action.action_type} helps only partially. This scenario needs {required_recovery} on {target}."
                    )
            else:
                trap = True
                message = f"{action.action_type} on {target} does not restore the compromised origin."
        elif action.action_type == "submit_resolution":
            if state.service_recovered:
                state.resolution_submitted = True
                state.incident_resolved = True
                changed = True
                resolved = True
                message = "Resolution submitted with a complete causal explanation."
            else:
                trap = True
                message = "Resolution is incomplete; the required containment and recovery chain is not finished."
        else:  # pragma: no cover - guarded by typing
            message = f"Unsupported action {action.action_type}."

        phase_advanced = self._update_workflow_stage(state)
        if state.incident_resolved:
            resolved = True

        if trap:
            self._apply_trap_projection(state, target)

        state.last_action_result = message
        state.action_history.append(action.rendered)
        state.event_sequence.append(
            {
                "service": target or action.target_agent or root,
                "message": message,
                "severity": "critical" if targeted_root or state.incident_resolved else "warning",
            }
        )
        state.refresh_progress_flags()

        return ActionResult(
            message=message,
            changed=changed,
            resolved=resolved,
            targeted_root_cause=targeted_root,
            phase_advanced=phase_advanced,
            trap=trap,
            follow_up_used=action.action_type == "request_followup",
        )

    def _required_recovery_action(self, state: MasterSREState) -> str:
        for action_type, service in state.objective.required_plan:
            if action_type in {"restart_service", "scale_service"} and service == state.root_cause_service:
                return action_type
        return "restart_service"

    def _update_workflow_stage(self, state: MasterSREState) -> bool:
        before = state.workflow_stage
        if state.resolution_submitted:
            state.workflow_stage = "done"
        elif state.service_recovered:
            state.workflow_stage = "retrospective"
        elif state.config_remediated:
            state.workflow_stage = "recovery"
        elif state.attack_isolated:
            state.workflow_stage = "remediation"
        elif state.root_cause_confirmed or state.follow_up_count > 0:
            state.workflow_stage = "containment"
        else:
            state.workflow_stage = "triage"
        return state.workflow_stage != before

    def _apply_trap_projection(self, state: MasterSREState, target: ServiceName | None) -> None:
        if target is None:
            return
        if target != state.root_cause_service:
            state.cpu_pct[target] = round(min(100.0, state.cpu_pct[target] + 2.0), 2)
            state.latency_ms[target] = round(state.latency_ms[target] * 1.04, 2)
            state.error_rate[target] = round(min(1.0, state.error_rate[target] + 0.02), 3)

    def _soften_service(self, state: MasterSREState, target: ServiceName, factor: float) -> None:
        state.cpu_pct[target] = round(state.cpu_pct[target] * factor, 2)
        state.mem_pct[target] = round(state.mem_pct[target] * factor, 2)
        state.latency_ms[target] = round(state.latency_ms[target] * factor, 2)
        state.error_rate[target] = round(max(0.0, state.error_rate[target] * factor), 3)
        if state.health[target] == "critical":
            state.health[target] = "degraded"

    def _apply_containment_projection(self, state: MasterSREState) -> None:
        impacted = impacted_services(state.root_cause_service)
        for service in impacted:
            self._soften_service(state, service, factor=0.75 if service == state.root_cause_service else 0.86)
        state.auth_fail_count = max(0, state.auth_fail_count // 2)
        state.alert_strings = state.alert_strings[:2]
        state.log_lines.insert(0, f"INFO: containment applied to {state.root_cause_service}")

    def _apply_remediation_projection(self, state: MasterSREState) -> None:
        self._soften_service(state, state.root_cause_service, factor=0.55)
        state.log_lines.insert(0, f"INFO: configuration rollback completed on {state.root_cause_service}")

    def _apply_recovery_projection(self, state: MasterSREState) -> None:
        impacted = impacted_services(state.root_cause_service)
        for service in impacted:
            state.cpu_pct[service] = 34.0 if service == state.root_cause_service else 28.0
            state.mem_pct[service] = 37.0 if service == state.root_cause_service else 31.0
            state.latency_ms[service] = 72.0 if service == state.root_cause_service else 56.0
            state.error_rate[service] = 0.01
            state.health[service] = "healthy"
        for service in state.health:
            if state.health[service] != "healthy":
                state.health[service] = "healthy"
                state.cpu_pct[service] = min(state.cpu_pct[service], 30.0)
                state.mem_pct[service] = min(state.mem_pct[service], 30.0)
                state.latency_ms[service] = min(state.latency_ms[service], 60.0)
                state.error_rate[service] = min(state.error_rate[service], 0.02)
        state.alert_strings = [f"RESOLVED: {state.root_cause_vector} path closed on {state.root_cause_service}"]
        state.log_lines.insert(0, f"INFO: services stabilized after fixing {state.root_cause_service}")

