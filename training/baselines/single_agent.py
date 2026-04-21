"""Condition B baseline: single-agent full-state heuristic."""

from __future__ import annotations

from environments.shared.state import CommanderAction, MasterSREState


class SingleAgentCommander:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.isolated_targets: set[str] = set()
        self.rolled_back_targets: set[str] = set()
        self.followups_used: bool = False

    def act_from_state(self, state: MasterSREState) -> CommanderAction:
        if "CVE-2023-45812" in " ".join(state.cve_flags):
            target = "auth_service"
            preferred_followup = "security"
        elif "SLSA" in " ".join(state.cve_flags):
            target = "worker"
            preferred_followup = "log"
        elif any("CACHE_POISONING" in alert for alert in state.alert_strings):
            target = "cache"
            preferred_followup = "infra"
        elif any("XSS" in alert for alert in state.alert_strings):
            target = "api-gateway"
            preferred_followup = "log"
        elif any("SQL" in alert for alert in state.alert_strings):
            target = "database"
            preferred_followup = "security"
        else:
            target = max(state.error_rate, key=state.error_rate.get)
            preferred_followup = "infra"

        if state.workflow_stage == "triage" and not self.followups_used:
            self.followups_used = True
            return CommanderAction(action_type="request_followup", target_agent=preferred_followup)

        if target in self.isolated_targets and target not in self.rolled_back_targets:
            action = CommanderAction(action_type="rollback_config", target_service=target)
            self.rolled_back_targets.add(target)
            return action

        if state.workflow_stage == "recovery":
            recovery_action = next(
                (
                    action_type
                    for action_type, service in state.objective.required_plan
                    if action_type in {"restart_service", "scale_service"} and service == target
                ),
                "restart_service",
            )
            return CommanderAction(action_type=recovery_action, target_service=target)

        if state.workflow_stage == "retrospective":
            return CommanderAction(
                action_type="submit_resolution",
                resolution_summary=f"Resolved {state.root_cause_vector} on {state.root_cause_service}.",
            )

        action = CommanderAction(action_type="isolate_service", target_service=target)
        self.isolated_targets.add(target)
        return action
