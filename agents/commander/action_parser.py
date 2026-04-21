"""Parse model outputs into CommanderAction objects."""

from __future__ import annotations

import json
import re

from environments.shared.state import CommanderAction

SERVICE_ACTION_PATTERN = re.compile(
    r"(investigate_service|isolate_service|rollback_config|scale_service|restart_service)\((api-gateway|database|cache|worker|auth_service)\)"
)
FOLLOWUP_PATTERN = re.compile(r"request_followup\((infra|log|security)\)")


def parse_action(text: str) -> CommanderAction:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("```", 2)[1]
        candidate = candidate.replace("json", "", 1).strip()
    if "<action>" in candidate and "</action>" in candidate:
        candidate = candidate.split("<action>", 1)[1].split("</action>", 1)[0].strip()

    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            return CommanderAction(**payload)
    except Exception:
        pass

    match = SERVICE_ACTION_PATTERN.search(candidate)
    if match:
        action_type, target_service = match.groups()
        return CommanderAction(action_type=action_type, target_service=target_service)

    match = FOLLOWUP_PATTERN.search(candidate)
    if match:
        return CommanderAction(action_type="request_followup", target_agent=match.group(1))

    if "submit_resolution" in candidate:
        return CommanderAction(
            action_type="submit_resolution",
            resolution_summary="Commander reports the incident as resolved.",
        )

    raise ValueError(f"Could not parse commander action from: {text!r}")

