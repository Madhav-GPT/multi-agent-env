from __future__ import annotations

from inference import safe_parse_action


def test_safe_parse_action_uses_valid_example_for_placeholder_service() -> None:
    action = safe_parse_action(
        '{"action_type":"isolate_service","target_service":"service"}',
        allowed_actions=["isolate_service(service)", "restart_service(service)"],
        valid_action_example={"action_type": "isolate_service", "target_service": "auth_service"},
    )

    assert action.action_type == "isolate_service"
    assert action.target_service == "auth_service"


def test_safe_parse_action_salvages_followup_agent_from_example() -> None:
    action = safe_parse_action(
        "please request_followup(agent)",
        allowed_actions=["request_followup(agent)"],
        valid_action_example={"action_type": "request_followup", "target_agent": "security"},
    )

    assert action.action_type == "request_followup"
    assert action.target_agent == "security"
