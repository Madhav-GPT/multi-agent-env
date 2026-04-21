from __future__ import annotations

import pytest

from environments.pomir_env.env import POMIREnv


@pytest.mark.parametrize(
    ("scenario_id", "max_steps"),
    [
        ("database_sqli_outage", 4),
        ("api_gateway_xss", 4),
        ("broken_auth_cascade", 5),
        ("worker_supply_chain_compromise", 5),
        ("cache_poisoning_campaign", 5),
    ],
)
def test_heuristic_commander_resolves_all_curated_scenarios(
    scenario_id: str,
    max_steps: int,
) -> None:
    env = POMIREnv(mode="training")
    observation = env.reset(scenario_id=scenario_id, seed=42)
    steps = 0
    actions: list[str] = []

    while not observation.done:
        decision = env.plan_next_action()
        actions.append(decision.action.rendered)
        observation = env.step(decision.action)
        steps += 1

    assert observation.incident_resolved is True
    assert steps <= max_steps
    assert observation.cumulative_reward > 1.0
    assert actions[-1] == "submit_resolution(summary)"


def test_broken_auth_scenario_requests_security_followup_first() -> None:
    env = POMIREnv(mode="training")
    env.reset(scenario_id="broken_auth_cascade", seed=42)

    decision = env.plan_next_action()

    assert decision.action.rendered == "request_followup(security)"
