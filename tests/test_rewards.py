from __future__ import annotations

from environments.pomir_env.env import POMIREnv


def test_first_followup_step_gets_root_cause_credit_on_broken_auth() -> None:
    env = POMIREnv(mode="training")
    env.reset(scenario_id="broken_auth_cascade", seed=42)

    first_decision = env.plan_next_action()
    first_obs = env.step(first_decision.action)

    assert first_decision.action.rendered == "request_followup(security)"
    assert float(first_obs.reward_breakdown["r2_root_cause"]) == 0.4
    assert float(first_obs.reward_breakdown["r3_coordination"]) == 0.2


def test_final_hard_episode_earns_positive_trust_reward() -> None:
    env = POMIREnv(mode="training")
    observation = env.reset(scenario_id="worker_supply_chain_compromise", seed=42)

    while not observation.done:
        action = env.decide_next_action()
        observation = env.step(action)

    assert float(observation.reward_breakdown["r5_trust"]) > 0.0
