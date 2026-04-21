from __future__ import annotations

from environments.pomir_env.env import POMIREnv


def test_first_followup_step_gets_root_cause_credit_on_broken_auth() -> None:
    env = POMIREnv(mode="training")
    env.reset(scenario_id="broken_auth_cascade", seed=42)

    first_decision = env.plan_next_action()
    first_obs = env.step(first_decision.action)

    assert first_decision.action.rendered == "request_followup(security)"
    assert float(first_obs.reward_breakdown["r2_root_cause"]) == 0.2
    assert float(first_obs.reward_breakdown["r3_coordination"]) == 0.15
    assert 0.0 <= float(first_obs.reward_breakdown["total"]) <= 1.0


def test_final_hard_episode_earns_positive_trust_reward() -> None:
    env = POMIREnv(mode="training")
    observation = env.reset(scenario_id="worker_supply_chain_compromise", seed=42)

    while not observation.done:
        action = env.decide_next_action()
        observation = env.step(action)

    assert float(observation.reward_breakdown["r5_trust"]) > 0.0


def test_single_agent_mode_disables_coordination_only_components() -> None:
    env = POMIREnv(mode="training", observation_mode="single_agent")
    observation = env.reset(scenario_id="broken_auth_cascade", seed=42, observation_mode="single_agent")

    while not observation.done:
        action = env.decide_next_action()
        observation = env.step(action)

    assert observation.observation_mode == "single_agent"
    assert float(observation.reward_breakdown["r3_coordination"]) == 0.0
    assert float(observation.reward_breakdown["r5_trust"]) == 0.0
