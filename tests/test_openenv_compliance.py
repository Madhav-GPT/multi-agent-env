from __future__ import annotations

from environments.infra_env.env import InfraEnv, InfraObserveAction
from environments.log_env.env import LogEnv, LogObserveAction
from environments.pomir_env.env import POMIREnv
from environments.sec_env.env import SecEnv, SecObserveAction
from environments.shared.state import CommanderAction


def test_partition_envs_expose_reset_step_state() -> None:
    infra = InfraEnv()
    infra_obs = infra.reset(difficulty="easy", seed=42)
    assert infra.state.tick == infra_obs.tick
    assert infra.step(InfraObserveAction()).channel == "infra"

    logs = LogEnv()
    log_obs = logs.reset(difficulty="medium", seed=42)
    assert logs.state.tick == log_obs.tick
    assert logs.step(LogObserveAction()).channel == "log"

    sec = SecEnv()
    sec_obs = sec.reset(difficulty="hard", seed=42)
    assert sec.state.tick == sec_obs.tick
    assert sec.step(SecObserveAction()).channel == "security"


def test_pomir_env_reset_step_state_and_reward_fields() -> None:
    env = POMIREnv(mode="training")
    obs = env.reset(scenario_id="database_sqli_outage", seed=42)

    assert env.state.prompt_text == obs.prompt_text
    assert obs.reward == 0.0
    assert obs.cumulative_reward == 0.0
    assert "isolate_service(service)" in obs.allowed_actions

    next_obs = env.step(
        CommanderAction(action_type="isolate_service", target_service="database")
    )

    assert isinstance(next_obs.prompt_text, str)
    assert next_obs.reward is not None
    assert next_obs.cumulative_reward > 0.0
