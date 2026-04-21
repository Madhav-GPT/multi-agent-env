"""Shared helpers for A/B/C condition comparisons."""

from __future__ import annotations

from environments.pomir_env.env import POMIREnv
from training.baselines.random_commander import RandomCommander


def _difficulty_for_episode(index: int, requested: str) -> str:
    if requested != "mixed":
        return requested
    cycle = ("easy", "medium", "hard")
    return cycle[index % len(cycle)]


def run_condition(condition: str, steps: int, difficulty: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for episode in range(steps):
        if condition == "B":
            env = POMIREnv(
                mode="deterministic",
                specialist_mode="deterministic",
                observation_mode="single_agent",
            )
        else:
            env = POMIREnv(
                mode="deterministic",
                specialist_mode="deterministic",
                observation_mode="multi_agent",
            )
        obs = env.reset(difficulty=_difficulty_for_episode(episode, difficulty), seed=42 + episode)
        random_commander = RandomCommander(seed=100 + episode)
        random_commander.reset()
        total_reward = 0.0
        total_component_rewards = {
            "r1_resolution": 0.0,
            "r2_root_cause": 0.0,
            "r3_coordination": 0.0,
            "r4_efficiency": 0.0,
            "r5_trust": 0.0,
            "penalty_wrong_target": 0.0,
        }
        actions: list[dict[str, str]] = []

        while not obs.done:
            if condition == "A":
                action = random_commander.act(env.allowed_action_strings)
            else:
                action = env.decide_next_action()
            obs = env.step(action)
            actions.append(action.model_dump())
            total_reward += float(obs.reward_breakdown.get("total", 0.0))
            for key in total_component_rewards:
                total_component_rewards[key] += float(obs.reward_breakdown.get(key, 0.0))

        records.append(
            {
                "episode": episode,
                "condition": condition,
                "difficulty": env.master_env.state.difficulty,
                "scenario_id": env.master_env.state.scenario_id,
                "success": obs.incident_resolved,
                "steps": len(actions),
                "total_reward": round(total_reward, 3),
                "actions": actions,
                **{key: round(value, 3) for key, value in total_component_rewards.items()},
            }
        )
    return records
