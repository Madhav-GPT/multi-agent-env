"""Runnable comparison harness and lightweight GRPO scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.commander.commander import Commander
from environments.pomir_env.env import POMIREnv
from training.baselines.random_commander import RandomCommander
from training.baselines.single_agent import SingleAgentCommander


def _difficulty_for_episode(index: int, requested: str) -> str:
    if requested != "mixed":
        return requested
    cycle = ("easy", "medium", "hard")
    return cycle[index % len(cycle)]


def run_condition(condition: str, steps: int, difficulty: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for episode in range(steps):
        env = POMIREnv(mode="training", commander=Commander())
        obs = env.reset(difficulty=_difficulty_for_episode(episode, difficulty), seed=42 + episode)
        random_commander = RandomCommander(seed=100 + episode)
        single_agent = SingleAgentCommander()
        random_commander.reset()
        single_agent.reset()
        total_reward = 0.0
        total_component_rewards = {
            "r1_resolution": 0.0,
            "r2_root_cause": 0.0,
            "r3_coordination": 0.0,
            "r4_efficiency": 0.0,
            "r5_trust": 0.0,
        }
        actions: list[dict[str, str]] = []

        while not obs.done:
            if condition == "A":
                action = random_commander.act(env.allowed_action_strings)
            elif condition == "B":
                action = single_agent.act_from_state(env.master_env.state)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="C", choices=["A", "B", "C"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--difficulty", default="mixed", choices=["easy", "medium", "hard", "mixed"])
    parser.add_argument("--output-dir", default="outputs/grpo_runs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = run_condition(args.condition, args.steps, args.difficulty)
    output_path = output_dir / f"condition_{args.condition.lower()}_{args.steps}.json"
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "episodes": len(records)}, indent=2))


if __name__ == "__main__":
    main()
