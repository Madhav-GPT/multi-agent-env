"""Dataset helpers for commander training."""

from __future__ import annotations

from dataclasses import dataclass

from environments.pomir_env.env import POMIREnv


@dataclass
class PromptRecord:
    prompt: str
    response: str
    reward: float
    difficulty: str
    scenario_id: str


def build_prompt_records(
    env: POMIREnv,
    *,
    episodes: int,
    difficulty: str = "easy",
    seed_start: int = 42,
) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    for offset in range(episodes):
        observation = env.reset(difficulty=difficulty, seed=seed_start + offset)
        while not observation.done:
            action = env.decide_next_action()
            next_observation = env.step(action)
            records.append(
                PromptRecord(
                    prompt=observation.prompt_text,
                    response=action.rendered,
                    reward=float(next_observation.reward_breakdown.get("total", 0.0)),
                    difficulty=observation.difficulty,
                    scenario_id=observation.scenario_id,
                )
            )
            observation = next_observation
    return records

