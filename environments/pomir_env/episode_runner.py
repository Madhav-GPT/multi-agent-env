"""Episode runner helpers for training and demos."""

from __future__ import annotations

from dataclasses import dataclass, field

from environments.shared.state import CommanderAction

from .env import POMIREnv


@dataclass
class EpisodeRecord:
    success: bool
    steps: int
    total_reward: float
    actions: list[CommanderAction] = field(default_factory=list)


def run_episode(env: POMIREnv, difficulty: str, seed: int | None = None) -> EpisodeRecord:
    observation = env.reset(difficulty=difficulty, seed=seed)
    actions: list[CommanderAction] = []
    total_reward = 0.0
    while not observation.done:
        action = env.decide_next_action()
        actions.append(action)
        observation = env.step(action)
        total_reward += float(observation.reward_breakdown.get("total", 0.0))
    return EpisodeRecord(
        success=observation.incident_resolved,
        steps=len(actions),
        total_reward=total_reward,
        actions=actions,
    )

