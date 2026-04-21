"""Evaluation metrics."""

from __future__ import annotations


def summarize(records: list[dict[str, object]]) -> dict[str, float]:
    if not records:
        return {"episodes": 0.0, "success_rate": 0.0, "mean_total_reward": 0.0}
    episodes = len(records)
    successes = sum(1 for record in records if record.get("success"))
    reward_sum = sum(float(record.get("total_reward", 0.0)) for record in records)
    return {
        "episodes": float(episodes),
        "success_rate": round(successes / episodes, 3),
        "mean_total_reward": round(reward_sum / episodes, 3),
    }

