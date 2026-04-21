"""Probe commander trust weights on the hard scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.pomir_env.env import POMIREnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    traces: list[dict[str, object]] = []
    for episode in range(args.episodes):
        env = POMIREnv(mode="training")
        obs = env.reset(difficulty="hard", seed=400 + episode)
        while not obs.done:
            action = env.decide_next_action()
            traces.append(
                {
                    "episode": episode,
                    "step": env.master_env.state.tick,
                    "weights": env.commander.last_trust_weights,
                    "action": action.model_dump(),
                }
            )
            obs = env.step(action)
    print(json.dumps(traces, indent=2))


if __name__ == "__main__":
    main()
