"""Replay a deterministic episode trajectory."""

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
    parser.add_argument("--difficulty", default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = POMIREnv(mode="training")
    trajectory = env.run_episode(difficulty=args.difficulty, seed=args.seed)
    print(json.dumps(trajectory, indent=2))


if __name__ == "__main__":
    main()
