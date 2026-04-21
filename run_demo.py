#!/usr/bin/env python3
"""Run a terminal-first SPECTRA demo with parallel specialist tables."""

from __future__ import annotations

import argparse

from rich.console import Console

from environments.pomir_env.env import POMIREnv
from runtime.terminal import render_feedback, render_intro, render_round, render_summary, save_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", default="training", choices=["training", "demo"])
    parser.add_argument("--output-dir", default="outputs/raw_traces")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    console = Console()
    env = POMIREnv(mode=args.mode)
    observation = env.reset(difficulty=args.difficulty, scenario_id=args.scenario_id, seed=args.seed)
    render_intro(console, observation)

    trace: dict[str, object] = {
        "mode": args.mode,
        "seed": args.seed,
        "scenario_id": observation.scenario_id,
        "scenario_name": observation.scenario_name,
        "difficulty": observation.difficulty,
        "rounds": [],
    }

    round_index = 1
    while not observation.done:
        decision = env.plan_next_action()
        render_round(
            console,
            round_index=round_index,
            observation=observation,
            execution=decision.execution,
            action=decision.action,
        )
        next_observation = env.step(decision.action)
        render_feedback(console, next_observation)
        trace["rounds"].append(
            {
                "round_index": round_index,
                "specialist_executions": [item.model_dump() for item in observation.specialist_executions],
                "commander_execution": decision.execution.model_dump(),
                "action": decision.action.model_dump(),
                "observation_after": next_observation.model_dump(),
            }
        )
        observation = next_observation
        round_index += 1

    render_summary(console, observation)
    if not args.no_save:
        trace["final_observation"] = observation.model_dump()
        path = save_trace(
            output_dir=args.output_dir,
            scenario_id=observation.scenario_id,
            seed=args.seed,
            payload=trace,
        )
        console.print(f"Saved trace to {path}")


if __name__ == "__main__":
    main()
