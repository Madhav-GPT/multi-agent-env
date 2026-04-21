#!/usr/bin/env python3
"""Run SPECTRA locally or against the server and save a full execution trace."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Protocol

import httpx
from rich.console import Console

from environments.pomir_env.env import POMIREnv, POMIRObservation
from environments.shared.state import CommanderAction, CommanderExecution
from runtime.terminal import render_feedback, render_intro, render_round, render_summary, save_trace


def log_start(*, scenario: str, runtime: str, seed: int, mode: str) -> None:
    print(f"[START] scenario={scenario} runtime={runtime} seed={seed} mode={mode}", flush=True)


def log_round(*, round_index: int, stage: str, actions: list[str]) -> None:
    allowed = ",".join(actions)
    print(f"[ROUND] round={round_index} stage={stage} allowed={allowed}", flush=True)


def log_step(*, round_index: int, action: CommanderAction, reward: float, done: bool, stage: str) -> None:
    print(
        f"[STEP] round={round_index} action={action.rendered} reward={reward:.3f} "
        f"done={str(done).lower()} stage={stage}",
        flush=True,
    )


def log_end(*, scenario: str, success: bool, rounds: int, reward: float, trace_path: str) -> None:
    print(
        f"[END] scenario={scenario} success={str(success).lower()} rounds={rounds} "
        f"total_reward={reward:.3f} trace={trace_path}",
        flush=True,
    )


@dataclass
class PlannedAction:
    action: CommanderAction
    execution: CommanderExecution


class RuntimeClient(Protocol):
    def reset(self, *, difficulty: str, scenario_id: str | None, seed: int) -> POMIRObservation:
        ...

    def plan(self) -> PlannedAction:
        ...

    def step(self, action: CommanderAction) -> POMIRObservation:
        ...

    def close(self) -> None:
        ...


class LocalRuntime:
    def __init__(self, mode: str) -> None:
        self.env = POMIREnv(mode=mode)

    def reset(self, *, difficulty: str, scenario_id: str | None, seed: int) -> POMIRObservation:
        return self.env.reset(difficulty=difficulty, scenario_id=scenario_id, seed=seed)

    def plan(self) -> PlannedAction:
        decision = self.env.plan_next_action()
        return PlannedAction(action=decision.action, execution=decision.execution)

    def step(self, action: CommanderAction) -> POMIRObservation:
        return self.env.step(action)

    def close(self) -> None:
        self.env.close()


class RemoteRuntime:
    def __init__(self, *, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=30.0)

    def _hydrate_observation(self, payload: dict) -> POMIRObservation:
        observation = dict(payload["observation"])
        observation["reward"] = payload.get("reward")
        observation["done"] = payload.get("done", observation.get("done", False))
        return POMIRObservation(**observation)

    def reset(self, *, difficulty: str, scenario_id: str | None, seed: int) -> POMIRObservation:
        response = self.client.post(
            "/reset",
            json={
                "difficulty": difficulty,
                "scenario_id": scenario_id,
                "seed": seed,
            },
        )
        response.raise_for_status()
        return self._hydrate_observation(response.json())

    def plan(self) -> PlannedAction:
        response = self.client.post("/plan")
        response.raise_for_status()
        payload = response.json()
        return PlannedAction(
            action=CommanderAction(**payload["action"]),
            execution=CommanderExecution(**payload["execution"]),
        )

    def step(self, action: CommanderAction) -> POMIRObservation:
        response = self.client.post("/step", json={"action": action.model_dump()})
        response.raise_for_status()
        return self._hydrate_observation(response.json())

    def close(self) -> None:
        self.client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runtime", default="local", choices=["local", "remote"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", default="training", choices=["training", "demo"])
    parser.add_argument("--output-dir", default="outputs/raw_traces")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    console = Console()
    runtime: RuntimeClient = LocalRuntime(args.mode) if args.runtime == "local" else RemoteRuntime(base_url=args.base_url)

    try:
        observation = runtime.reset(difficulty=args.difficulty, scenario_id=args.scenario_id, seed=args.seed)
        if args.pretty:
            render_intro(console, observation)
        log_start(
            scenario=observation.scenario_id,
            runtime=args.runtime,
            seed=args.seed,
            mode=args.mode,
        )

        trace: dict[str, object] = {
            "runtime": args.runtime,
            "mode": args.mode,
            "seed": args.seed,
            "scenario_id": observation.scenario_id,
            "scenario_name": observation.scenario_name,
            "difficulty": observation.difficulty,
            "rounds": [],
        }

        round_index = 1
        while not observation.done:
            planned = runtime.plan()
            log_round(
                round_index=round_index,
                stage=observation.workflow_stage,
                actions=observation.allowed_actions,
            )
            if args.pretty:
                render_round(
                    console,
                    round_index=round_index,
                    observation=observation,
                    execution=planned.execution,
                    action=planned.action,
                )
            next_observation = runtime.step(planned.action)
            if args.pretty:
                render_feedback(console, next_observation)
            reward_total = float(next_observation.reward_breakdown.get("total", 0.0))
            log_step(
                round_index=round_index,
                action=planned.action,
                reward=reward_total,
                done=next_observation.done,
                stage=next_observation.workflow_stage,
            )
            trace["rounds"].append(
                {
                    "round_index": round_index,
                    "planned_action": planned.action.model_dump(),
                    "commander_execution": planned.execution.model_dump(),
                    "observation_after": next_observation.model_dump(),
                }
            )
            observation = next_observation
            round_index += 1

        if args.pretty:
            render_summary(console, observation)
        trace["final_observation"] = observation.model_dump()
        trace_path = save_trace(
            output_dir=args.output_dir,
            scenario_id=observation.scenario_id,
            seed=args.seed,
            payload=trace,
        )
        log_end(
            scenario=observation.scenario_id,
            success=observation.incident_resolved,
            rounds=round_index - 1,
            reward=observation.cumulative_reward,
            trace_path=str(trace_path),
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
