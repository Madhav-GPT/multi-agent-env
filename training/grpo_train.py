"""Environment-backed GRPO training for the SPECTRA commander."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from agents.commander.action_parser import parse_action
from environments.pomir_env.env import POMIREnv
from training.dataset_builder import load_step_records


def load_grpo_dataset(jsonl_path: str | Path) -> Dataset:
    records = load_step_records(jsonl_path)
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "prompt": record.prompt,
                "scenario_id": record.scenario_id,
                "seed": record.seed,
                "step_index": record.step_index,
                "prior_actions": record.prior_actions,
                "observation_mode": record.observation_mode,
                "specialist_mode": record.specialist_mode,
                "reference_completion": record.completion,
                "reference_reward": record.reward,
            }
        )
    return Dataset.from_list(rows)


def replay_reward(
    *,
    prompts: list[str],
    completions: list[str],
    scenario_id: list[str],
    seed: list[int],
    prior_actions: list[list[dict[str, Any]]],
    observation_mode: list[str],
    specialist_mode: list[str],
    replay_specialist_mode: str,
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    for completion, item_scenario_id, item_seed, item_prior_actions, item_observation_mode, item_specialist_mode in zip(
        completions,
        scenario_id,
        seed,
        prior_actions,
        observation_mode,
        specialist_mode,
        strict=True,
    ):
        env = POMIREnv(
            mode=replay_specialist_mode,
            specialist_mode=replay_specialist_mode,
            observation_mode=item_observation_mode,
        )
        try:
            env.reset(
                scenario_id=item_scenario_id,
                seed=item_seed,
                observation_mode=item_observation_mode,
                specialist_mode=replay_specialist_mode,
            )
            for prior in item_prior_actions:
                env.step(prior)
            try:
                action = parse_action(completion)
                observation = env.step(action)
                rewards.append(float(observation.reward_breakdown.get("total", 0.0)))
            except Exception:
                rewards.append(0.0)
        finally:
            env.close()
    return rewards


def dry_run_report(dataset: Dataset, sample_size: int, replay_specialist_mode: str) -> dict[str, Any]:
    sample = dataset.select(range(min(sample_size, len(dataset))))
    rewards = replay_reward(
        prompts=sample["prompt"],
        completions=sample["reference_completion"],
        scenario_id=sample["scenario_id"],
        seed=sample["seed"],
        prior_actions=sample["prior_actions"],
        observation_mode=sample["observation_mode"],
        specialist_mode=sample["specialist_mode"],
        replay_specialist_mode=replay_specialist_mode,
    )
    return {
        "rows_checked": len(rewards),
        "mean_reference_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
        "max_reference_reward": round(max(rewards), 4) if rewards else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="outputs/grpo_runs")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-completion-length", type=int, default=96)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-samples", type=int, default=8)
    parser.add_argument("--replay-specialist-mode", default="deterministic", choices=["deterministic", "hybrid", "llm"])
    args = parser.parse_args()

    dataset = load_grpo_dataset(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        report = dry_run_report(dataset, args.dry_run_samples, args.replay_specialist_mode)
        report["dataset_rows"] = len(dataset)
        print(json.dumps(report, indent=2))
        return

    config = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        logging_steps=1,
        save_steps=max(1, args.max_steps),
        report_to=args.report_to,
        use_cpu=args.use_cpu,
        remove_unused_columns=False,
        do_train=True,
        beta=0.0,
    )
    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=partial(replay_reward, replay_specialist_mode=args.replay_specialist_mode),
        args=config,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "final_model": str(output_dir / "final"),
                "steps": args.max_steps,
                "dataset_rows": len(dataset),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
