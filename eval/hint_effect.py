"""Compare the same commander before and after applying a hint pack."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import LLMCommander, LocalRuntime, RemoteRuntime, build_prompt, scenario_sequence
from runtime.env import load_runtime_env
from training.dataset_builder import EpisodeSummary, StepRecord, write_episode_summaries, write_step_records


def _run_condition(
    *,
    name: str,
    runtime_mode: str,
    base_url: str,
    episodes: int,
    difficulty: str,
    scenario_id: str | None,
    seed: int,
    observation_mode: str,
    specialist_mode: str,
    commander: LLMCommander,
    hint_pack: dict[str, Any] | None,
) -> tuple[list[StepRecord], list[EpisodeSummary]]:
    runtime = (
        LocalRuntime(specialist_mode=specialist_mode, observation_mode=observation_mode)
        if runtime_mode == "local"
        else RemoteRuntime(base_url=base_url)
    )
    records: list[StepRecord] = []
    summaries: list[EpisodeSummary] = []
    try:
        for episode_index, (episode_scenario_id, episode_difficulty) in enumerate(
            scenario_sequence(scenario_id=scenario_id, difficulty=difficulty, episodes=episodes),
            start=1,
        ):
            episode_seed = seed + episode_index - 1
            observation = runtime.reset(
                difficulty=episode_difficulty,
                scenario_id=episode_scenario_id,
                seed=episode_seed,
                observation_mode=observation_mode,
                specialist_mode=specialist_mode,
            )
            prior_actions: list[dict[str, object]] = []
            actions_rendered: list[str] = []
            round_index = 1

            while not observation.done:
                prompt, hint_id = build_prompt(observation, hint_pack=hint_pack)
                reply = commander.complete(
                    prompt=prompt,
                    observation_mode=observation_mode,
                    allowed_actions=observation.allowed_actions,
                    valid_action_example=observation.valid_action_example,
                )
                next_observation = runtime.step(reply.action)
                reward_total = float(next_observation.reward_breakdown.get("total", 0.0))
                records.append(
                    StepRecord(
                        prompt=prompt,
                        completion=reply.completion,
                        reference_action=reply.action.model_dump(),
                        reward=reward_total,
                        reward_breakdown=dict(next_observation.reward_breakdown),
                        runtime=runtime_mode,
                        commander_backend=f"{name}:{reply.backend}",
                        commander_model=reply.model,
                        observation_mode=observation_mode,
                        specialist_mode=specialist_mode,
                        episode_id=observation.metadata.get("episode_id", "") or f"{name}_episode_{episode_index}",
                        episode_index=episode_index,
                        step_index=round_index,
                        scenario_id=observation.scenario_id,
                        scenario_name=observation.scenario_name,
                        difficulty=observation.difficulty,
                        workflow_stage=observation.workflow_stage,
                        seed=episode_seed,
                        allowed_actions=list(observation.allowed_actions),
                        prior_actions=list(prior_actions),
                        report_targets={report.agent_id.value: report.top_hypothesis_service for report in observation.reports},
                        report_confidences={report.agent_id.value: report.confidence for report in observation.reports},
                        specialist_reports=[report.model_dump() for report in observation.reports],
                        specialist_executions=[execution.model_dump() for execution in observation.specialist_executions],
                        stage_goal=observation.stage_goal,
                        valid_action_example=dict(observation.valid_action_example),
                        commander_parse_status=reply.parse_status,
                        commander_repair_retry_used=reply.repair_retry_used,
                        commander_latency_ms=reply.latency_ms,
                        hint_used=hint_pack is not None,
                        hint_digest=hint_id,
                        environment_feedback=next_observation.last_action_result,
                    )
                )
                prior_actions.append(reply.action.model_dump())
                actions_rendered.append(reply.action.rendered)
                observation = next_observation
                round_index += 1

            summaries.append(
                EpisodeSummary(
                    episode_id=records[-1].episode_id,
                    episode_index=episode_index,
                    scenario_id=observation.scenario_id,
                    scenario_name=observation.scenario_name,
                    difficulty=observation.difficulty,
                    seed=episode_seed,
                    runtime=runtime_mode,
                    commander_backend=f"{name}:{commander.provider}",
                    commander_model=commander.model_name,
                    observation_mode=observation_mode,
                    specialist_mode=specialist_mode,
                    steps=round_index - 1,
                    incident_resolved=observation.incident_resolved,
                    cumulative_reward=observation.cumulative_reward,
                    actions=actions_rendered,
                )
            )
    finally:
        runtime.close()
    return records, summaries


def _summary_payload(summaries: list[EpisodeSummary]) -> dict[str, Any]:
    if not summaries:
        return {"episodes": 0, "success_rate": 0.0, "mean_reward": 0.0, "mean_steps": 0.0}
    return {
        "episodes": len(summaries),
        "success_rate": round(sum(1 for item in summaries if item.incident_resolved) / len(summaries), 3),
        "mean_reward": round(sum(item.cumulative_reward for item in summaries) / len(summaries), 3),
        "mean_steps": round(sum(item.steps for item in summaries) / len(summaries), 3),
    }


def main() -> None:
    load_runtime_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard", "mixed"])
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runtime", default="local", choices=["local", "remote"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--observation-mode", default="multi_agent", choices=["multi_agent", "single_agent"])
    parser.add_argument("--specialist-mode", default="llm", choices=["deterministic", "hybrid", "llm"])
    parser.add_argument("--hint-file", required=True)
    parser.add_argument("--commander-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--commander-provider", default="hf", choices=["openai", "hf"])
    parser.add_argument("--commander-base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--commander-api-key", default=None)
    parser.add_argument("--commander-hf-provider", nargs="?", const="", default=None)
    parser.add_argument("--output-dir", default="outputs/hint_effect")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hint_pack = json.loads(Path(args.hint_file).read_text(encoding="utf-8"))
    commander_api_key = args.commander_api_key
    if commander_api_key is None:
        commander_api_key = (
            os.environ.get("HF_TOKEN")
            if args.commander_provider == "hf"
            else os.environ.get("OPENAI_API_KEY", "local")
        )

    blind_commander = LLMCommander(
        model_name=args.commander_model,
        provider=args.commander_provider,
        base_url=args.commander_base_url,
        api_key=commander_api_key,
        hf_provider=args.commander_hf_provider,
    )
    hinted_commander = LLMCommander(
        model_name=args.commander_model,
        provider=args.commander_provider,
        base_url=args.commander_base_url,
        api_key=commander_api_key,
        hf_provider=args.commander_hf_provider,
    )

    blind_records, blind_summaries = _run_condition(
        name="blind",
        runtime_mode=args.runtime,
        base_url=args.base_url,
        episodes=args.episodes,
        difficulty=args.difficulty,
        scenario_id=args.scenario_id,
        seed=args.seed,
        observation_mode=args.observation_mode,
        specialist_mode=args.specialist_mode,
        commander=blind_commander,
        hint_pack=None,
    )
    hinted_records, hinted_summaries = _run_condition(
        name="hinted",
        runtime_mode=args.runtime,
        base_url=args.base_url,
        episodes=args.episodes,
        difficulty=args.difficulty,
        scenario_id=args.scenario_id,
        seed=args.seed,
        observation_mode=args.observation_mode,
        specialist_mode=args.specialist_mode,
        commander=hinted_commander,
        hint_pack=hint_pack,
    )

    blind_dataset = output_dir / "blind.jsonl"
    blind_summary = output_dir / "blind.summary.json"
    hinted_dataset = output_dir / "hinted.jsonl"
    hinted_summary = output_dir / "hinted.summary.json"
    write_step_records(blind_dataset, blind_records)
    write_episode_summaries(blind_summary, blind_summaries)
    write_step_records(hinted_dataset, hinted_records)
    write_episode_summaries(hinted_summary, hinted_summaries)

    comparison = {
        "blind": _summary_payload(blind_summaries),
        "hinted": _summary_payload(hinted_summaries),
        "delta_success_rate": round(
            _summary_payload(hinted_summaries)["success_rate"] - _summary_payload(blind_summaries)["success_rate"],
            3,
        ),
        "delta_mean_reward": round(
            _summary_payload(hinted_summaries)["mean_reward"] - _summary_payload(blind_summaries)["mean_reward"],
            3,
        ),
        "artifacts": {
            "blind_dataset": str(blind_dataset),
            "blind_summary": str(blind_summary),
            "hinted_dataset": str(hinted_dataset),
            "hinted_summary": str(hinted_summary),
        },
    }
    comparison_path = output_dir / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
