#!/usr/bin/env python3
"""Collect SPECTRA runs for demos, hint packs, and GRPO training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Protocol

import httpx
from openai import OpenAI
from rich.console import Console

from agents.commander.action_parser import parse_action
from environments.pomir_env.env import POMIREnv, POMIRObservation
from environments.shared.scenarios import list_scenarios
from environments.shared.state import CommanderAction, CommanderExecution
from runtime.terminal import render_feedback, render_intro, render_round, render_summary
from runtime.env import load_runtime_env
from training.dataset_builder import (
    EpisodeSummary,
    EpisodeTrace,
    EpisodeTraceStep,
    StepRecord,
    write_episode_summaries,
    write_episode_trace,
    write_step_records,
)
from training.hint_builder import build_hint_pack, hint_digest, render_hint_prefix, write_hint_pack
from training.baselines.random_commander import RandomCommander

try:
    from huggingface_hub import InferenceClient
except Exception:  # pragma: no cover - optional runtime dependency
    InferenceClient = None  # type: ignore[assignment]


MULTI_AGENT_SYSTEM_PROMPT = (
    Path(__file__).resolve().parent / "agents" / "commander" / "system_prompt.txt"
).read_text(encoding="utf-8")
SINGLE_AGENT_SYSTEM_PROMPT = (
    Path(__file__).resolve().parent / "agents" / "commander" / "single_agent_system_prompt.txt"
).read_text(encoding="utf-8")


def log_start(*, scenario: str, runtime: str, seed: int, observation_mode: str, commander: str) -> None:
    print(
        f"[START] scenario={scenario} runtime={runtime} seed={seed} "
        f"observation_mode={observation_mode} commander={commander}",
        flush=True,
    )


def log_round(*, round_index: int, stage: str, actions: list[str]) -> None:
    allowed = ",".join(actions)
    print(f"[ROUND] round={round_index} stage={stage} allowed={allowed}", flush=True)


def log_step(*, round_index: int, action: CommanderAction, reward: float, done: bool, stage: str) -> None:
    print(
        f"[STEP] round={round_index} action={action.rendered} reward={reward:.3f} "
        f"done={str(done).lower()} stage={stage}",
        flush=True,
    )


def log_end(*, scenario: str, success: bool, rounds: int, reward: float, dataset_path: str, summary_path: str) -> None:
    print(
        f"[END] scenario={scenario} success={str(success).lower()} rounds={rounds} "
        f"total_reward={reward:.3f} dataset={dataset_path} summary={summary_path}",
        flush=True,
    )


@dataclass
class PlannedAction:
    action: CommanderAction
    execution: CommanderExecution


@dataclass
class CommanderReply:
    action: CommanderAction
    completion: str
    backend: str
    model: str | None
    latency_ms: float
    parse_status: str = "ok"
    repair_retry_used: bool = False


class RuntimeClient(Protocol):
    def reset(
        self,
        *,
        difficulty: str,
        scenario_id: str | None,
        seed: int,
        observation_mode: str,
        specialist_mode: str,
    ) -> POMIRObservation:
        ...

    def plan(self) -> PlannedAction:
        ...

    def step(self, action: CommanderAction) -> POMIRObservation:
        ...

    def close(self) -> None:
        ...


class LocalRuntime:
    def __init__(self, *, specialist_mode: str, observation_mode: str) -> None:
        self.env = POMIREnv(
            mode=specialist_mode,
            specialist_mode=specialist_mode,
            observation_mode=observation_mode,
        )

    def reset(
        self,
        *,
        difficulty: str,
        scenario_id: str | None,
        seed: int,
        observation_mode: str,
        specialist_mode: str,
    ) -> POMIRObservation:
        self.env.observation_mode = observation_mode
        self.env.specialist_mode = specialist_mode
        return self.env.reset(
            difficulty=difficulty,
            scenario_id=scenario_id,
            seed=seed,
            observation_mode=observation_mode,
            specialist_mode=specialist_mode,
        )

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

    def reset(
        self,
        *,
        difficulty: str,
        scenario_id: str | None,
        seed: int,
        observation_mode: str,
        specialist_mode: str,
    ) -> POMIRObservation:
        response = self.client.post(
            "/reset",
            json={
                "difficulty": difficulty,
                "scenario_id": scenario_id,
                "seed": seed,
                "observation_mode": observation_mode,
                "specialist_mode": specialist_mode,
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


class LLMCommander:
    def __init__(
        self,
        *,
        model_name: str,
        provider: str,
        base_url: str | None,
        api_key: str | None,
        hf_provider: str | None,
    ) -> None:
        self.model_name = model_name
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key or "local"
        self.hf_provider = hf_provider or None
        self.client = None
        if provider == "hf":
            if InferenceClient is None:
                raise RuntimeError("huggingface_hub is not available for commander_provider=hf")
            self.client = InferenceClient(
                model=self.model_name,
                token=self.api_key,
                provider=self.hf_provider,
            )
        else:
            self.client = OpenAI(base_url=base_url, api_key=self.api_key, timeout=45.0)

    def _request(self, *, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "hf":
            response = self.client.chat_completion(  # type: ignore[union-attr]
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=220,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or ""
        response = self.client.chat.completions.create(  # type: ignore[union-attr]
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=220,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    def complete(
        self,
        *,
        prompt: str,
        observation_mode: str,
        allowed_actions: list[str],
        valid_action_example: dict[str, object],
    ) -> CommanderReply:
        system_prompt = SINGLE_AGENT_SYSTEM_PROMPT if observation_mode == "single_agent" else MULTI_AGENT_SYSTEM_PROMPT
        started = time.perf_counter()
        content = self._request(system_prompt=system_prompt, user_prompt=prompt)
        parse_status = "ok"
        repair_retry_used = False
        try:
            action = safe_parse_action(
                content,
                allowed_actions=allowed_actions,
                valid_action_example=valid_action_example,
                fail_closed=True,
            )
        except Exception:
            repair_retry_used = True
            parse_status = "repaired"
            repair_prompt = build_repair_prompt(
                raw_bad_output=content,
                allowed_actions=allowed_actions,
                valid_action_example=valid_action_example,
            )
            repaired = self._request(
                system_prompt="Return exactly one valid JSON action and nothing else.",
                user_prompt=repair_prompt,
            )
            try:
                action = safe_parse_action(
                    repaired,
                    allowed_actions=allowed_actions,
                    valid_action_example=valid_action_example,
                    fail_closed=True,
                )
                content = repaired
            except Exception:
                parse_status = "fallback"
                action = safe_parse_action(
                    repaired,
                    allowed_actions=allowed_actions,
                    valid_action_example=valid_action_example,
                )
                content = repaired
        return CommanderReply(
            action=action,
            completion=content,
            backend=f"llm:{self.provider}",
            model=self.model_name,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
            parse_status=parse_status,
            repair_retry_used=repair_retry_used,
        )


def safe_parse_action(
    text: str,
    *,
    allowed_actions: list[str] | None = None,
    valid_action_example: dict[str, object] | None = None,
    fail_closed: bool = False,
) -> CommanderAction:
    try:
        action = parse_action(text)
        if allowed_actions:
            allowed_names = {item.split("(", 1)[0] for item in allowed_actions}
            if action.action_type not in allowed_names:
                raise ValueError(f"Action {action.action_type} not allowed here")
        return action
    except Exception:
        if fail_closed:
            raise
        if allowed_actions:
            allowed_names = [item.split("(", 1)[0] for item in allowed_actions]
            candidate = text.lower()
            service_names = ("api-gateway", "database", "cache", "worker", "auth_service")
            agent_names = ("infra", "log", "security")

            action_type = next((name for name in allowed_names if name in candidate), None)
            if action_type is None:
                example_action_type = valid_action_example.get("action_type") if valid_action_example else None
                if isinstance(example_action_type, str) and example_action_type in allowed_names:
                    action_type = example_action_type
                else:
                    action_type = allowed_names[0]

            if action_type == "request_followup":
                target_agent = next((name for name in agent_names if name in candidate), None)
                if target_agent is None:
                    example_target_agent = valid_action_example.get("target_agent") if valid_action_example else None
                    if isinstance(example_target_agent, str) and example_target_agent in agent_names:
                        target_agent = example_target_agent
                    else:
                        target_agent = "infra"
                return CommanderAction(action_type="request_followup", target_agent=target_agent)

            if action_type == "submit_resolution":
                resolution_summary = "Commander marked the incident as resolved."
                example_summary = valid_action_example.get("resolution_summary") if valid_action_example else None
                if isinstance(example_summary, str) and example_summary.strip():
                    resolution_summary = example_summary
                return CommanderAction(
                    action_type="submit_resolution",
                    resolution_summary=resolution_summary,
                )

            target_service = next((name for name in service_names if name in candidate), None)
            if target_service is None:
                example_target_service = valid_action_example.get("target_service") if valid_action_example else None
                if isinstance(example_target_service, str) and example_target_service in service_names:
                    target_service = example_target_service
                else:
                    target_service = "api-gateway"
            return CommanderAction(action_type=action_type, target_service=target_service)
        return CommanderAction(action_type="investigate_service", target_service="api-gateway")


def scenario_sequence(*, scenario_id: str | None, difficulty: str, episodes: int) -> list[tuple[str | None, str]]:
    if scenario_id is not None:
        return [(scenario_id, difficulty)] * episodes
    if difficulty != "mixed":
        return [(None, difficulty)] * episodes
    ordered = list_scenarios()
    result: list[tuple[str | None, str]] = []
    for index in range(episodes):
        scenario = ordered[index % len(ordered)]
        result.append((scenario.scenario_id, scenario.difficulty))
    return result


def default_output_paths(
    *,
    output_dir: str,
    commander_backend: str,
    observation_mode: str,
    episodes: int,
    scenario_id: str | None,
) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = scenario_id or f"{episodes}eps"
    stem = f"{timestamp}_{commander_backend}_{observation_mode}_{suffix}"
    base = Path(output_dir)
    return base / f"{stem}.jsonl", base / f"{stem}.summary.json"


def build_prompt(observation: POMIRObservation, *, hint_pack: dict | None) -> tuple[str, str | None]:
    hint_id = None
    hint_prefix = ""
    if hint_pack is not None:
        hint_prefix = render_hint_prefix(hint_pack, scenario_id=observation.scenario_id)
        hint_id = hint_digest(hint_pack)
    contract = build_runtime_prompt(observation, hint_prefix=hint_prefix)
    return contract, hint_id


def build_runtime_prompt(observation: POMIRObservation, *, hint_prefix: str = "") -> str:
    allowed_block = "\n".join(f"- {action}" for action in observation.allowed_actions) or "- none"
    required_fields_block = "\n".join(
        f"- {action}: {', '.join(fields) if fields else 'none'}"
        for action, fields in observation.required_fields_by_action.items()
    ) or "- none"
    progress_block = "\n".join(
        f"- {flag}: {str(value).lower()}" for flag, value in observation.progress_flags.items()
    ) or "- none"
    parts = [
        f"Current stage: {observation.workflow_stage}",
        f"Stage goal: {observation.stage_goal}",
        f"Observation mode: {observation.observation_mode}",
        f"Incident resolved: {str(observation.incident_resolved).lower()}",
        f"Last action result: {observation.last_action_result or 'none yet'}",
    ]
    if hint_prefix:
        parts.extend(["", "Cheat sheet:", hint_prefix])
    if observation.loop_warning:
        parts.extend(["", f"Loop warning: {observation.loop_warning}"])
    parts.extend(
        [
            "",
            "Allowed actions:",
            allowed_block,
            "",
            "Required fields:",
            required_fields_block,
            "",
            "Progress flags:",
            progress_block,
            "",
            "Decision rules:",
            "- Return exactly one valid JSON action.",
            "- You may include an optional `reasoning` field in the JSON.",
            "- Do not use submit_resolution unless service_recovered is true.",
            "- If the last action failed or produced no progress, change the action family or target.",
            "- Prefer the root-cause service over the hottest victim service when evidence disagrees.",
            "",
            f"Valid example: {json.dumps(observation.valid_action_example, ensure_ascii=True)}",
            "",
            "Environment state:",
            observation.prompt_text,
            "",
            "Return exactly one JSON object.",
        ]
    )
    return "\n".join(parts)


def build_repair_prompt(
    *,
    raw_bad_output: str,
    allowed_actions: list[str],
    valid_action_example: dict[str, object],
) -> str:
    allowed_block = "\n".join(f"- {action}" for action in allowed_actions) or "- none"
    return "\n".join(
        [
            "Your previous response was invalid or disallowed.",
            "Allowed actions:",
            allowed_block,
            f"Previous output: {raw_bad_output}",
            f"Valid example: {json.dumps(valid_action_example, ensure_ascii=True)}",
            "Return exactly one JSON object and nothing else.",
        ]
    )


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
    parser.add_argument("--specialist-mode", default="deterministic", choices=["deterministic", "hybrid", "llm"])
    parser.add_argument("--commander", default="heuristic", choices=["heuristic", "random", "llm", "single-agent"])
    parser.add_argument("--commander-model", default=os.getenv("COMMANDER_MODEL", "Qwen/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--commander-provider", default="openai", choices=["openai", "hf"])
    parser.add_argument("--commander-base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--commander-api-key", default=None)
    parser.add_argument("--commander-hf-provider", nargs="?", const="", default=None)
    parser.add_argument("--output-dir", default="outputs/runs")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--hint-file", default=None)
    parser.add_argument("--export-hint-file", default=None)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    if args.commander == "single-agent":
        args.observation_mode = "single_agent"

    dataset_path, summary_path = default_output_paths(
        output_dir=args.output_dir,
        commander_backend=args.commander,
        observation_mode=args.observation_mode,
        episodes=args.episodes,
        scenario_id=args.scenario_id,
    )
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
    if args.summary_path:
        summary_path = Path(args.summary_path)

    hint_pack = None
    if args.hint_file:
        hint_pack = json.loads(Path(args.hint_file).read_text(encoding="utf-8"))

    console = Console()
    runtime: RuntimeClient
    if args.runtime == "local":
        runtime = LocalRuntime(specialist_mode=args.specialist_mode, observation_mode=args.observation_mode)
    else:
        runtime = RemoteRuntime(base_url=args.base_url)

    llm_commander = None
    if args.commander == "llm":
        commander_api_key = args.commander_api_key
        if commander_api_key is None:
            commander_api_key = os.getenv("HF_TOKEN") if args.commander_provider == "hf" else os.getenv("OPENAI_API_KEY", "local")
        llm_commander = LLMCommander(
            model_name=args.commander_model,
            provider=args.commander_provider,
            base_url=args.commander_base_url,
            api_key=commander_api_key,
            hf_provider=args.commander_hf_provider,
        )

    random_commander = RandomCommander(seed=args.seed)
    records: list[StepRecord] = []
    episode_summaries: list[EpisodeSummary] = []

    try:
        for episode_index, (scenario_id, difficulty) in enumerate(
            scenario_sequence(scenario_id=args.scenario_id, difficulty=args.difficulty, episodes=args.episodes),
            start=1,
        ):
            seed = args.seed + episode_index - 1
            observation = runtime.reset(
                difficulty=difficulty,
                scenario_id=scenario_id,
                seed=seed,
                observation_mode=args.observation_mode,
                specialist_mode=args.specialist_mode,
            )
            if args.pretty and episode_index == 1:
                render_intro(console, observation)
            log_start(
                scenario=observation.scenario_id,
                runtime=args.runtime,
                seed=seed,
                observation_mode=args.observation_mode,
                commander=args.commander,
            )

            prior_actions: list[dict[str, object]] = []
            actions_rendered: list[str] = []
            episode_trace_steps: list[EpisodeTraceStep] = []
            round_index = 1

            while not observation.done:
                prompt, hint_id = build_prompt(observation, hint_pack=hint_pack)
                log_round(
                    round_index=round_index,
                    stage=observation.workflow_stage,
                    actions=observation.allowed_actions,
                )

                if args.commander in {"heuristic", "single-agent"}:
                    planned = runtime.plan()
                    commander_reply = CommanderReply(
                        action=planned.action,
                        completion=planned.execution.raw_response,
                        backend=planned.execution.mode,
                        model=None,
                        latency_ms=planned.execution.latency_ms,
                    )
                    execution = planned.execution
                elif args.commander == "random":
                    action = random_commander.act(observation.allowed_actions)
                    commander_reply = CommanderReply(
                        action=action,
                        completion=json.dumps(action.model_dump(), ensure_ascii=True),
                        backend="random",
                        model=None,
                        latency_ms=0.0,
                    )
                    execution = CommanderExecution(
                        mode="heuristic",
                        raw_response=commander_reply.completion,
                        latency_ms=0.0,
                        trust_weights={},
                        action=action.model_dump(),
                    )
                else:
                    assert llm_commander is not None
                    commander_reply = llm_commander.complete(
                        prompt=prompt,
                        observation_mode=args.observation_mode,
                        allowed_actions=observation.allowed_actions,
                        valid_action_example=observation.valid_action_example,
                    )
                    execution = CommanderExecution(
                        mode="llm",
                        raw_response=commander_reply.completion,
                        latency_ms=commander_reply.latency_ms,
                        trust_weights={},
                        action=commander_reply.action.model_dump(),
                    )

                if args.pretty and episode_index == 1:
                    render_round(
                        console,
                        round_index=round_index,
                        observation=observation,
                        execution=execution,
                        action=commander_reply.action,
                    )

                next_observation = runtime.step(commander_reply.action)
                if args.pretty and episode_index == 1:
                    render_feedback(console, next_observation)

                reward_total = float(next_observation.reward_breakdown.get("total", 0.0))
                log_step(
                    round_index=round_index,
                    action=commander_reply.action,
                    reward=reward_total,
                    done=next_observation.done,
                    stage=next_observation.workflow_stage,
                )

                records.append(
                    StepRecord(
                        prompt=prompt,
                        completion=commander_reply.completion,
                        reference_action=commander_reply.action.model_dump(),
                        reward=reward_total,
                        reward_breakdown=dict(next_observation.reward_breakdown),
                        runtime=args.runtime,
                        commander_backend=commander_reply.backend,
                        commander_model=commander_reply.model,
                        observation_mode=args.observation_mode,
                        specialist_mode=args.specialist_mode,
                        episode_id=observation.metadata.get("episode_id", observation.metadata.get("episodeId", "")) or execution.episode_id or f"episode_{episode_index}",
                        episode_index=episode_index,
                        step_index=round_index,
                        scenario_id=observation.scenario_id,
                        scenario_name=observation.scenario_name,
                        difficulty=observation.difficulty,
                        workflow_stage=observation.workflow_stage,
                        seed=seed,
                        allowed_actions=list(observation.allowed_actions),
                        prior_actions=list(prior_actions),
                        report_targets={report.agent_id.value: report.top_hypothesis_service for report in observation.reports},
                        report_confidences={report.agent_id.value: report.confidence for report in observation.reports},
                        specialist_reports=[report.model_dump() for report in observation.reports],
                        specialist_executions=[execution.model_dump() for execution in observation.specialist_executions],
                        stage_goal=observation.stage_goal,
                        valid_action_example=dict(observation.valid_action_example),
                        commander_parse_status=commander_reply.parse_status,
                        commander_repair_retry_used=commander_reply.repair_retry_used,
                        commander_latency_ms=commander_reply.latency_ms,
                        hint_used=hint_pack is not None,
                        hint_digest=hint_id,
                        environment_feedback=next_observation.last_action_result,
                    )
                )

                episode_trace_steps.append(
                    EpisodeTraceStep(
                        step_index=round_index,
                        workflow_stage=observation.workflow_stage,
                        stage_goal=observation.stage_goal,
                        prompt=prompt,
                        allowed_actions=list(observation.allowed_actions),
                        valid_action_example=dict(observation.valid_action_example),
                        specialist_reports=[report.model_dump() for report in observation.reports],
                        specialist_executions=[execution.model_dump() for execution in observation.specialist_executions],
                        commander_reply={
                            "backend": commander_reply.backend,
                            "model": commander_reply.model,
                            "latency_ms": commander_reply.latency_ms,
                            "parse_status": commander_reply.parse_status,
                            "repair_retry_used": commander_reply.repair_retry_used,
                            "raw_completion": commander_reply.completion,
                        },
                        reference_action=commander_reply.action.model_dump(),
                        reward_breakdown=dict(next_observation.reward_breakdown),
                        environment_feedback=next_observation.last_action_result,
                        next_workflow_stage=next_observation.workflow_stage,
                        done=next_observation.done,
                    )
                )

                prior_actions.append(commander_reply.action.model_dump())
                actions_rendered.append(commander_reply.action.rendered)
                observation = next_observation
                round_index += 1

            if args.pretty and episode_index == 1:
                render_summary(console, observation)

            episode_summaries.append(
                EpisodeSummary(
                    episode_id=records[-1].episode_id,
                    episode_index=episode_index,
                    scenario_id=observation.scenario_id,
                    scenario_name=observation.scenario_name,
                    difficulty=observation.difficulty,
                    seed=seed,
                    runtime=args.runtime,
                    commander_backend=args.commander,
                    commander_model=args.commander_model if args.commander == "llm" else None,
                    observation_mode=args.observation_mode,
                    specialist_mode=args.specialist_mode,
                    steps=round_index - 1,
                    incident_resolved=observation.incident_resolved,
                    cumulative_reward=observation.cumulative_reward,
                    actions=actions_rendered,
                )
            )

            if args.trace_dir:
                trace_path = Path(args.trace_dir) / f"{observation.scenario_id}_episode_{episode_index:02d}.trace.json"
                write_episode_trace(
                    trace_path,
                    EpisodeTrace(
                        episode_id=records[-1].episode_id,
                        episode_index=episode_index,
                        scenario_id=observation.scenario_id,
                        scenario_name=observation.scenario_name,
                        difficulty=observation.difficulty,
                        seed=seed,
                        runtime=args.runtime,
                        commander_backend=episode_summaries[-1].commander_backend,
                        commander_model=episode_summaries[-1].commander_model,
                        observation_mode=args.observation_mode,
                        specialist_mode=args.specialist_mode,
                        hint_used=hint_pack is not None,
                        hint_digest=hint_digest(hint_pack) if hint_pack is not None else None,
                        steps=episode_trace_steps,
                    ),
                )

        write_step_records(dataset_path, records)
        write_episode_summaries(summary_path, episode_summaries)

        if args.export_hint_file:
            write_hint_pack(args.export_hint_file, build_hint_pack(records))

        final_summary = episode_summaries[-1]
        log_end(
            scenario=final_summary.scenario_id,
            success=final_summary.incident_resolved,
            rounds=final_summary.steps,
            reward=final_summary.cumulative_reward,
            dataset_path=str(dataset_path),
            summary_path=str(summary_path),
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
