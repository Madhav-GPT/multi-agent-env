"""Dataset models and file helpers for inference and training."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class StepRecord:
    prompt: str
    completion: str
    reference_action: dict[str, Any]
    reward: float
    reward_breakdown: dict[str, Any]
    runtime: str
    commander_backend: str
    commander_model: str | None
    observation_mode: str
    specialist_mode: str
    episode_id: str
    episode_index: int
    step_index: int
    scenario_id: str
    scenario_name: str
    difficulty: str
    workflow_stage: str
    seed: int
    allowed_actions: list[str] = field(default_factory=list)
    prior_actions: list[dict[str, Any]] = field(default_factory=list)
    report_targets: dict[str, str] = field(default_factory=dict)
    report_confidences: dict[str, float] = field(default_factory=dict)
    specialist_reports: list[dict[str, Any]] = field(default_factory=list)
    specialist_executions: list[dict[str, Any]] = field(default_factory=list)
    stage_goal: str = ""
    valid_action_example: dict[str, Any] = field(default_factory=dict)
    commander_parse_status: str = "ok"
    commander_repair_retry_used: bool = False
    commander_latency_ms: float = 0.0
    hint_used: bool = False
    hint_digest: str | None = None
    environment_feedback: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeSummary:
    episode_id: str
    episode_index: int
    scenario_id: str
    scenario_name: str
    difficulty: str
    seed: int
    runtime: str
    commander_backend: str
    commander_model: str | None
    observation_mode: str
    specialist_mode: str
    steps: int
    incident_resolved: bool
    cumulative_reward: float
    actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeTraceStep:
    step_index: int
    workflow_stage: str
    stage_goal: str
    prompt: str
    allowed_actions: list[str]
    valid_action_example: dict[str, Any]
    specialist_reports: list[dict[str, Any]] = field(default_factory=list)
    specialist_executions: list[dict[str, Any]] = field(default_factory=list)
    commander_reply: dict[str, Any] = field(default_factory=dict)
    reference_action: dict[str, Any] = field(default_factory=dict)
    reward_breakdown: dict[str, Any] = field(default_factory=dict)
    environment_feedback: str = ""
    next_workflow_stage: str = ""
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeTrace:
    episode_id: str
    episode_index: int
    scenario_id: str
    scenario_name: str
    difficulty: str
    seed: int
    runtime: str
    commander_backend: str
    commander_model: str | None
    observation_mode: str
    specialist_mode: str
    hint_used: bool
    hint_digest: str | None
    steps: list[EpisodeTraceStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_step_records(path: str | Path, records: list[StepRecord]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=True) + "\n")
    return output_path


def _render_action(action: dict[str, Any]) -> str:
    action_type = action.get("action_type", "")
    if action_type == "request_followup":
        return f"request_followup({action.get('target_agent', 'unknown')})"
    if action_type == "submit_resolution":
        return "submit_resolution(summary)"
    if action.get("target_service"):
        return f"{action_type}({action['target_service']})"
    return action_type


def write_episode_summaries(path: str | Path, summaries: list[EpisodeSummary]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([summary.to_dict() for summary in summaries], indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return output_path


def write_episode_trace(path: str | Path, trace: EpisodeTrace) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(trace.to_dict(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return output_path


def load_step_records(path: str | Path) -> list[StepRecord]:
    records: list[StepRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            records.append(StepRecord(**payload))
    return records


def summarize_records(records: list[StepRecord]) -> list[EpisodeSummary]:
    grouped: dict[str, list[StepRecord]] = defaultdict(list)
    for record in records:
        grouped[record.episode_id].append(record)

    summaries: list[EpisodeSummary] = []
    for episode_id, items in grouped.items():
        ordered = sorted(items, key=lambda item: item.step_index)
        final = ordered[-1]
        summaries.append(
            EpisodeSummary(
                episode_id=episode_id,
                episode_index=final.episode_index,
                scenario_id=final.scenario_id,
                scenario_name=final.scenario_name,
                difficulty=final.difficulty,
                seed=final.seed,
                runtime=final.runtime,
                commander_backend=final.commander_backend,
                commander_model=final.commander_model,
                observation_mode=final.observation_mode,
                specialist_mode=final.specialist_mode,
                steps=len(ordered),
                incident_resolved=bool(final.reward_breakdown.get("r1_resolution", 0.0) >= 0.5),
                cumulative_reward=round(sum(item.reward for item in ordered), 3),
                actions=[_render_action(item.reference_action) for item in ordered],
            )
        )
    return sorted(summaries, key=lambda item: item.episode_index)
