"""Terminal rendering helpers for SPECTRA episodes."""

from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from environments.pomir_env.env import POMIRObservation
from environments.shared.state import CommanderAction, CommanderExecution, SpecialistRoundExecution


def truncate(value: Any, limit: int) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


def render_intro(console: Console, observation: POMIRObservation) -> None:
    body = (
        f"Scenario: {observation.scenario_name} ({observation.scenario_id})\n"
        f"Difficulty: {observation.difficulty} | Stage: {observation.workflow_stage} | Step budget: {observation.step_budget}\n"
        f"Common trap: {observation.common_trap}"
    )
    console.print(Panel.fit(body, title="SPECTRA Episode", border_style="cyan"))


def build_specialist_table(executions: list[SpecialistRoundExecution]) -> Table:
    table = Table(title="Parallel Specialist Round", expand=True)
    table.add_column("Agent", style="cyan", no_wrap=True)
    table.add_column("Channel", no_wrap=True)
    table.add_column("Mode", no_wrap=True)
    table.add_column("Latency", no_wrap=True, justify="right")
    table.add_column("Follow-up", no_wrap=True)
    table.add_column("Hypothesis", overflow="fold")
    table.add_column("Action", overflow="fold")
    table.add_column("Digest", overflow="fold")
    for execution in executions:
        report = execution.report
        table.add_row(
            execution.agent,
            execution.channel,
            execution.mode,
            f"{execution.latency_ms:.2f} ms",
            "yes" if execution.followup_applied else "no",
            truncate(
                f"{report.get('top_hypothesis_service')} | {report.get('top_hypothesis_cause')} "
                f"| conf {float(report.get('confidence', 0.0)):.2f}",
                74,
            ),
            truncate(report.get("recommended_action", ""), 36),
            truncate(execution.observation_digest, 72),
        )
    return table


def build_commander_table(execution: CommanderExecution, action: CommanderAction) -> Table:
    table = Table(title="Commander Decision", expand=True)
    table.add_column("Mode", no_wrap=True)
    table.add_column("Action", no_wrap=True)
    table.add_column("Latency", no_wrap=True, justify="right")
    table.add_column("Trust Weights")
    table.add_column("Raw Output")
    trust = ", ".join(f"{agent}={weight:.2f}" for agent, weight in execution.trust_weights.items())
    table.add_row(
        execution.mode,
        action.rendered,
        f"{execution.latency_ms:.2f} ms",
        trust,
        truncate(execution.raw_response, 160),
    )
    return table


def build_feedback_table(observation: POMIRObservation) -> Table:
    reward = observation.reward_breakdown
    table = Table(title="Environment Feedback", expand=True)
    table.add_column("Stage", no_wrap=True)
    table.add_column("Remaining", no_wrap=True, justify="right")
    table.add_column("Progress")
    table.add_column("Reward")
    table.add_column("Cumulative", no_wrap=True, justify="right")
    table.add_column("Feedback")
    progress = ", ".join(
        f"{key}={'yes' if value else 'no'}" for key, value in observation.progress_flags.items()
    )
    reward_summary = ", ".join(
        f"{key}={value}"
        for key, value in reward.items()
        if key in {"r1_resolution", "r2_root_cause", "r3_coordination", "r4_efficiency", "r5_trust", "total"}
    )
    table.add_row(
        observation.workflow_stage,
        str(observation.step_budget),
        truncate(progress, 90),
        reward_summary,
        f"{observation.cumulative_reward:.3f}",
        truncate(observation.last_action_result or "No feedback.", 140),
    )
    return table


def render_round(
    console: Console,
    *,
    round_index: int,
    observation: POMIRObservation,
    execution: CommanderExecution,
    action: CommanderAction,
) -> None:
    console.rule(Text(f"Round {round_index}", style="bold green"))
    console.print(build_specialist_table(observation.specialist_executions))
    console.print(build_commander_table(execution, action))


def render_feedback(console: Console, observation: POMIRObservation) -> None:
    console.print(build_feedback_table(observation))


def render_summary(console: Console, observation: POMIRObservation) -> None:
    title = "Incident Resolved" if observation.incident_resolved else "Episode Finished"
    body = (
        f"Scenario: {observation.scenario_name}\n"
        f"Final stage: {observation.workflow_stage}\n"
        f"Total reward: {observation.cumulative_reward:.3f}\n"
        f"Result: {'resolved' if observation.incident_resolved else 'unresolved'}"
    )
    console.print(Panel.fit(body, title=title, border_style="green" if observation.incident_resolved else "yellow"))


def save_trace(
    *,
    output_dir: str | Path,
    scenario_id: str,
    seed: int | None,
    payload: dict[str, Any],
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"{scenario_id}_seed{seed}" if seed is not None else scenario_id
    path = directory / f"{timestamp}_{suffix}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
