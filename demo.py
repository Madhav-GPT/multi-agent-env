#!/usr/bin/env python3
"""Council-style terminal experience for the full SPECTRA pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.pomir_env.env import POMIRObservation
from environments.shared.scenarios import ScenarioBlueprint, list_scenarios
from environments.shared.state import CommanderAction, CommanderExecution
from inference import LLMCommander, LocalRuntime, RemoteRuntime, build_prompt
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
from training.hint_builder import build_hint_pack, hint_digest, write_hint_pack

PHASE_ORDER = ("untrained", "multi_agent", "hinted")
PHASE_META = {
    "untrained": {
        "index": "01",
        "nav": "BLIND COMMANDER",
        "banner": "FLYING BLIND",
        "subtitle": "full-state commander with no specialist support",
        "style": "red",
        "title": "Blind Commander",
    },
    "multi_agent": {
        "index": "02",
        "nav": "THE COUNCIL",
        "banner": "COUNCIL ACTIVE",
        "subtitle": "three specialists synthesize conflicting evidence",
        "style": "cyan",
        "title": "The Council",
    },
    "hinted": {
        "index": "03",
        "nav": "HINTED RUN",
        "banner": "CHEAT SHEET LIVE",
        "subtitle": "local commander reuses the council's exported knowledge",
        "style": "green",
        "title": "Hinted Command",
    },
}


def truncate(value: Any, limit: int) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


def _agent_name(value: Any) -> str:
    text = str(value)
    if text.startswith("AgentID."):
        return text.split(".", 1)[1].lower()
    return text.lower()


def _format_summary(summary: EpisodeSummary | None) -> str:
    if summary is None:
        return "-"
    status = "OK" if summary.incident_resolved else "FAIL"
    return f"{status} {summary.cumulative_reward:.3f}/{summary.steps}"


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _bar(value: float, *, max_value: float, width: int = 16) -> str:
    if max_value <= 0:
        max_value = 1.0
    ratio = max(0.0, min(value / max_value, 1.0))
    filled = int(round(ratio * width))
    filled = max(0, min(filled, width))
    return f"{'█' * filled}{'░' * (width - filled)} {value:.3f}"


def _sparkline(values: list[float], *, width: int = 18, max_value: float | None = None) -> str:
    if not values:
        return "-"
    blocks = "▁▂▃▄▅▆▇█"
    if width > 0 and len(values) > width:
        values = values[-width:]
    ceiling = max_value if max_value is not None else max(values) or 1.0
    if ceiling <= 0:
        ceiling = 1.0
    chars = []
    for value in values:
        ratio = max(0.0, min(value / ceiling, 1.0))
        index = min(int(round(ratio * (len(blocks) - 1))), len(blocks) - 1)
        chars.append(blocks[index])
    return "".join(chars)


@dataclass
class PhaseSpec:
    slug: str
    title: str
    observation_mode: str
    specialist_mode: str
    commander_provider: str
    commander_model: str
    commander_base_url: str | None
    commander_api_key: str | None
    commander_hf_provider: str | None
    output_dir: Path
    dataset_path: Path
    summary_path: Path
    trace_dir: Path
    hint_file: Path | None = None
    export_hint_file: Path | None = None


@dataclass
class DashboardState:
    scenario_order: list[str] = field(default_factory=list)
    scenario_labels: dict[str, str] = field(default_factory=dict)
    phase_results: dict[str, dict[str, EpisodeSummary]] = field(
        default_factory=lambda: {phase: {} for phase in PHASE_ORDER}
    )
    artifact_paths: dict[str, dict[str, str]] = field(default_factory=dict)
    recent_events: list[str] = field(default_factory=list)
    phase_slug: str = ""
    phase_title: str = ""
    scenario_id: str = ""
    scenario_name: str = ""
    scenario_index: int = 0
    total_scenarios: int = 0
    seed: int = 0
    round_index: int = 0
    workflow_stage: str = ""
    step_budget: int = 0
    observation_mode: str = ""
    specialist_mode: str = ""
    commander_model: str = ""
    commander_provider: str = ""
    cumulative_reward: float = 0.0
    reward_total: float = 0.0
    current_observation: POMIRObservation | None = None
    current_reports: list[dict[str, Any]] = field(default_factory=list)
    current_specialist_executions: list[dict[str, Any]] = field(default_factory=list)
    current_prompt_lines: list[str] = field(default_factory=list)
    commander_action: str = "awaiting commander"
    commander_raw: str = ""
    commander_latency_ms: float = 0.0
    commander_parse_status: str = "ok"
    environment_feedback: str = ""
    stage_goal: str = ""
    common_trap: str = ""
    difficulty: str = ""
    current_reward_history: list[float] = field(default_factory=list)
    current_action_history: list[str] = field(default_factory=list)
    hint_profile: dict[str, Any] | None = None
    final_note: str = ""


class CouncilDisplay:
    def __init__(self, console: Console, *, live_mode: bool) -> None:
        self.console = console
        self.live_mode = live_mode
        self.state = DashboardState()
        self._live: Live | None = None

    def __enter__(self) -> "CouncilDisplay":
        if self.live_mode:
            self._live = Live(self.render(), console=self.console, screen=True, refresh_per_second=8)
            self._live.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._live is not None:
            self._live.stop()

    def add_event(self, message: str) -> None:
        self.state.recent_events.append(message)
        self.state.recent_events = self.state.recent_events[-12:]
        if not self.live_mode:
            self.console.print(message)
        self.refresh()

    def refresh(self) -> None:
        if self._live is not None:
            self._live.update(self.render(), refresh=True)

    def render(self) -> Layout:
        if self.state.phase_slug == "multi_agent":
            return self._render_council_layout()
        if self.state.phase_slug == "hinted":
            return self._render_hinted_layout()
        return self._render_untrained_layout()

    def _render_untrained_layout(self) -> Layout:
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="header", size=6),
            Layout(name="top", ratio=5),
            Layout(name="commander", size=12),
            Layout(name="footer", size=14),
        )
        layout["top"].split_row(
            Layout(name="mission", ratio=2),
            Layout(name="world", ratio=3),
        )
        layout["footer"].split_row(
            Layout(name="matrix", ratio=3),
            Layout(name="timeline", ratio=2),
        )
        layout["header"].update(self._build_header_panel())
        layout["mission"].update(self._build_mission_panel())
        layout["world"].update(self._build_world_panel())
        layout["commander"].update(self._build_single_agent_commander_panel(blind=True))
        layout["matrix"].update(self._build_matrix_panel())
        layout["timeline"].update(self._build_timeline_panel())
        return layout

    def _render_council_layout(self) -> Layout:
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="header", size=6),
            Layout(name="main", ratio=7),
            Layout(name="footer", size=14),
        )
        layout["main"].split_row(
            Layout(name="left", ratio=3),
            Layout(name="center", ratio=2),
            Layout(name="right", ratio=3),
        )
        layout["center"].split_column(
            Layout(name="center_top", ratio=3),
            Layout(name="center_bottom", ratio=4),
        )
        layout["footer"].split_row(
            Layout(name="matrix", ratio=3),
            Layout(name="timeline", ratio=2),
        )

        layout["header"].update(self._build_header_panel())
        if self.state.observation_mode == "multi_agent":
            layout["left"].update(self._build_specialist_panel("security"))
            layout["center_top"].update(self._build_specialist_panel("infra"))
            layout["right"].update(self._build_specialist_panel("log"))
        else:
            layout["left"].update(self._build_mission_panel())
            layout["center_top"].update(self._build_world_panel())
            layout["right"].update(self._build_hint_panel())
        layout["center_bottom"].update(self._build_commander_panel())
        layout["matrix"].update(self._build_matrix_panel())
        layout["timeline"].update(self._build_timeline_panel())
        return layout

    def _render_hinted_layout(self) -> Layout:
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="header", size=6),
            Layout(name="main", ratio=6),
            Layout(name="footer", size=16),
        )
        layout["main"].split_row(
            Layout(name="hint", ratio=2),
            Layout(name="commander", ratio=3),
        )
        layout["footer"].split_row(
            Layout(name="matrix", ratio=3),
            Layout(name="timeline", ratio=2),
        )
        layout["header"].update(self._build_header_panel())
        layout["hint"].update(self._build_hint_panel())
        layout["commander"].update(self._build_single_agent_commander_panel(blind=False))
        layout["matrix"].update(self._build_matrix_panel())
        layout["timeline"].update(self._build_timeline_panel())
        return layout

    def _phase_meta(self) -> dict[str, str]:
        return PHASE_META.get(
            self.state.phase_slug or "untrained",
            PHASE_META["untrained"],
        )

    def _build_header_panel(self) -> Panel:
        meta = self._phase_meta()
        phase_counts = []
        for phase in PHASE_ORDER:
            entries = list(self.state.phase_results[phase].values())
            resolved = sum(1 for item in entries if item.incident_resolved)
            phase_counts.append(f"{phase}:{resolved}/{len(entries)}")
        nav = Text()
        for phase in PHASE_ORDER:
            phase_meta = PHASE_META[phase]
            label = f"[{phase_meta['index']}] {phase_meta['nav']}"
            style = (
                f"bold white on {phase_meta['style']}"
                if phase == self.state.phase_slug
                else "dim white"
            )
            nav.append(f" {label} ", style=style)
            nav.append("  ")

        scenario_line = (
            f"scenario {self.state.scenario_index}/{self.state.total_scenarios} "
            f"{self.state.scenario_id or '-'}"
        )
        detail_line = (
            f"seed={self.state.seed}  difficulty={self.state.difficulty or '-'}  "
            f"stage={self.state.workflow_stage or '-'}  steps_left={self.state.step_budget}  "
            f"mode={self.state.observation_mode or '-'}  specialists={self.state.specialist_mode or '-'}"
        )
        phase_line = (
            f"{meta['banner']}  |  {self.state.phase_title or meta['title']}  |  "
            f"{meta['subtitle']}"
        )
        stats_line = f"commander={self.state.commander_model or '-'}  stats={' | '.join(phase_counts)}"
        body = Group(
            nav,
            Text(phase_line, style=f"bold {meta['style']}"),
            Text(scenario_line, style="white"),
            Text(f"{detail_line}  |  {stats_line}", style="dim"),
        )
        return Panel(body, title="SPECTRA | 3-STAGE COMMAND ARC", border_style=meta["style"], box=box.DOUBLE)

    def _build_specialist_panel(self, agent: str) -> Panel:
        report = next((item for item in self.state.current_reports if _agent_name(item.get("agent_id")) == agent), None)
        execution = next(
            (item for item in self.state.current_specialist_executions if str(item.get("agent")) == agent),
            None,
        )
        title_map = {
            "infra": "INFRA LENS",
            "log": "LOG LENS",
            "security": "SECURITY LENS",
        }
        style_map = {
            "infra": "bright_blue",
            "log": "bright_yellow",
            "security": "bright_red",
        }
        if report is None or execution is None:
            return Panel("awaiting council", title=title_map[agent], border_style=style_map[agent], box=box.HEAVY)

        evidence = report.get("supporting_evidence", [])[:3]
        lines = [
            f"channel={execution.get('channel')} mode={execution.get('mode')} latency={execution.get('latency_ms')}ms",
            f"digest: {truncate(execution.get('observation_digest', ''), 72)}",
            f"target: {report.get('top_hypothesis_service')} conf={float(report.get('confidence', 0.0)):.2f}",
            f"meter: {_bar(float(report.get('confidence', 0.0)), max_value=1.0, width=14)}",
            f"cause: {truncate(report.get('top_hypothesis_cause', ''), 72)}",
            f"action: {truncate(report.get('recommended_action', ''), 72)}",
        ]
        if evidence:
            lines.append("evidence:")
            lines.extend(f"- {truncate(item, 64)}" for item in evidence)
        raw_excerpt = truncate(execution.get("raw_response", ""), 120)
        if raw_excerpt:
            lines.append(f"raw: {raw_excerpt}")
        return Panel(
            "\n".join(lines),
            title=title_map[agent],
            border_style=style_map[agent],
            box=box.HEAVY,
        )

    def _build_mission_panel(self) -> Panel:
        observation = self.state.current_observation
        progress = observation.progress_flags if observation is not None else {}
        progress_lines = [f"{key}={'yes' if value else 'no'}" for key, value in progress.items()]
        lines = [
            f"phase: {self.state.phase_title or '-'}",
            f"scenario: {self.state.scenario_name or self.state.scenario_id or '-'}",
            f"trap: {truncate(self.state.common_trap or '-', 88)}",
            f"goal: {truncate(self.state.stage_goal or '-', 88)}",
            "progress:",
            *(progress_lines or ["- none yet"]),
        ]
        title = "MISSION"
        if self.state.phase_slug == "untrained":
            title = "MISSION | BLIND BASELINE"
        return Panel("\n".join(lines), title=title, border_style="magenta", box=box.HEAVY)

    def _build_world_panel(self) -> Panel:
        lines = self.state.current_prompt_lines[:12] or ["awaiting full-state observation"]
        title = "WORLD SNAPSHOT"
        if self.state.phase_slug == "untrained":
            title = "WORLD SNAPSHOT | FULL STATE ONLY"
        return Panel("\n".join(lines), title=title, border_style="bright_blue", box=box.HEAVY)

    def _build_hint_panel(self) -> Panel:
        if self.state.phase_slug != "hinted":
            body = "No cheat sheet in this phase.\nThis is the blind baseline."
            return Panel(body, title="CHEAT SHEET", border_style="yellow", box=box.HEAVY)

        profile = self.state.hint_profile or {}
        lines = [
            f"scenario: {self.state.scenario_id or '-'}",
            f"source: council export from multi-agent phase",
        ]
        if profile.get("dominant_target_service"):
            lines.append(f"target: {profile['dominant_target_service']}")
        if profile.get("preferred_followup_agent"):
            lines.append(f"followup: {profile['preferred_followup_agent']}")
        if profile.get("required_recovery_action"):
            lines.append(f"recovery: {profile['required_recovery_action']}")
        sequence = profile.get("recommended_action_sequence", [])
        if sequence:
            lines.append("sequence:")
            lines.extend(f"- {truncate(item, 56)}" for item in sequence[:5])
        conflicts = profile.get("conflict_signatures", [])
        if conflicts:
            lines.append("conflicts:")
            lines.extend(f"- {truncate(item, 56)}" for item in conflicts[:2])
        evidence = profile.get("evidence_snippets", [])
        if evidence:
            lines.append("evidence:")
            lines.extend(f"- {truncate(item, 56)}" for item in evidence[:3])
        if not lines:
            lines = ["hint pack loaded", "no scenario-specific profile yet"]
        return Panel("\n".join(lines), title="CHEAT SHEET | COUNCIL DISTILLED", border_style="yellow", box=box.HEAVY)

    def _current_reward_total(self) -> float:
        observation = self.state.current_observation
        if observation is None:
            return 0.0
        return float(observation.reward_breakdown.get("total", 0.0))

    def _reward_breakdown_line(self) -> str:
        observation = self.state.current_observation
        if observation is None:
            return "-"
        parts = []
        for key, value in observation.reward_breakdown.items():
            if key == "total" or not isinstance(value, (int, float)):
                continue
            sign = "+" if float(value) >= 0 else ""
            parts.append(f"{key}={sign}{float(value):.3f}")
        return ", ".join(parts) if parts else "-"

    def _latest_action_lines(self) -> list[str]:
        reward_total = self._current_reward_total()
        emphasis = "WRONG OR LOW-VALUE ACTION" if reward_total <= 0.05 else "ACTION MOVED THE INCIDENT FORWARD"
        if self.state.phase_slug == "hinted":
            emphasis = "COUNCIL KNOWLEDGE APPLIED"
        lines = [
            f"action: {self.state.commander_action}",
            f"status: {emphasis}",
            f"parse={self.state.commander_parse_status} latency={self.state.commander_latency_ms:.2f}ms",
            f"reward={reward_total:.3f} cumulative={self.state.cumulative_reward:.3f}",
            f"feedback: {truncate(self.state.environment_feedback or '-', 120)}",
            f"reward_breakdown: {self._reward_breakdown_line()}",
        ]
        if self.state.commander_raw:
            lines.append(f"raw: {truncate(self.state.commander_raw, 200)}")
        return lines

    def _current_episode_chart(self) -> Table:
        table = Table(expand=True, box=box.SIMPLE, show_header=False, pad_edge=False)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value")
        current_reward = self._current_reward_total()
        table.add_row("step_reward", _bar(current_reward, max_value=1.0, width=18))
        table.add_row("cumulative", _bar(self.state.cumulative_reward, max_value=1.6, width=18))
        table.add_row("trend", _sparkline(self.state.current_reward_history, width=18, max_value=1.0))
        recent_actions = ", ".join(self.state.current_action_history[-3:]) or "-"
        table.add_row("recent_actions", truncate(recent_actions, 60))
        return table

    def _phase_chart_table(self) -> Table:
        table = Table(expand=True, box=box.SIMPLE, show_header=True)
        table.add_column("Phase", style="cyan", no_wrap=True)
        table.add_column("Reward")
        table.add_column("Steps")
        table.add_column("Success", no_wrap=True)
        stats_by_phase: dict[str, dict[str, float | int]] = {}
        reward_max = 1.0
        for phase in PHASE_ORDER:
            entries = list(self.state.phase_results[phase].values())
            stats = _phase_stats(entries)
            stats_by_phase[phase] = stats
            reward_max = max(reward_max, float(stats["mean_reward"]))
        reward_max = max(reward_max, 1.6)
        for phase in PHASE_ORDER:
            stats = stats_by_phase[phase]
            table.add_row(
                PHASE_META[phase]["index"] + " " + PHASE_META[phase]["nav"],
                _bar(float(stats["mean_reward"]), max_value=reward_max, width=16),
                f"{float(stats['mean_steps']):.1f}",
                f"{int(stats['resolved'])}/{int(stats['episodes'])}",
            )
        return table

    def _trust_chart_table(self) -> Table:
        table = Table(expand=True, box=box.SIMPLE, show_header=False, pad_edge=False)
        table.add_column("Lens", style="cyan", no_wrap=True)
        table.add_column("Trust")
        ordered = sorted(
            self.state.current_reports,
            key=lambda item: float(item.get("confidence", 0.0)),
            reverse=True,
        )
        if not ordered:
            table.add_row("trust", "-")
            return table
        for report in ordered:
            agent = _agent_name(report.get("agent_id"))
            confidence = float(report.get("confidence", 0.0))
            target = report.get("top_hypothesis_service")
            table.add_row(agent, f"{_bar(confidence, max_value=1.0, width=12)} -> {target}")
        return table

    def _build_single_agent_commander_panel(self, *, blind: bool) -> Panel:
        subtitle = "COMMANDER | NO SPECIALIST INPUT" if blind else "HINTED COMMANDER | COUNCIL REUSE"
        lines = [
            f"provider={self.state.commander_provider or '-'} model={self.state.commander_model or '-'}",
            f"goal: {truncate(self.state.stage_goal or '-', 96)}",
            f"trap: {truncate(self.state.common_trap or '-', 96)}",
            *self._latest_action_lines(),
        ]
        observation_lines = self.state.current_prompt_lines[:8]
        if observation_lines:
            lines.append("snapshot:")
            lines.extend(f"- {truncate(item, 88)}" for item in observation_lines)
        border = "red" if blind else "green"
        return Panel(
            Group("\n".join(lines), self._current_episode_chart()),
            title=subtitle,
            border_style=border,
            box=box.DOUBLE,
        )

    def _build_commander_panel(self) -> Panel:
        consensus = []
        for report in self.state.current_reports:
            consensus.append(
                f"{_agent_name(report.get('agent_id'))}->{report.get('top_hypothesis_service')} "
                f"{float(report.get('confidence', 0.0)):.2f}"
            )
        lines = [
            f"provider={self.state.commander_provider or '-'} model={self.state.commander_model or '-'}",
            f"goal: {truncate(self.state.stage_goal or '-', 96)}",
            *self._latest_action_lines(),
        ]
        if consensus and self.state.observation_mode == "multi_agent":
            lines.append("council:")
            lines.extend(f"- {item}" for item in consensus)
        return Panel(
            Group("\n".join(lines), self._trust_chart_table(), self._current_episode_chart()),
            title="COMMANDER | COUNCIL SYNTHESIS",
            border_style="bright_white",
            box=box.DOUBLE,
        )

    def _build_matrix_panel(self) -> Panel:
        table = Table(expand=True, box=box.SIMPLE_HEAVY)
        table.add_column("Scenario", style="cyan", no_wrap=True)
        table.add_column("Untrained", no_wrap=True)
        table.add_column("Multi-Agent", no_wrap=True)
        table.add_column("Hinted", no_wrap=True)
        for scenario_id in self.state.scenario_order:
            label = self.state.scenario_labels.get(scenario_id, scenario_id)
            available = {
                phase: self.state.phase_results[phase].get(scenario_id)
                for phase in PHASE_ORDER
                if self.state.phase_results[phase].get(scenario_id) is not None
            }
            best_phase = None
            if available:
                best_phase = max(
                    available,
                    key=lambda phase: (
                        available[phase].incident_resolved,
                        available[phase].cumulative_reward,
                        -available[phase].steps,
                    ),
                )
            row: list[Any] = [truncate(label, 26)]
            for phase in PHASE_ORDER:
                summary = self.state.phase_results[phase].get(scenario_id)
                if summary is None:
                    row.append(Text("—", style="dim"))
                    continue
                style = "white"
                if not summary.incident_resolved:
                    style = "bold red"
                elif phase == best_phase:
                    style = "bold bright_green"
                elif phase == self.state.phase_slug:
                    style = "bold bright_cyan"
                row.append(Text(_format_summary(summary), style=style))
            table.add_row(*row)
        footer = self.state.final_note or "pipeline active"
        title = "SCENARIO MATRIX"
        if self.state.phase_slug == "hinted":
            title = "FINAL COMPARISON MATRIX"
        return Panel(
            Group(table, self._phase_chart_table(), Text(footer, style="dim")),
            title=title,
            border_style="green",
        )

    def _build_timeline_panel(self) -> Panel:
        events = self.state.recent_events or ["awaiting events"]
        body = "\n".join(events[-12:])
        title = "TIMELINE"
        if self.state.phase_slug == "multi_agent":
            title = "COUNCIL TIMELINE"
        elif self.state.phase_slug == "hinted":
            title = "REUSE TIMELINE"
        elif self.state.phase_slug == "untrained":
            title = "BLIND TIMELINE"
        return Panel(body, title=title, border_style="bright_black", box=box.HEAVY)


def _phase_stats(summaries: list[EpisodeSummary]) -> dict[str, float | int]:
    return {
        "episodes": len(summaries),
        "resolved": sum(1 for item in summaries if item.incident_resolved),
        "success_rate": round(sum(1 for item in summaries if item.incident_resolved) / len(summaries), 3)
        if summaries
        else 0.0,
        "mean_reward": _mean([item.cumulative_reward for item in summaries]),
        "mean_steps": _mean([float(item.steps) for item in summaries]),
    }


def _build_phase_specs(args: argparse.Namespace, output_root: Path) -> list[PhaseSpec]:
    local_api_key = os.getenv("OPENAI_API_KEY", "local")
    hf_api_key = os.getenv("HF_TOKEN")
    untrained_output = output_root / "untrained"
    multi_output = output_root / "multi_agent"
    hinted_output = output_root / "hinted"
    hint_file = args.existing_hint_file or multi_output / "hints.json"

    specs: dict[str, PhaseSpec] = {
        "untrained": PhaseSpec(
            slug="untrained",
            title="Untrained Baseline",
            observation_mode="single_agent",
            specialist_mode="deterministic",
            commander_provider="openai",
            commander_model=args.untrained_model,
            commander_base_url=args.commander_base_url,
            commander_api_key=local_api_key,
            commander_hf_provider=None,
            output_dir=untrained_output,
            dataset_path=untrained_output / "data.jsonl",
            summary_path=untrained_output / "data.summary.json",
            trace_dir=untrained_output / "traces",
        ),
        "multi_agent": PhaseSpec(
            slug="multi_agent",
            title="Multi-Agent Collection",
            observation_mode="multi_agent",
            specialist_mode=args.specialist_mode,
            commander_provider=args.multi_agent_provider,
            commander_model=args.multi_agent_model,
            commander_base_url=args.commander_base_url,
            commander_api_key=hf_api_key if args.multi_agent_provider == "hf" else local_api_key,
            commander_hf_provider=args.multi_agent_hf_provider,
            output_dir=multi_output,
            dataset_path=multi_output / "data.jsonl",
            summary_path=multi_output / "data.summary.json",
            trace_dir=multi_output / "traces",
            export_hint_file=multi_output / "hints.json",
        ),
        "hinted": PhaseSpec(
            slug="hinted",
            title="Hinted Re-Run",
            observation_mode="single_agent",
            specialist_mode="deterministic",
            commander_provider="openai",
            commander_model=args.hinted_model,
            commander_base_url=args.commander_base_url,
            commander_api_key=local_api_key,
            commander_hf_provider=None,
            output_dir=hinted_output,
            dataset_path=hinted_output / "data.jsonl",
            summary_path=hinted_output / "data.summary.json",
            trace_dir=hinted_output / "traces",
            hint_file=Path(hint_file),
        ),
    }
    return [specs[name] for name in args.phases]


def _scenario_list(scope: str, scenario_id: str | None) -> list[ScenarioBlueprint]:
    scenarios = list_scenarios()
    if scope == "single":
        if scenario_id is None:
            raise SystemExit("--scenario-id is required when --scenario-scope=single")
        for scenario in scenarios:
            if scenario.scenario_id == scenario_id:
                return [scenario]
        raise SystemExit(f"Unknown scenario_id: {scenario_id}")
    return scenarios


def _runtime_for(mode: str, base_url: str, *, specialist_mode: str, observation_mode: str):
    if mode == "local":
        return LocalRuntime(specialist_mode=specialist_mode, observation_mode=observation_mode)
    return RemoteRuntime(base_url=base_url)


def _build_records_for_phase(
    spec: PhaseSpec,
    scenarios: list[ScenarioBlueprint],
    *,
    seed: int,
    runtime_mode: str,
    base_url: str,
    display: CouncilDisplay,
) -> tuple[list[StepRecord], list[EpisodeSummary]]:
    hint_pack = None
    if spec.hint_file is not None:
        hint_pack = json.loads(spec.hint_file.read_text(encoding="utf-8"))

    runtime = _runtime_for(
        runtime_mode,
        base_url,
        specialist_mode=spec.specialist_mode,
        observation_mode=spec.observation_mode,
    )
    llm_commander = LLMCommander(
        model_name=spec.commander_model,
        provider=spec.commander_provider,
        base_url=spec.commander_base_url,
        api_key=spec.commander_api_key,
        hf_provider=spec.commander_hf_provider,
    )

    records: list[StepRecord] = []
    summaries: list[EpisodeSummary] = []
    try:
        for episode_index, scenario in enumerate(scenarios, start=1):
            episode_seed = seed + episode_index - 1
            observation = runtime.reset(
                difficulty=scenario.difficulty,
                scenario_id=scenario.scenario_id,
                seed=episode_seed,
                observation_mode=spec.observation_mode,
                specialist_mode=spec.specialist_mode,
            )
            display.state.phase_slug = spec.slug
            display.state.phase_title = spec.title
            display.state.scenario_id = observation.scenario_id
            display.state.scenario_name = observation.scenario_name
            display.state.scenario_index = episode_index
            display.state.total_scenarios = len(scenarios)
            display.state.seed = episode_seed
            display.state.observation_mode = spec.observation_mode
            display.state.specialist_mode = spec.specialist_mode
            display.state.commander_model = spec.commander_model
            display.state.commander_provider = spec.commander_provider
            display.state.commander_action = "awaiting commander"
            display.state.commander_raw = ""
            display.state.commander_latency_ms = 0.0
            display.state.commander_parse_status = "ok"
            display.state.environment_feedback = ""
            display.state.current_observation = observation
            display.state.current_reports = [report.model_dump() for report in observation.reports]
            display.state.current_specialist_executions = [
                execution.model_dump() for execution in observation.specialist_executions
            ]
            display.state.current_prompt_lines = observation.prompt_text.splitlines()[:14]
            display.state.workflow_stage = observation.workflow_stage
            display.state.step_budget = observation.step_budget
            display.state.difficulty = observation.difficulty
            display.state.cumulative_reward = observation.cumulative_reward
            display.state.reward_total = 0.0
            display.state.stage_goal = observation.stage_goal
            display.state.common_trap = observation.common_trap
            display.state.current_reward_history = []
            display.state.current_action_history = []
            display.state.hint_profile = (
                (hint_pack or {}).get("scenario_profiles", {}).get(observation.scenario_id)
                if hint_pack is not None
                else None
            )
            display.add_event(
                f"{spec.slug}: scenario {episode_index}/{len(scenarios)} "
                f"{observation.scenario_id} seed={episode_seed} started"
            )

            prior_actions: list[dict[str, Any]] = []
            actions_rendered: list[str] = []
            trace_steps: list[EpisodeTraceStep] = []
            round_index = 1

            while not observation.done:
                prompt, hint_id = build_prompt(observation, hint_pack=hint_pack)
                commander_reply = llm_commander.complete(
                    prompt=prompt,
                    observation_mode=spec.observation_mode,
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

                display.state.round_index = round_index
                display.state.workflow_stage = observation.workflow_stage
                display.state.step_budget = observation.step_budget
                display.state.difficulty = observation.difficulty
                display.state.current_observation = observation
                display.state.current_reports = [report.model_dump() for report in observation.reports]
                display.state.current_specialist_executions = [
                    item.model_dump() for item in observation.specialist_executions
                ]
                display.state.current_prompt_lines = observation.prompt_text.splitlines()[:14]
                display.state.stage_goal = observation.stage_goal
                display.state.commander_action = commander_reply.action.rendered
                display.state.commander_raw = commander_reply.completion
                display.state.commander_latency_ms = commander_reply.latency_ms
                display.state.commander_parse_status = commander_reply.parse_status
                display.refresh()

                next_observation = runtime.step(commander_reply.action)
                reward_total = float(next_observation.reward_breakdown.get("total", 0.0))
                display.state.current_observation = next_observation
                display.state.current_reports = [report.model_dump() for report in observation.reports]
                display.state.current_specialist_executions = [
                    item.model_dump() for item in observation.specialist_executions
                ]
                display.state.current_prompt_lines = next_observation.prompt_text.splitlines()[:14]
                display.state.workflow_stage = next_observation.workflow_stage
                display.state.step_budget = next_observation.step_budget
                display.state.difficulty = next_observation.difficulty
                display.state.cumulative_reward = next_observation.cumulative_reward
                display.state.reward_total = reward_total
                display.state.environment_feedback = next_observation.last_action_result
                display.state.stage_goal = next_observation.stage_goal
                display.state.current_reward_history.append(reward_total)
                display.state.current_action_history.append(commander_reply.action.rendered)
                display.add_event(
                    f"{spec.slug}: {observation.scenario_id} round {round_index}: "
                    f"{commander_reply.action.rendered} -> reward {reward_total:.3f} "
                    f"next={next_observation.workflow_stage}"
                )

                record_episode_id = (
                    observation.metadata.get("episode_id", observation.metadata.get("episodeId", ""))
                    or f"{spec.slug}_episode_{episode_index}"
                )
                records.append(
                    StepRecord(
                        prompt=prompt,
                        completion=commander_reply.completion,
                        reference_action=commander_reply.action.model_dump(),
                        reward=reward_total,
                        reward_breakdown=dict(next_observation.reward_breakdown),
                        runtime=runtime_mode,
                        commander_backend=f"llm:{spec.commander_provider}",
                        commander_model=spec.commander_model,
                        observation_mode=spec.observation_mode,
                        specialist_mode=spec.specialist_mode,
                        episode_id=record_episode_id,
                        episode_index=episode_index,
                        step_index=round_index,
                        scenario_id=observation.scenario_id,
                        scenario_name=observation.scenario_name,
                        difficulty=observation.difficulty,
                        workflow_stage=observation.workflow_stage,
                        seed=episode_seed,
                        allowed_actions=list(observation.allowed_actions),
                        prior_actions=list(prior_actions),
                        report_targets={
                            report.agent_id.value: report.top_hypothesis_service for report in observation.reports
                        },
                        report_confidences={
                            report.agent_id.value: report.confidence for report in observation.reports
                        },
                        specialist_reports=[report.model_dump() for report in observation.reports],
                        specialist_executions=[item.model_dump() for item in observation.specialist_executions],
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
                trace_steps.append(
                    EpisodeTraceStep(
                        step_index=round_index,
                        workflow_stage=observation.workflow_stage,
                        stage_goal=observation.stage_goal,
                        prompt=prompt,
                        allowed_actions=list(observation.allowed_actions),
                        valid_action_example=dict(observation.valid_action_example),
                        specialist_reports=[report.model_dump() for report in observation.reports],
                        specialist_executions=[item.model_dump() for item in observation.specialist_executions],
                        commander_reply={
                            "backend": f"llm:{spec.commander_provider}",
                            "model": spec.commander_model,
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

            summary = EpisodeSummary(
                episode_id=records[-1].episode_id,
                episode_index=episode_index,
                scenario_id=observation.scenario_id,
                scenario_name=observation.scenario_name,
                difficulty=observation.difficulty,
                seed=episode_seed,
                runtime=runtime_mode,
                commander_backend=f"llm:{spec.commander_provider}",
                commander_model=spec.commander_model,
                observation_mode=spec.observation_mode,
                specialist_mode=spec.specialist_mode,
                steps=round_index - 1,
                incident_resolved=observation.incident_resolved,
                cumulative_reward=observation.cumulative_reward,
                actions=actions_rendered,
            )
            summaries.append(summary)
            display.state.phase_results[spec.slug][observation.scenario_id] = summary
            display.state.final_note = (
                f"{spec.title} finished {observation.scenario_id}: "
                f"{'resolved' if observation.incident_resolved else 'unresolved'} "
                f"reward={observation.cumulative_reward:.3f}"
            )
            display.add_event(
                f"{spec.slug}: {observation.scenario_id} completed "
                f"resolved={observation.incident_resolved} reward={observation.cumulative_reward:.3f}"
            )

            trace_path = spec.trace_dir / f"{observation.scenario_id}_episode_{episode_index:02d}.trace.json"
            write_episode_trace(
                trace_path,
                EpisodeTrace(
                    episode_id=summary.episode_id,
                    episode_index=episode_index,
                    scenario_id=observation.scenario_id,
                    scenario_name=observation.scenario_name,
                    difficulty=observation.difficulty,
                    seed=episode_seed,
                    runtime=runtime_mode,
                    commander_backend=summary.commander_backend,
                    commander_model=summary.commander_model,
                    observation_mode=spec.observation_mode,
                    specialist_mode=spec.specialist_mode,
                    hint_used=hint_pack is not None,
                    hint_digest=hint_digest(hint_pack) if hint_pack is not None else None,
                    steps=trace_steps,
                ),
            )
    finally:
        runtime.close()
    return records, summaries


def _write_phase_artifacts(
    spec: PhaseSpec,
    display: CouncilDisplay,
    *,
    records: list[StepRecord],
    summaries: list[EpisodeSummary],
) -> dict[str, str]:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    write_step_records(spec.dataset_path, records)
    write_episode_summaries(spec.summary_path, summaries)
    artifacts = {
        "dataset": str(spec.dataset_path),
        "summary": str(spec.summary_path),
        "trace_dir": str(spec.trace_dir),
    }
    if spec.export_hint_file is not None:
        hint_pack = build_hint_pack(records)
        write_hint_pack(spec.export_hint_file, hint_pack)
        artifacts["hint_file"] = str(spec.export_hint_file)
        display.state.final_note = (
            f"COUNCIL COMPLETE -> exported cheat sheet to {spec.export_hint_file}"
        )
        display.add_event(f"{spec.slug}: council complete -> exporting hint pack -> {spec.export_hint_file}")
    display.state.artifact_paths[spec.slug] = artifacts
    return artifacts


def main() -> None:
    load_runtime_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-scope", default="all", choices=["all", "single"])
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runtime", default="local", choices=["local", "remote"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--commander-base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--untrained-model", default="qwen2.5:3b")
    parser.add_argument("--hinted-model", default="qwen2.5:3b")
    parser.add_argument("--multi-agent-provider", default="openai", choices=["openai", "hf"])
    parser.add_argument("--multi-agent-model", default=os.getenv("LOCAL_MULTI_MODEL", "qwen2.5:3b"))
    parser.add_argument("--multi-agent-hf-provider", default=os.getenv("COMMANDER_HF_PROVIDER", "nscale"))
    parser.add_argument("--specialist-mode", default="deterministic", choices=["deterministic", "hybrid", "llm"])
    parser.add_argument("--output-root", default="outputs/council_pipeline")
    parser.add_argument("--existing-hint-file", default=None)
    parser.add_argument("--phases", default="untrained,multi_agent,hinted")
    parser.add_argument("--no-live", action="store_true")
    args = parser.parse_args()

    args.phases = [item.strip() for item in args.phases.split(",") if item.strip()]
    unknown_phases = [item for item in args.phases if item not in PHASE_ORDER]
    if unknown_phases:
        raise SystemExit(f"Unknown phases: {', '.join(unknown_phases)}")

    scenarios = _scenario_list(args.scenario_scope, args.scenario_id)
    output_root = Path(args.output_root)
    specs = _build_phase_specs(args, output_root)

    console = Console()
    with CouncilDisplay(console, live_mode=not args.no_live) as display:
        display.state.scenario_order = [scenario.scenario_id for scenario in scenarios]
        display.state.scenario_labels = {scenario.scenario_id: scenario.name for scenario in scenarios}
        display.state.total_scenarios = len(scenarios)
        display.add_event(
            f"Council pipeline ready: phases={','.join(args.phases)} scenarios={len(scenarios)} seed={args.seed}"
        )

        report: dict[str, Any] = {
            "scenario_scope": args.scenario_scope,
            "scenario_count": len(scenarios),
            "seed": args.seed,
            "phases": {},
        }

        for spec in specs:
            if spec.slug == "hinted" and spec.hint_file is not None and not spec.hint_file.exists():
                raise SystemExit(
                    f"Hint file for hinted phase does not exist yet: {spec.hint_file}. "
                    "Run the multi_agent phase first or pass --existing-hint-file."
                )
            display.state.phase_slug = spec.slug
            display.state.phase_title = spec.title
            display.state.commander_model = spec.commander_model
            display.state.commander_provider = spec.commander_provider
            display.state.specialist_mode = spec.specialist_mode
            display.add_event(f"Starting phase {spec.slug} with commander {spec.commander_model}")

            records, summaries = _build_records_for_phase(
                spec,
                scenarios,
                seed=args.seed,
                runtime_mode=args.runtime,
                base_url=args.base_url,
                display=display,
            )
            artifacts = _write_phase_artifacts(spec, display, records=records, summaries=summaries)
            report["phases"][spec.slug] = {
                "artifacts": artifacts,
                "stats": _phase_stats(summaries),
            }

        report_path = output_root / "council_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        display.state.final_note = f"Council pipeline complete. report={report_path}"
        display.add_event(f"Council pipeline complete. report={report_path}")
        if args.no_live:
            console.print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
