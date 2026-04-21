"""Build reusable hint packs from recorded SPECTRA runs without oracle leakage."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from .dataset_builder import StepRecord, summarize_records


def _render_action(action: dict[str, Any]) -> str:
    action_type = action.get("action_type", "")
    if action_type == "request_followup":
        return f"request_followup({action.get('target_agent', 'unknown')})"
    if action_type == "submit_resolution":
        return "submit_resolution(summary)"
    target_service = action.get("target_service", "unknown")
    return f"{action_type}({target_service})"


def _action_target(action: dict[str, Any]) -> str | None:
    if "target_service" in action and action["target_service"]:
        return str(action["target_service"])
    if "target_agent" in action and action["target_agent"]:
        return str(action["target_agent"])
    return None


def _conflict_signature(reports: list[dict[str, Any]]) -> str | None:
    if len(reports) < 2:
        return None
    targets = {report.get("top_hypothesis_service") for report in reports}
    if len(targets) < 2:
        return None

    def _agent_name(raw_agent: Any) -> str:
        value = str(raw_agent)
        return value.split(".", 1)[1].lower() if value.startswith("AgentID.") else value

    ordered = sorted(
        (
            f"{_agent_name(report.get('agent_id', 'unknown'))}->{report.get('top_hypothesis_service', 'unknown')}"
            for report in reports
        )
    )
    return " | ".join(ordered)


def _stage_action_hints(records: list[StepRecord]) -> dict[str, str]:
    by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        by_stage[record.workflow_stage][_render_action(record.reference_action)] += 1
    return {
        stage: counter.most_common(1)[0][0]
        for stage, counter in by_stage.items()
        if counter
    }


def build_hint_pack(records: list[StepRecord]) -> dict[str, Any]:
    summaries = summarize_records(records)
    summary_by_episode = {summary.episode_id: summary for summary in summaries}
    records_by_scenario: dict[str, list[StepRecord]] = defaultdict(list)
    for record in records:
        records_by_scenario[record.scenario_id].append(record)

    scenario_profiles: dict[str, Any] = {}
    for scenario_id, scenario_records in sorted(records_by_scenario.items()):
        ordered_records = sorted(scenario_records, key=lambda item: (item.episode_index, item.step_index))
        resolved_records = [
            record
            for record in ordered_records
            if summary_by_episode.get(record.episode_id) is not None
            and summary_by_episode[record.episode_id].incident_resolved
        ]
        resolved_summaries = [
            summary
            for summary in summaries
            if summary.scenario_id == scenario_id and summary.incident_resolved
        ]

        sequence_counter = Counter(
            " -> ".join(summary.actions)
            for summary in resolved_summaries
            if summary.actions
        )
        recommended_sequence = (
            sequence_counter.most_common(1)[0][0].split(" -> ")
            if sequence_counter
            else []
        )

        dominant_target_counter = Counter(
            target
            for record in resolved_records
            for target in [_action_target(record.reference_action)]
            if target and record.reference_action.get("action_type") not in {"request_followup", "submit_resolution"}
        )
        followup_counter = Counter(
            str(record.reference_action.get("target_agent"))
            for record in resolved_records
            if record.reference_action.get("action_type") == "request_followup"
            and record.reference_action.get("target_agent")
        )
        recovery_counter = Counter(
            _render_action(record.reference_action)
            for record in resolved_records
            if record.reference_action.get("action_type") in {"restart_service", "scale_service"}
        )
        evidence_counter = Counter(
            evidence
            for record in resolved_records
            for report in record.specialist_reports
            for evidence in report.get("supporting_evidence", [])[:2]
            if evidence
        )
        conflict_counter = Counter(
            signature
            for record in resolved_records
            for signature in [_conflict_signature(record.specialist_reports)]
            if signature
        )

        first_record = ordered_records[0]
        scenario_profiles[scenario_id] = {
            "scenario_name": first_record.scenario_name,
            "difficulty": first_record.difficulty,
            "episodes_seen": len({record.episode_id for record in ordered_records}),
            "resolved_episodes": len({record.episode_id for record in resolved_records}),
            "dominant_target_service": (
                dominant_target_counter.most_common(1)[0][0] if dominant_target_counter else None
            ),
            "preferred_followup_agent": (
                followup_counter.most_common(1)[0][0] if followup_counter else None
            ),
            "required_recovery_action": (
                recovery_counter.most_common(1)[0][0] if recovery_counter else None
            ),
            "recommended_action_sequence": recommended_sequence,
            "stage_action_hints": _stage_action_hints(resolved_records),
            "conflict_signatures": [item for item, _ in conflict_counter.most_common(3)],
            "evidence_snippets": [item for item, _ in evidence_counter.most_common(5)],
        }

    prompt_prefix = "\n".join(
        [
            "Use the following notes only as compact lessons from prior successful trajectories.",
            "These hints are distilled from past runs, not hidden ground truth.",
            "If current specialist evidence conflicts with a hint, trust the current evidence.",
            "Do not submit_resolution before the required recovery action succeeds.",
        ]
    )

    return {
        "format_version": 2,
        "prompt_prefix": prompt_prefix,
        "episode_count": len(summaries),
        "resolved_episodes": sum(1 for summary in summaries if summary.incident_resolved),
        "scenario_profiles": scenario_profiles,
    }


def hint_digest(hint_pack: dict[str, Any]) -> str:
    payload = json.dumps(hint_pack, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def render_hint_prefix(hint_pack: dict[str, Any], *, scenario_id: str | None = None) -> str:
    prefix = [hint_pack.get("prompt_prefix", "").strip()]
    profiles = hint_pack.get("scenario_profiles", {})

    if scenario_id and scenario_id in profiles:
        profile = profiles[scenario_id]
        recommended_sequence = " -> ".join(profile.get("recommended_action_sequence", []))
        prefix.extend(
            [
                f"Scenario profile: {profile.get('scenario_name', scenario_id)} ({profile.get('difficulty', 'unknown')})",
                f"Episodes seen: {profile.get('episodes_seen', 0)} | resolved: {profile.get('resolved_episodes', 0)}",
            ]
        )
        if profile.get("dominant_target_service"):
            prefix.append(f"Dominant target from successful runs: {profile['dominant_target_service']}")
        if profile.get("preferred_followup_agent"):
            prefix.append(f"Preferred follow-up agent: {profile['preferred_followup_agent']}")
        if profile.get("required_recovery_action"):
            prefix.append(f"Required recovery action: {profile['required_recovery_action']}")
        if recommended_sequence:
            prefix.append(f"Successful sequence seen most: {recommended_sequence}")
        stage_hints = profile.get("stage_action_hints", {})
        if stage_hints:
            prefix.append("Stage action hints:")
            for stage, action in stage_hints.items():
                prefix.append(f"- {stage}: {action}")
        conflicts = profile.get("conflict_signatures", [])
        if conflicts:
            prefix.append("Common conflict patterns:")
            for item in conflicts[:2]:
                prefix.append(f"- {item}")
        evidence = profile.get("evidence_snippets", [])
        if evidence:
            prefix.append("Recurring evidence from successful runs:")
            for item in evidence[:4]:
                prefix.append(f"- {item}")
    else:
        prefix.append("Known scenario profiles:")
        for scenario_key, profile in sorted(profiles.items()):
            dominant_target = profile.get("dominant_target_service") or "unknown"
            resolved = profile.get("resolved_episodes", 0)
            prefix.append(f"- {scenario_key}: target={dominant_target} | resolved_runs={resolved}")
    return "\n".join(line for line in prefix if line)


def write_hint_pack(path: str | Path, hint_pack: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(hint_pack, indent=2, ensure_ascii=True), encoding="utf-8")
    return output_path
