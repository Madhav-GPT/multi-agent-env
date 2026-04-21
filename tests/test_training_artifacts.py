from __future__ import annotations

from training.dataset_builder import StepRecord, write_step_records
from training.grpo_train import load_grpo_dataset
from training.hint_builder import build_hint_pack, render_hint_prefix


def _sample_records() -> list[StepRecord]:
    shared = dict(
        runtime="local",
        commander_backend="llm:hf",
        commander_model="Qwen/Qwen2.5-3B-Instruct",
        observation_mode="multi_agent",
        specialist_mode="llm",
        episode_id="episode_1",
        episode_index=1,
        scenario_id="broken_auth_cascade",
        scenario_name="Broken Auth Cascade",
        difficulty="hard",
        seed=42,
        specialist_reports=[
            {
                "agent_id": "infra",
                "top_hypothesis_service": "cache",
                "supporting_evidence": ["cache CPU at 94%", "cache latency over 900ms"],
            },
            {
                "agent_id": "log",
                "top_hypothesis_service": "auth_service",
                "supporting_evidence": ["JWT verification failed repeatedly"],
            },
            {
                "agent_id": "security",
                "top_hypothesis_service": "auth_service",
                "supporting_evidence": ["CVE-2023-45812 detected on auth_service"],
            },
        ],
    )
    return [
        StepRecord(
            prompt="prompt-1",
            completion='{"action_type":"request_followup","target_agent":"security"}',
            reference_action={"action_type": "request_followup", "target_agent": "security"},
            reward=0.35,
            reward_breakdown={"total": 0.35, "r1_resolution": 0.0},
            step_index=1,
            workflow_stage="triage",
            **shared,
        ),
        StepRecord(
            prompt="prompt-2",
            completion='{"action_type":"isolate_service","target_service":"auth_service"}',
            reference_action={"action_type": "isolate_service", "target_service": "auth_service"},
            reward=0.15,
            reward_breakdown={"total": 0.15, "r1_resolution": 0.0},
            step_index=2,
            workflow_stage="containment",
            prior_actions=[{"action_type": "request_followup", "target_agent": "security"}],
            **shared,
        ),
        StepRecord(
            prompt="prompt-3",
            completion='{"action_type":"restart_service","target_service":"auth_service"}',
            reference_action={"action_type": "restart_service", "target_service": "auth_service"},
            reward=0.55,
            reward_breakdown={"total": 0.55, "r1_resolution": 0.5},
            step_index=3,
            workflow_stage="recovery",
            prior_actions=[
                {"action_type": "request_followup", "target_agent": "security"},
                {"action_type": "isolate_service", "target_service": "auth_service"},
            ],
            **shared,
        ),
    ]


def _sample_record() -> StepRecord:
    return _sample_records()[0]


def test_hint_pack_contains_trace_derived_scenario_profiles() -> None:
    hint_pack = build_hint_pack(_sample_records())

    assert "broken_auth_cascade" in hint_pack["scenario_profiles"]
    rendered = render_hint_prefix(hint_pack, scenario_id="broken_auth_cascade")
    assert "Dominant target from successful runs: auth_service" in rendered
    assert "Preferred follow-up agent: security" in rendered
    assert "Required recovery action: restart_service(auth_service)" in rendered
    assert (
        "Successful sequence seen most: request_followup(security) -> isolate_service(auth_service) -> restart_service(auth_service)"
        in rendered
    )
    assert "infra->cache" in rendered
    assert "CVE-2023-45812 detected on auth_service" in rendered


def test_load_grpo_dataset_reads_jsonl(tmp_path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    write_step_records(dataset_path, [_sample_record()])

    dataset = load_grpo_dataset(dataset_path)

    assert len(dataset) == 1
    assert dataset[0]["scenario_id"] == "broken_auth_cascade"
    assert dataset[0]["prior_actions"] == []
