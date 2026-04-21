from __future__ import annotations

from fastapi.testclient import TestClient

from environments.pomir_env.server import app


def test_server_exposes_challenge_routes() -> None:
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["environment"] == "spectra_main"

    tasks = client.get("/tasks")
    assert tasks.status_code == 200
    assert len(tasks.json()["scenarios"]) >= 5

    baseline = client.get("/baseline", params={"scenario_id": "database_sqli_outage"})
    assert baseline.status_code == 200
    assert baseline.json()["baselines"][0]["scenario_id"] == "database_sqli_outage"


def test_server_reset_plan_step_flow() -> None:
    client = TestClient(app)

    reset = client.post(
        "/reset",
        json={"scenario_id": "database_sqli_outage", "seed": 42},
    )
    assert reset.status_code == 200
    assert reset.json()["observation"]["scenario_id"] == "database_sqli_outage"

    plan = client.post("/plan")
    assert plan.status_code == 200
    assert plan.json()["action"]["action_type"] == "isolate_service"

    step = client.post("/step", json={"action": plan.json()["action"]})
    assert step.status_code == 200
    assert step.json()["observation"]["workflow_stage"] == "remediation"
