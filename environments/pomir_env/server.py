"""FastAPI server for the SPECTRA orchestrator."""

from __future__ import annotations

import argparse
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from environments.shared.openenv_compat import create_app
from environments.shared.scenarios import baseline_plan_for_scenario, list_scenarios
from environments.shared.state import CommanderAction
from runtime.env import load_runtime_env

from .env import POMIREnv, POMIRObservation

load_runtime_env()
_RUNTIME_ENV = POMIREnv()


class ScenarioCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    name: str
    difficulty: str
    domain: str
    description: str
    common_trap: str
    best_followup_agent: str | None = None
    step_budget: int


class ScenarioCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenarios: list[ScenarioCard] = Field(default_factory=list)


class BaselinePlanCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    name: str
    steps: list[dict[str, Any]] = Field(default_factory=list)


class BaselineCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baselines: list[BaselinePlanCard] = Field(default_factory=list)


class RuntimeStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    scenario_name: str
    workflow_stage: str
    done: bool
    progress_flags: dict[str, bool] = Field(default_factory=dict)
    allowed_actions: list[str] = Field(default_factory=list)
    reward_breakdown: dict[str, float | str | int] = Field(default_factory=dict)
    round_history_length: int = 0
    last_action_result: str = ""


class PlanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: dict[str, Any]
    execution: dict[str, Any]


app = create_app(
    lambda: _RUNTIME_ENV,
    CommanderAction,
    POMIRObservation,
    env_name="spectra_main",
    max_concurrent_envs=1,
)
app.router.routes = [
    route
    for route in app.router.routes
    if not (getattr(route, "path", None) == "/health")
]


@app.get("/tasks", response_model=ScenarioCatalog, tags=["challenge"])
def tasks(difficulty: str | None = None) -> ScenarioCatalog:
    scenarios = list_scenarios()
    if difficulty is not None:
        scenarios = [scenario for scenario in scenarios if scenario.difficulty == difficulty]
        if not scenarios:
            raise HTTPException(status_code=404, detail=f"No scenarios found for difficulty={difficulty}")
    return ScenarioCatalog(
        scenarios=[
            ScenarioCard(
                scenario_id=scenario.scenario_id,
                name=scenario.name,
                difficulty=scenario.difficulty,
                domain=scenario.domain,
                description=scenario.description,
                common_trap=scenario.common_trap,
                best_followup_agent=scenario.best_followup_agent,
                step_budget=scenario.step_budget,
            )
            for scenario in scenarios
        ]
    )


@app.get("/baseline", response_model=BaselineCatalog, tags=["challenge"])
def baseline(scenario_id: str | None = None) -> BaselineCatalog:
    scenarios = list_scenarios()
    if scenario_id is not None:
        scenarios = [scenario for scenario in scenarios if scenario.scenario_id == scenario_id]
        if not scenarios:
            raise HTTPException(status_code=404, detail=f"Unknown scenario_id={scenario_id}")
    return BaselineCatalog(
        baselines=[
            BaselinePlanCard(
                scenario_id=scenario.scenario_id,
                name=scenario.name,
                steps=baseline_plan_for_scenario(scenario.scenario_id),
            )
            for scenario in scenarios
        ]
    )


@app.get("/status", response_model=RuntimeStatus, tags=["challenge"])
def status() -> RuntimeStatus:
    state = _RUNTIME_ENV.state
    master_state = state.master_state
    return RuntimeStatus(
        scenario_id=str(master_state.get("scenario_id", "")),
        scenario_name=str(master_state.get("scenario_name", "")),
        workflow_stage=state.workflow_stage,
        done=state.done,
        progress_flags=state.progress_flags,
        allowed_actions=state.allowed_actions,
        reward_breakdown=state.reward_breakdown,
        round_history_length=int(master_state.get("round_history_length", len(state.round_history))),
        last_action_result=str(master_state.get("last_action_result", "")),
    )


@app.post("/plan", response_model=PlanResponse, tags=["challenge"])
def plan() -> PlanResponse:
    decision = _RUNTIME_ENV.plan_next_action()
    return PlanResponse(
        action=decision.action.model_dump(),
        execution=decision.execution.model_dump(),
    )


@app.get("/health", tags=["challenge"])
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "environment": "spectra_main",
        "version": "0.2.0",
        "mode": _RUNTIME_ENV.mode,
        "scenarios": [scenario.scenario_id for scenario in list_scenarios()],
        "stages": [
            "triage",
            "containment",
            "remediation",
            "recovery",
            "retrospective",
            "done",
        ],
    }


def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
