"""Meta-orchestrator environment for SPECTRA."""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
from typing import Any

from pydantic import ConfigDict, Field

from agents.commander.commander import Commander, CommanderDecision
from agents.commander.single_agent_commander import SingleAgentCommander
from agents.full_state_observation_builder import build_single_agent_observation
from agents.observation_builder import build_commander_observation
from agents.specialist_report import SpecialistReport
from agents.specialists.infra_specialist import InfraSpecialist
from agents.specialists.log_specialist import LogSpecialist
from agents.specialists.sec_specialist import SecSpecialist
from environments.infra_env.env import InfraEnv
from environments.log_env.env import LogEnv
from environments.sec_env.env import SecEnv
from environments.shared.master_env import MasterSREEnv
from environments.shared.openenv_compat import Environment, EnvironmentMetadata, Observation, State
from environments.shared.scenarios import baseline_plan_for_scenario, list_scenarios
from environments.shared.state import CommanderAction, CommanderExecution, MultiAgentRound, SpecialistRoundExecution
from rewards.composite import CompositeRewardCalculator, CompositeRewardResult

from .mode import current_mode

ALLOWED_ACTIONS_BY_STAGE: dict[str, list[str]] = {
    "triage": [
        "request_followup(agent)",
        "investigate_service(service)",
        "isolate_service(service)",
    ],
    "containment": [
        "request_followup(agent)",
        "isolate_service(service)",
        "rollback_config(service)",
    ],
    "remediation": [
        "rollback_config(service)",
        "restart_service(service)",
        "scale_service(service)",
    ],
    "recovery": [
        "restart_service(service)",
        "scale_service(service)",
        "submit_resolution(summary)",
    ],
    "retrospective": [
        "submit_resolution(summary)",
    ],
    "done": [],
}

LEGACY_SPECIALIST_MODE = {
    "training": "deterministic",
    "demo": "hybrid",
}

STAGE_GOALS: dict[str, str] = {
    "triage": "Narrow the likely root cause from the currently visible evidence.",
    "containment": "Contain the most likely root-cause service before chasing recovery.",
    "remediation": "Remove the risky config or release that is still causing the incident.",
    "recovery": "Restore service health with the scenario-appropriate recovery action.",
    "retrospective": "Submit the final incident summary now that the system is stable.",
    "done": "Episode complete.",
}

REQUIRED_FIELDS_BY_ACTION: dict[str, list[str]] = {
    "request_followup": ["target_agent"],
    "investigate_service": ["target_service"],
    "isolate_service": ["target_service"],
    "rollback_config": ["target_service"],
    "scale_service": ["target_service"],
    "restart_service": ["target_service"],
    "submit_resolution": ["resolution_summary"],
}


def _required_fields_by_action(allowed_actions: list[str]) -> dict[str, list[str]]:
    names = [action.split("(", 1)[0] for action in allowed_actions]
    return {name: list(REQUIRED_FIELDS_BY_ACTION.get(name, [])) for name in names}


class POMIRObservation(Observation):
    model_config = ConfigDict(extra="forbid")

    prompt_text: str
    workflow_stage: str
    reports: list[SpecialistReport] = Field(default_factory=list)
    specialist_executions: list[SpecialistRoundExecution] = Field(default_factory=list)
    commander_execution: CommanderExecution | None = None
    allowed_actions: list[str] = Field(default_factory=list)
    progress_flags: dict[str, bool] = Field(default_factory=dict)
    common_trap: str = ""
    last_action_result: str = ""
    reward_breakdown: dict[str, float | str | int] = Field(default_factory=dict)
    cumulative_reward: float = 0.0
    incident_resolved: bool = False
    difficulty: str
    scenario_id: str
    scenario_name: str
    step_budget: int
    observation_mode: str = "multi_agent"
    specialist_mode: str = "deterministic"
    stage_goal: str = ""
    required_fields_by_action: dict[str, list[str]] = Field(default_factory=dict)
    valid_action_example: dict[str, Any] = Field(default_factory=dict)
    loop_warning: str = ""
    done: bool = False


class POMIRState(State):
    model_config = ConfigDict(extra="forbid")

    prompt_text: str
    workflow_stage: str
    reports: list[SpecialistReport] = Field(default_factory=list)
    specialist_executions: list[SpecialistRoundExecution] = Field(default_factory=list)
    commander_execution: CommanderExecution | None = None
    allowed_actions: list[str] = Field(default_factory=list)
    reward_breakdown: dict[str, float | str | int] = Field(default_factory=dict)
    cumulative_reward: float = 0.0
    progress_flags: dict[str, bool] = Field(default_factory=dict)
    master_state: dict[str, Any] = Field(default_factory=dict)
    round_history: list[dict[str, Any]] = Field(default_factory=list)
    observation_mode: str = "multi_agent"
    specialist_mode: str = "deterministic"
    stage_goal: str = ""
    required_fields_by_action: dict[str, list[str]] = Field(default_factory=dict)
    valid_action_example: dict[str, Any] = Field(default_factory=dict)
    loop_warning: str = ""
    done: bool = False


class POMIREnv(Environment[CommanderAction, POMIRObservation, POMIRState]):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(
        self,
        mode: str | None = None,
        commander: Commander | None = None,
        *,
        observation_mode: str = "multi_agent",
        specialist_mode: str | None = None,
    ) -> None:
        normalized_mode = mode or current_mode()
        self.mode = normalized_mode
        self.specialist_mode = specialist_mode or LEGACY_SPECIALIST_MODE.get(normalized_mode, normalized_mode)
        self.observation_mode = observation_mode
        self.master_env = MasterSREEnv()
        self.infra_env = InfraEnv(self.master_env)
        self.log_env = LogEnv(self.master_env)
        self.sec_env = SecEnv(self.master_env)
        self.commander = commander or Commander()
        self.single_agent_commander = SingleAgentCommander()
        self.reward_calculator = CompositeRewardCalculator()
        self._current_reports: list[SpecialistReport] = []
        self._current_specialist_executions: list[SpecialistRoundExecution] = []
        self._last_commander_execution: CommanderExecution | None = None
        self._prior_actions: list[CommanderAction] = []
        self._last_reward = CompositeRewardResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "", "easy", 0)
        self._cumulative_reward = 0.0
        self._done = False
        self._refresh_reports()

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="spectra_main",
            description="Meta-orchestrator for the SPECTRA multi-agent benchmark",
            version="0.2.0",
            author="Madhav Gupta",
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> POMIRObservation:
        observation_mode = kwargs.pop("observation_mode", None)
        specialist_mode = kwargs.pop("specialist_mode", None)
        if observation_mode is not None:
            self.observation_mode = observation_mode
        if specialist_mode is not None:
            self.specialist_mode = specialist_mode
        self.master_env.reset(seed=seed, episode_id=episode_id, **kwargs)
        self.commander.reset()
        self.single_agent_commander.reset()
        self._prior_actions = []
        self._done = False
        self._last_commander_execution = None
        self._last_reward = CompositeRewardResult(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            self.master_env.state.episode_id,
            self.master_env.state.difficulty,
            0,
        )
        self._cumulative_reward = 0.0
        self.master_env.state.round_history.clear()
        self._refresh_reports()
        return self._build_observation()

    def step(
        self,
        action: CommanderAction | dict[str, Any],
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> POMIRObservation:
        del timeout_s, kwargs
        if isinstance(action, dict):
            action = CommanderAction(**action)
        reports_used = list(self._current_reports)
        specialist_executions_used = list(self._current_specialist_executions)
        state_before = self.master_env.state
        state_after, info = self.master_env.step(action)
        self._last_reward = self.reward_calculator.compute(
            action=action,
            reports=reports_used,
            state=state_after,
            prior_actions=self._prior_actions,
            done=bool(info["done"]),
            first_action=self._prior_actions[0] if self._prior_actions else action,
            observation_mode=self.observation_mode,
        )
        self._cumulative_reward = round(self._cumulative_reward + self._last_reward.total, 3)
        self._prior_actions.append(action)
        self._done = bool(info["done"])
        round_trace = MultiAgentRound(
            round_index=state_after.tick,
            workflow_stage=state_before.workflow_stage,
            specialist_executions=specialist_executions_used,
            commander_execution=self._last_commander_execution,
            environment_feedback=info["message"],
            reward_breakdown=self._last_reward.as_dict(),
        )
        self.master_env.append_round_trace(round_trace.model_dump())
        self._refresh_reports()
        return self._build_observation(last_action_result=info["message"])

    @property
    def state(self) -> POMIRState:
        master_state = self.master_env.state
        allowed_actions = self.allowed_action_strings
        if self.observation_mode == "single_agent":
            prompt_text = build_single_agent_observation(
                master_state,
                allowed_actions=allowed_actions,
            )
        else:
            prompt_text = build_commander_observation(
                self._current_reports,
                master_state,
                allowed_actions=allowed_actions,
            )
        return POMIRState(
            episode_id=master_state.episode_id,
            step_count=master_state.tick,
            prompt_text=prompt_text,
            workflow_stage=master_state.workflow_stage,
            reports=self._current_reports,
            specialist_executions=self._current_specialist_executions,
            commander_execution=self._last_commander_execution,
            allowed_actions=allowed_actions,
            reward_breakdown=self._last_reward.as_dict(),
            cumulative_reward=self._cumulative_reward,
            progress_flags=master_state.progress_flags,
            master_state={
                "scenario_id": master_state.scenario_id,
                "scenario_name": master_state.objective.name,
                "difficulty": master_state.difficulty,
                "workflow_stage": master_state.workflow_stage,
                "step_budget": master_state.step_budget,
                "total_step_budget": master_state.total_step_budget,
                "last_action_result": master_state.last_action_result,
                "progress_flags": dict(master_state.progress_flags),
                "round_history_length": len(master_state.round_history),
            },
            round_history=list(master_state.round_history),
            observation_mode=self.observation_mode,
            specialist_mode=self.specialist_mode,
            stage_goal=STAGE_GOALS.get(master_state.workflow_stage, ""),
            required_fields_by_action=_required_fields_by_action(allowed_actions),
            valid_action_example=self._valid_action_example(),
            loop_warning=self._loop_warning(),
            done=self._done,
        )

    @property
    def allowed_action_strings(self) -> list[str]:
        actions = list(ALLOWED_ACTIONS_BY_STAGE.get(self.master_env.state.workflow_stage, []))
        if self.observation_mode == "single_agent":
            actions = [action for action in actions if not action.startswith("request_followup")]
        return actions

    @property
    def allowed_action_names(self) -> list[str]:
        return [action.split("(", 1)[0] for action in self.allowed_action_strings]

    def decide_next_action(self) -> CommanderAction:
        decision = self.plan_next_action()
        return decision.action

    def plan_next_action(self) -> CommanderDecision:
        if self.observation_mode == "single_agent":
            action = self.single_agent_commander.act_from_state(self.master_env.state)
            execution = CommanderExecution(
                mode="heuristic",
                raw_response=json.dumps(action.model_dump(), ensure_ascii=True),
                latency_ms=0.0,
                trust_weights={},
                action=action.model_dump(),
            )
            self._last_commander_execution = execution
            return CommanderDecision(action=action, execution=execution)
        decision = self.commander.decide(
            self._current_reports,
            self.master_env.state,
            allowed_actions=self.allowed_action_strings,
        )
        self._last_commander_execution = decision.execution
        return decision

    def baseline_plan(self, scenario_id: str) -> list[dict[str, Any]]:
        return baseline_plan_for_scenario(scenario_id)

    def close(self) -> None:
        return None

    def run_episode(
        self,
        difficulty: str = "easy",
        seed: int | None = None,
        scenario_id: str | None = None,
    ) -> list[dict[str, Any]]:
        trajectory: list[dict[str, Any]] = []
        observation = self.reset(difficulty=difficulty, scenario_id=scenario_id, seed=seed)
        trajectory.append({"observation": observation.model_dump()})
        while not observation.done:
            decision = self.plan_next_action()
            observation = self.step(decision.action)
            trajectory.append(
                {
                    "action": decision.action.model_dump(),
                    "observation": observation.model_dump(),
                }
            )
        return trajectory

    async def _gather_specialists(self):
        infra_state = self.infra_env.state
        log_state = self.log_env.state
        sec_state = self.sec_env.state
        infra_specialist = InfraSpecialist()
        log_specialist = LogSpecialist()
        sec_specialist = SecSpecialist()
        return await asyncio.gather(
            infra_specialist.generate_execution_async(infra_state, specialist_mode=self.specialist_mode),
            log_specialist.generate_execution_async(log_state, specialist_mode=self.specialist_mode),
            sec_specialist.generate_execution_async(sec_state, specialist_mode=self.specialist_mode),
        )

    def _refresh_reports(self) -> None:
        if self.observation_mode == "single_agent":
            self._current_reports = []
            self._current_specialist_executions = []
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            results = asyncio.run(self._gather_specialists())
        else:
            captured: list[Any] = []
            error: list[BaseException] = []

            def _runner() -> None:
                try:
                    captured.append(asyncio.run(self._gather_specialists()))
                except BaseException as exc:  # pragma: no cover - defensive bridge for server startup
                    error.append(exc)

            thread = threading.Thread(target=_runner, daemon=True)
            thread.start()
            thread.join()
            if error:
                raise error[0]
            results = captured[0]
        self._current_reports = [result.report for result in results]
        self._current_specialist_executions = [result.execution for result in results]
        pending = self.master_env.state.pending_followup_agent
        if pending:
            self.master_env.mark_followup_seen(pending)
            self.master_env.clear_pending_followup()

    def _build_observation(self, last_action_result: str | None = None) -> POMIRObservation:
        master_state = self.master_env.state
        allowed_actions = self.allowed_action_strings
        if self.observation_mode == "single_agent":
            prompt_text = build_single_agent_observation(
                master_state,
                allowed_actions=allowed_actions,
            )
        else:
            prompt_text = build_commander_observation(
                self._current_reports,
                master_state,
                allowed_actions=allowed_actions,
            )
        return POMIRObservation(
            metadata={
                "scenario_id": master_state.scenario_id,
                "scenario_name": master_state.objective.name,
                "workflow_stage": master_state.workflow_stage,
            },
            reward=self._last_reward.total,
            prompt_text=prompt_text,
            workflow_stage=master_state.workflow_stage,
            reports=self._current_reports,
            specialist_executions=self._current_specialist_executions,
            commander_execution=self._last_commander_execution,
            allowed_actions=allowed_actions,
            progress_flags=master_state.progress_flags,
            common_trap=master_state.common_trap,
            last_action_result=last_action_result or master_state.last_action_result,
            reward_breakdown=self._last_reward.as_dict(),
            cumulative_reward=self._cumulative_reward,
            incident_resolved=master_state.incident_resolved,
            difficulty=master_state.difficulty,
            scenario_id=master_state.scenario_id,
            scenario_name=master_state.objective.name,
            step_budget=master_state.step_budget,
            observation_mode=self.observation_mode,
            specialist_mode=self.specialist_mode,
            stage_goal=STAGE_GOALS.get(master_state.workflow_stage, ""),
            required_fields_by_action=_required_fields_by_action(allowed_actions),
            valid_action_example=self._valid_action_example(),
            loop_warning=self._loop_warning(),
            done=self._done,
        )

    def _valid_action_example(self) -> dict[str, Any]:
        allowed_names = {action.split("(", 1)[0] for action in self.allowed_action_strings}
        for candidate in baseline_plan_for_scenario(self.master_env.state.scenario_id):
            if candidate["action_type"] in allowed_names:
                return candidate
        if "request_followup" in allowed_names and self.master_env.state.best_followup_agent is not None:
            return {
                "action_type": "request_followup",
                "target_agent": self.master_env.state.best_followup_agent,
            }
        if "submit_resolution" in allowed_names:
            return {
                "action_type": "submit_resolution",
                "resolution_summary": f"Recovered {self.master_env.state.root_cause_service} after containment.",
            }
        if "investigate_service" in allowed_names:
            return {
                "action_type": "investigate_service",
                "target_service": self.master_env.state.root_cause_service,
            }
        if self.master_env.state.root_cause_service and allowed_names:
            action_type = next(iter(allowed_names))
            return {
                "action_type": action_type,
                "target_service": self.master_env.state.root_cause_service,
            }
        return {}

    def _loop_warning(self) -> str:
        if len(self._prior_actions) < 2:
            return ""
        if self._last_reward.total > 0.0:
            return ""
        if self._prior_actions[-1].rendered != self._prior_actions[-2].rendered:
            return ""
        return (
            f"Recent no-progress repeat detected: {self._prior_actions[-1].rendered}. "
            "Change target or action family."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    env = POMIREnv()
    trajectory = env.run_episode(difficulty=args.difficulty, seed=args.seed)
    print(json.dumps(trajectory, indent=2))


if __name__ == "__main__":
    main()
