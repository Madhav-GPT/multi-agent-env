"""OpenEnv-compatible log partition."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict

from environments.shared.master_env import MasterSREEnv
from environments.shared.openenv_compat import Action, Environment, EnvironmentMetadata, Observation

from .state import LogState


class LogObserveAction(Action):
    model_config = ConfigDict(extra="forbid")
    action_type: Literal["observe"] = "observe"


class LogObservation(Observation):
    model_config = ConfigDict(extra="forbid")

    channel: Literal["log"] = "log"
    summary: str
    tick: int
    step_budget: int
    observation: LogState
    done: bool = False


class LogEnv(Environment[LogObserveAction, LogObservation, LogState]):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, master_env: MasterSREEnv | None = None) -> None:
        self._owned_master = master_env is None
        self._master = master_env or MasterSREEnv()

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="log_env",
            description="Application-log-only view of the SPECTRA environment",
            version="0.1.0",
            author="Madhav Gupta",
        )

    def reset(self, seed: int | None = None, **kwargs: Any) -> LogObservation:
        if self._owned_master:
            self._master.reset(seed=seed, **kwargs)
        return self._observe()

    def step(
        self,
        action: LogObserveAction | dict[str, Any],
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> LogObservation:
        del action, timeout_s, kwargs
        return self._observe()

    @property
    def state(self) -> LogState:
        return self._master.get_partial_observation("log")

    def close(self) -> None:
        return None

    def _observe(self) -> LogObservation:
        state = self.state
        dominant_error = max(state.error_types, key=state.error_types.get)
        return LogObservation(
            summary=f"Dominant error pattern: {dominant_error}.",
            tick=state.tick,
            step_budget=state.step_budget,
            observation=state,
        )
