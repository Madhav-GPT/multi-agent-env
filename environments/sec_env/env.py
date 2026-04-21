"""OpenEnv-compatible security partition."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict

from environments.shared.master_env import MasterSREEnv
from environments.shared.openenv_compat import Action, Environment, EnvironmentMetadata, Observation

from .state import SecState


class SecObserveAction(Action):
    model_config = ConfigDict(extra="forbid")
    action_type: Literal["observe"] = "observe"


class SecObservation(Observation):
    model_config = ConfigDict(extra="forbid")

    channel: Literal["security"] = "security"
    summary: str
    tick: int
    step_budget: int
    observation: SecState
    done: bool = False


class SecEnv(Environment[SecObserveAction, SecObservation, SecState]):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, master_env: MasterSREEnv | None = None) -> None:
        self._owned_master = master_env is None
        self._master = master_env or MasterSREEnv()

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="sec_env",
            description="Security-only view of the SPECTRA environment",
            version="0.1.0",
            author="Madhav Gupta",
        )

    def reset(self, seed: int | None = None, **kwargs: Any) -> SecObservation:
        if self._owned_master:
            self._master.reset(seed=seed, **kwargs)
        return self._observe()

    def step(
        self,
        action: SecObserveAction | dict[str, Any],
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> SecObservation:
        del action, timeout_s, kwargs
        return self._observe()

    @property
    def state(self) -> SecState:
        return self._master.get_partial_observation("security")

    def close(self) -> None:
        return None

    def _observe(self) -> SecObservation:
        state = self.state
        headline = state.alert_strings[0] if state.alert_strings else "No active security alert."
        return SecObservation(
            summary=headline,
            tick=state.tick,
            step_budget=state.step_budget,
            observation=state,
        )
