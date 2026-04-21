"""OpenEnv-compatible infra partition."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict

from environments.shared.master_env import MasterSREEnv
from environments.shared.openenv_compat import Action, Environment, EnvironmentMetadata, Observation

from .state import InfraState


class InfraObserveAction(Action):
    model_config = ConfigDict(extra="forbid")
    action_type: Literal["observe"] = "observe"


class InfraObservation(Observation):
    model_config = ConfigDict(extra="forbid")

    channel: Literal["infra"] = "infra"
    summary: str
    tick: int
    step_budget: int
    observation: InfraState
    done: bool = False


class InfraEnv(Environment[InfraObserveAction, InfraObservation, InfraState]):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, master_env: MasterSREEnv | None = None) -> None:
        self._owned_master = master_env is None
        self._master = master_env or MasterSREEnv()

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="infra_env",
            description="Infrastructure-only view of the SPECTRA environment",
            version="0.1.0",
            author="Madhav Gupta",
        )

    def reset(self, seed: int | None = None, **kwargs: Any) -> InfraObservation:
        if self._owned_master:
            self._master.reset(seed=seed, **kwargs)
        return self._observe()

    def step(
        self,
        action: InfraObserveAction | dict[str, Any],
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> InfraObservation:
        del action, timeout_s, kwargs
        return self._observe()

    @property
    def state(self) -> InfraState:
        return self._master.get_partial_observation("infra")

    def close(self) -> None:
        return None

    def _observe(self) -> InfraObservation:
        state = self.state
        worst_service = max(state.cpu_pct, key=state.cpu_pct.get)
        return InfraObservation(
            summary=f"{worst_service} has the hottest metrics profile.",
            tick=state.tick,
            step_budget=state.step_budget,
            observation=state,
        )
