"""Compatibility layer for local development without OpenEnv installed."""

from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

try:
    from openenv.core import Action, Observation, State
    from openenv.core.env_server import Environment, create_app
    from openenv.core.env_server.types import EnvironmentMetadata
except Exception:  # pragma: no cover - used only when openenv-core is unavailable.
    ActionT = TypeVar("ActionT", bound="Action")
    ObservationT = TypeVar("ObservationT", bound="Observation")
    StateT = TypeVar("StateT", bound="State")

    class Action(BaseModel):
        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

        metadata: dict[str, Any] = Field(default_factory=dict)

    class Observation(BaseModel):
        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

        done: bool = False
        reward: bool | int | float | None = None
        metadata: dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        model_config = ConfigDict(
            extra="allow",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

        episode_id: str | None = None
        step_count: int = 0

    class EnvironmentMetadata(BaseModel):
        model_config = ConfigDict(extra="forbid")

        name: str
        description: str
        version: str
        author: str

    class Environment(Generic[ActionT, ObservationT, StateT]):
        SUPPORTS_CONCURRENT_SESSIONS = False

        def get_metadata(self) -> EnvironmentMetadata:
            raise NotImplementedError

        def reset(self, seed: int | None = None, **kwargs: Any) -> ObservationT:
            raise NotImplementedError

        def step(
            self,
            action: ActionT | dict[str, Any],
            timeout_s: float | None = None,
            **kwargs: Any,
        ) -> ObservationT:
            raise NotImplementedError

        @property
        def state(self) -> StateT:
            raise NotImplementedError

        def close(self) -> None:
            return None

    def create_app(
        env_factory: Callable[[], Environment[Any, Any, Any]],
        action_model: type[BaseModel],
        observation_model: type[BaseModel],
        *,
        env_name: str,
        max_concurrent_envs: int = 1,
    ) -> FastAPI:
        del max_concurrent_envs
        app = FastAPI(title=env_name)

        def _serialize_observation(observation: Any) -> dict[str, Any]:
            if hasattr(observation, "model_dump"):
                payload = observation.model_dump()
            else:
                payload = dict(observation)
            reward = payload.pop("reward", None)
            done = bool(payload.get("done", False))
            return {
                "observation": payload,
                "reward": reward,
                "done": done,
            }

        @app.get("/health")
        def health() -> dict[str, str]:
            return {"status": "ok", "environment": env_name}

        @app.get("/metadata")
        def metadata() -> dict[str, Any]:
            env = env_factory()
            try:
                return env.get_metadata().model_dump()
            finally:
                env.close()

        @app.get("/schema")
        def schema() -> dict[str, Any]:
            return {
                "action": action_model.model_json_schema(),
                "observation": observation_model.model_json_schema(),
                "state": State.model_json_schema(),
            }

        @app.post("/reset")
        def reset(payload: dict[str, Any] | None = None) -> dict[str, Any]:
            env = env_factory()
            try:
                observation = env.reset(**(payload or {}))
                return _serialize_observation(observation)
            finally:
                env.close()

        @app.post("/step")
        def step(payload: dict[str, Any]) -> dict[str, Any]:
            env = env_factory()
            try:
                action = payload.get("action", payload)
                kwargs = {key: value for key, value in payload.items() if key != "action"}
                observation = env.step(action, **kwargs)
                return _serialize_observation(observation)
            finally:
                env.close()

        @app.get("/state")
        def get_state() -> dict[str, Any]:
            env = env_factory()
            try:
                return env.state.model_dump()
            finally:
                env.close()

        return app
