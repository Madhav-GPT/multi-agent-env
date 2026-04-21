from __future__ import annotations

import pytest
from pydantic import ValidationError

from environments.infra_env.state import InfraState
from environments.log_env.state import LogState
from environments.sec_env.state import SecState
from environments.shared.master_env import MasterSREEnv


def test_full_state_cannot_instantiate_infra_partition() -> None:
    env = MasterSREEnv()
    payload = env.state.model_dump()
    with pytest.raises(ValidationError):
        InfraState(**payload)


def test_full_state_cannot_instantiate_log_partition() -> None:
    env = MasterSREEnv()
    payload = env.state.model_dump()
    with pytest.raises(ValidationError):
        LogState(**payload)


def test_full_state_cannot_instantiate_sec_partition() -> None:
    env = MasterSREEnv()
    payload = env.state.model_dump()
    with pytest.raises(ValidationError):
        SecState(**payload)

