from __future__ import annotations

import pytest

from environments.infra_env.extractor import InfraExtractor
from environments.log_env.extractor import LogExtractor
from environments.sec_env.extractor import SecExtractor
from environments.shared.master_env import MasterSREEnv


@pytest.mark.parametrize(
    ("scenario_id", "expected"),
    [
        (
            "broken_auth_cascade",
            {
                "infra": "cache",
                "log": "auth_service",
                "security": "auth_service",
            },
        ),
        (
            "worker_supply_chain_compromise",
            {
                "infra": "database",
                "log": "worker",
                "security": "worker",
            },
        ),
        (
            "cache_poisoning_campaign",
            {
                "infra": "cache",
                "log": "api-gateway",
                "security": "cache",
            },
        ),
    ],
)
def test_extractors_respect_signal_map(
    scenario_id: str,
    expected: dict[str, str],
) -> None:
    env = MasterSREEnv()
    env.reset(scenario_id=scenario_id, seed=42)
    infra = InfraExtractor().extract(env.get_partial_observation("infra"))
    logs = LogExtractor().extract(env.get_partial_observation("log"))
    sec = SecExtractor().extract(env.get_partial_observation("security"))

    assert infra.top_hypothesis_service == expected["infra"]
    assert logs.top_hypothesis_service == expected["log"]
    assert sec.top_hypothesis_service == expected["security"]
