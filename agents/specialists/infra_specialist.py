"""Demo-mode infrastructure specialist."""

from __future__ import annotations

from environments.infra_env.extractor import InfraExtractor
from environments.infra_env.state import InfraState

from .base_specialist import BaseSpecialist


class InfraSpecialist(BaseSpecialist):
    model_id = "Qwen/Qwen3-4B-Thinking-2507"
    provider = "nscale"
    model_env_var = "INFRA_SPECIALIST_MODEL"
    provider_env_var = "INFRA_SPECIALIST_PROVIDER"
    prompt_file = "infra_system_prompt.txt"

    def __init__(self) -> None:
        super().__init__()
        self._extractor = InfraExtractor()

    def fallback_report(self, state: InfraState):
        return self._extractor.extract(state)
