"""Demo-mode log specialist."""

from __future__ import annotations

from environments.log_env.extractor import LogExtractor
from environments.log_env.state import LogState

from .base_specialist import BaseSpecialist


class LogSpecialist(BaseSpecialist):
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    model_env_var = "LOG_SPECIALIST_MODEL"
    provider_env_var = "LOG_SPECIALIST_PROVIDER"
    prompt_file = "log_system_prompt.txt"

    def __init__(self) -> None:
        super().__init__()
        self._extractor = LogExtractor()

    def fallback_report(self, state: LogState):
        return self._extractor.extract(state)
