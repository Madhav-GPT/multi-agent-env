"""Demo-mode security specialist."""

from __future__ import annotations

from environments.sec_env.extractor import SecExtractor
from environments.sec_env.state import SecState

from .base_specialist import BaseSpecialist


class SecSpecialist(BaseSpecialist):
    model_id = "Qwen/Qwen2.5-72B-Instruct"
    provider = "novita"
    prompt_file = "sec_system_prompt.txt"

    def __init__(self) -> None:
        super().__init__()
        self._extractor = SecExtractor()

    def fallback_report(self, state: SecState):
        return self._extractor.extract(state)

