"""Runtime mode selection for SPECTRA."""

from __future__ import annotations

import os
from typing import Literal

SpectraMode = Literal["training", "demo"]


def current_mode() -> SpectraMode:
    value = os.getenv("SPECTRA_MODE", "training").strip().lower()
    return "demo" if value == "demo" else "training"

