"""Small runtime helpers for loading local `.env` files without extra dependencies."""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def load_runtime_env() -> None:
    """Load repo-local `.env` once if it exists.

    The file is optional and ignored when variables are already set in the shell.
    """

    global _LOADED
    if _LOADED:
        return

    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value

    _LOADED = True
