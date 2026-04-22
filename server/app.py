"""Top-level OpenEnv entrypoint for the SPECTRA orchestrator."""

from environments.pomir_env.server import app, serve
from environments.pomir_env.server import main as _main

__all__ = ["app", "main", "serve"]


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
