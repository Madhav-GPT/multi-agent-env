"""FastAPI server for SecEnv."""

from __future__ import annotations

import argparse

from environments.shared.openenv_compat import create_app

from .env import SecEnv, SecObservation, SecObserveAction

app = create_app(
    lambda: SecEnv(),
    SecObserveAction,
    SecObservation,
    env_name="sec_env",
    max_concurrent_envs=1,
)


def serve(host: str = "0.0.0.0", port: int = 8003) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

