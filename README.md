# SPECTRA

SPECTRA is a terminal-first OpenEnv benchmark for multi-agent incident response under enforced partial observability. It rebuilds the earlier single-agent `openenv` incident project into a proper coordination benchmark: three specialists work in parallel on disjoint state partitions, and a commander acts only on their reports.

This project is intentionally not a browser dashboard. The main experience is the round-by-round terminal trace from `run_demo.py` and `inference.py`, where you can see the specialists, commander, workflow stage, and reward stack evolve together.

## What This Project Does

- Keeps one hidden `MasterSREState` with metrics, logs, security telemetry, causal graph, and staged resolution state.
- Exposes three strict sub-environments:
  - `InfraEnv`: metrics only
  - `LogEnv`: logs only
  - `SecEnv`: security telemetry only
- Runs three specialists concurrently each round and stores their raw execution artifacts.
- Forces the commander to operate on serialized `SpecialistReport` objects, not raw state.
- Scores each step with the five-part SPECTRA reward:
  - `R1` resolution
  - `R2` first-step root-cause targeting
  - `R3` coordination
  - `R4` efficiency
  - `R5` specialist trust calibration

## Architecture

```text
MasterSREEnv (hidden full state)
  -> InfraEnv -> Infra specialist
  -> LogEnv   -> Log specialist
  -> SecEnv   -> Security specialist
  -> POMIREnv orchestrator -> Commander -> DeterministicJudge -> reward stack
```

Key implementation points:

- Partitioning is enforced by Pydantic schemas, not prompt text.
- Specialists can run in `training` mode with deterministic extractors or `demo` mode with Hugging Face-backed inference plus fallback.
- The commander loop is stage-aware:
  - `triage`
  - `containment`
  - `remediation`
  - `recovery`
  - `retrospective`
  - `done`
- All parallel round artifacts are saved in `round_history` and can be exported as raw traces.

## Scenario Catalog

SPECTRA now ships with five benchmark incidents instead of only the old easy/medium/hard trio:

| Scenario | Difficulty | Root service | Core twist |
| --- | --- | --- | --- |
| `database_sqli_outage` | easy | `database` | clean SQLi causal chain |
| `api_gateway_xss` | medium | `api-gateway` | gateway recovery uses `scale_service` |
| `broken_auth_cascade` | hard | `auth_service` | infra is misled by cache saturation |
| `worker_supply_chain_compromise` | hard | `worker` | poisoned release destabilizes downstream systems |
| `cache_poisoning_campaign` | medium | `cache` | logs mislead toward `api-gateway` |

## Project Layout

```text
Madhav_task/
├── agents/
├── environments/
├── rewards/
├── runtime/
├── training/
├── eval/
├── server/
├── tests/
├── inference.py
├── run_demo.py
└── openenv.yaml
```

Highlights:

- `environments/shared/scenarios.py`: scenario catalog and deterministic state builder
- `environments/shared/judge.py`: staged judge logic
- `environments/pomir_env/env.py`: orchestrator and parallel specialist gathering
- `runtime/terminal.py`: terminal tables and trace export
- `environments/pomir_env/server.py`: OpenEnv app plus `/tasks`, `/baseline`, `/status`, `/plan`

## Quick Start

Two supported environment options:

- Local: create `Madhav_task/.venv`
- Shared fallback: reuse `/Users/madhav_189/Documents/Scalar_hackathon/openenv/.venv`

Recommended first-time setup:

```bash
cd /Users/madhav_189/Documents/Scalar_hackathon/Madhav_task
make setup-venv
make doctor
make test
make demo
```

If you do not create a local `.venv`, the `Makefile` falls back automatically to the older shared environment at `/Users/madhav_189/Documents/Scalar_hackathon/openenv/.venv`.

## Main Entry Points

Run a local terminal demo:

```bash
make demo SCENARIO=cache_poisoning_campaign
```

Run a local benchmark trace:

```bash
make inference SCENARIO=worker_supply_chain_compromise
```

Start the OpenEnv server:

```bash
make server
```

Run against the live server:

```bash
make remote REMOTE_SCENARIO=database_sqli_outage
```

Smoke the comparison harness:

```bash
make train-smoke
```

## Server Routes

- `POST /reset`: reset the persistent runtime environment
- `POST /step`: apply a commander action
- `POST /plan`: ask the built-in commander for the next action
- `GET /state`: current orchestrator state
- `GET /tasks`: scenario catalog
- `GET /baseline`: baseline plans
- `GET /status`: runtime progress summary
- `GET /health`: environment metadata and stage list

## Output Artifacts

- `outputs/raw_traces/`: per-episode execution traces
- `outputs/grpo_smoke/` and `outputs/grpo_runs/`: comparison harness outputs
- `outputs/remote_smoke/`: server-backed inference traces

## Validation Status

Validated locally in the rebuilt project:

- `18` pytest checks passing
- local `inference.py` smoke test
- local `run_demo.py` smoke test
- `training/grpo_train.py` smoke test
- live server-backed remote inference smoke test
