# SPECTRA Execution Guide

This guide is for a first-time user of `/Users/madhav_189/Documents/Scalar_hackathon/Madhav_task`.

The biggest confusion was the Python environment, so this file starts there.

## 1. Python Environment: Two Supported Ways

There are **two valid ways** to run this project.

### Option A: Use a local `.venv` inside `Madhav_task`

This is the cleanest option for a first-time user.

```bash
cd /Users/madhav_189/Documents/Scalar_hackathon/Madhav_task
make setup-venv
```

This creates:

```text
/Users/madhav_189/Documents/Scalar_hackathon/Madhav_task/.venv
```

After that, all `make` commands automatically use this local `.venv`.

### Option B: Reuse the older shared environment from `openenv`

This project also supports the environment you already created earlier:

```text
/Users/madhav_189/Documents/Scalar_hackathon/openenv/.venv
```

If there is **no local `.venv`** in `Madhav_task`, the `Makefile` automatically falls back to this shared `openenv/.venv`.

### How the project chooses

Run:

```bash
make doctor
```

It will tell you:

- whether it is using `local`
- or `shared-openenv`
- and the exact Python path

## 2. What Kind Of Session Are You Running?

There are **three main ways** to run SPECTRA. This is the other important distinction.

| Mode | Command | What it is for |
| --- | --- | --- |
| Terminal demo | `make demo` | Best for seeing multi-agent behavior live |
| Local inference | `make inference` | Best for running one episode and saving a trace |
| Remote inference | `make remote` | Best for testing through the actual server API |

### 1. `make demo`

This is the best first run.

It shows:

- the three specialists in parallel
- their hypotheses
- their recommended actions
- commander trust weights
- reward breakdown
- workflow stage progression

### 2. `make inference`

This runs one episode without needing the server.

It prints `[START]`, `[ROUND]`, `[STEP]`, `[END]` logs and saves a JSON trace.

### 3. `make remote`

This is different from `make inference`.

`make remote` talks to a live FastAPI/OpenEnv server using:

- `/reset`
- `/plan`
- `/step`

So for `make remote`, you must first start the server in another terminal.

## 3. First-Time User: Best Order

If this is your first time, run exactly this:

```bash
cd /Users/madhav_189/Documents/Scalar_hackathon/Madhav_task
make setup-venv
make doctor
make test
make demo
make inference
```

If you do **not** want a local `.venv`, then skip `make setup-venv` and just do:

```bash
cd /Users/madhav_189/Documents/Scalar_hackathon/Madhav_task
make doctor
make test
make demo
```

## 4. Most Important Commands

### Check environment and scenarios

```bash
make doctor
make scenarios
```

### Run tests

```bash
make test
```

### Watch a multi-agent run in the terminal

```bash
make demo
```

Run a specific scenario:

```bash
make demo SCENARIO=database_sqli_outage
make demo SCENARIO=api_gateway_xss
make demo SCENARIO=cache_poisoning_campaign
make demo SCENARIO=broken_auth_cascade
make demo SCENARIO=worker_supply_chain_compromise
```

Short aliases:

```bash
make demo-easy
make demo-medium
make demo-hard
make demo-cache
```

### Save a local trace

```bash
make inference
```

Pretty local inference:

```bash
make inference-pretty
```

Run all five scenarios:

```bash
make inference-all
```

## 5. Server-Based Session

### Terminal 1: start the server

```bash
make server
```

or reload mode:

```bash
make dev
```

### Terminal 2: run through the server

```bash
make remote
```

Pretty remote run:

```bash
make remote-pretty
```

Custom server URL:

```bash
make remote BASE_URL=http://127.0.0.1:8010 REMOTE_SCENARIO=database_sqli_outage
```

## 6. Benchmark / Training Commands

Quick benchmark smoke:

```bash
make train-smoke
```

Run one condition manually:

```bash
make train CONDITION=A EPISODES=10
make train CONDITION=B EPISODES=10
make train CONDITION=C EPISODES=10
```

Generate A/B/C comparison outputs:

```bash
make compare
```

Trust calibration probe:

```bash
make trust-probe
```

## 7. Variables You Can Override

You can change behavior directly from the command line.

Examples:

```bash
make demo SCENARIO=cache_poisoning_campaign SEED=99
make inference SCENARIO=worker_supply_chain_compromise OUTPUT_DIR=outputs/my_runs
make remote REMOTE_SCENARIO=database_sqli_outage BASE_URL=http://127.0.0.1:8010
make train CONDITION=C EPISODES=25
```

Main variables:

- `SCENARIO`: used by local `demo` and `inference`
- `REMOTE_SCENARIO`: used by remote/server inference
- `SEED`: episode seed
- `PORT`: server port
- `BASE_URL`: remote server URL
- `OUTPUT_DIR`: local trace directory
- `CONDITION`: benchmark condition `A`, `B`, or `C`
- `EPISODES`: number of episodes for `make train`

## 8. Scenario IDs

| Scenario ID | Difficulty | Root cause |
| --- | --- | --- |
| `database_sqli_outage` | easy | `database` |
| `api_gateway_xss` | medium | `api-gateway` |
| `cache_poisoning_campaign` | medium | `cache` |
| `broken_auth_cascade` | hard | `auth_service` |
| `worker_supply_chain_compromise` | hard | `worker` |

## 9. Output Directories

- `outputs/raw_traces/`: local inference traces
- `outputs/remote_smoke/`: remote/server traces
- `outputs/grpo_smoke/`: quick benchmark smoke outputs
- `outputs/grpo_runs/`: benchmark condition outputs
- `outputs/eval/`: charts and summaries

## 10. Cleanup

```bash
make clean
```

## 11. One-Line Clarification

If you want the shortest explanation:

- `make demo` = watch the agents live in one terminal
- `make inference` = run locally and save a trace
- `make server` + `make remote` = test the real API/server path
- `make setup-venv` = create a local `.venv` if you do not want to rely on `openenv/.venv`
