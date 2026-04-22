# SPECTRA Execution Guide

This is the concrete runbook for the current repo. It matches the real `Makefile`, `demo.py`, `inference.py`, and test status in `/Users/madhav_189/Documents/Scalar_hackathon/Madhav_task`.

Project root:

```bash
cd /Users/madhav_189/Documents/Scalar_hackathon/Madhav_task
```

## 1. Python Environment

The repo supports two Python paths.

### Option A: create a local `.venv`

```bash
make setup-venv
```

This creates:

```text
/Users/madhav_189/Documents/Scalar_hackathon/Madhav_task/.venv
```

### Option B: reuse the older shared environment

If there is no local `.venv`, the `Makefile` falls back to:

```text
/Users/madhav_189/Documents/Scalar_hackathon/openenv/.venv/bin/python
```

### Check what the repo is using

```bash
make doctor
make which-python
```

## 2. Optional `.env`

The repo auto-loads a local `.env` file if it exists.

Create it from the template:

```bash
cp .env.example .env
```

Most useful keys:

- `HF_TOKEN`
- `INFRA_SPECIALIST_MODEL`
- `INFRA_SPECIALIST_PROVIDER`
- `LOG_SPECIALIST_MODEL`
- `LOG_SPECIALIST_PROVIDER`
- `SEC_SPECIALIST_MODEL`
- `SEC_SPECIALIST_PROVIDER`
- `COMMANDER_PROVIDER`
- `COMMANDER_MODEL`
- `COMMANDER_HF_PROVIDER`
- `API_BASE_URL`
- `OPENAI_API_KEY`

Important notes:

- `.env` is ignored by git
- hosted specialist runs need `HF_TOKEN`
- local commander runs use an OpenAI-compatible endpoint, usually Ollama at `http://127.0.0.1:11434/v1`
- the repo defaults to `SPECIALIST_MODE=hybrid` for hosted multi-agent runs and `LOCAL_MULTI_SPECIALIST_MODE=deterministic` for local smoke runs

## 3. First 5 Minutes

For a first-time user:

```bash
make doctor
./.venv/bin/python -m pytest tests -q
make scenarios
make demo SCENARIO=broken_auth_cascade
```

That verifies:

- Python environment selection
- test suite health
- scenario catalog
- council-style terminal experience

Current local test result:

- `23 passed`

## 4. The Main User Flows

Think of SPECTRA in five practical flows:

1. `make demo`: single-scenario council story
2. `make council-local`: all-scenario local council pipeline
3. `make multi-agent-local-smoke` or `make multi-agent-smoke`: trace, dataset, and hint-pack collection
4. `make hinted` and `make hinted-check`: blind versus hinted replay
5. `make train-smoke` or `make train`: replay-backed GRPO workflow

If you only remember one sequence, use this:

```bash
make doctor
./.venv/bin/python -m pytest tests -q
make demo
```

## 5. Demo Experience

### Single-scenario money demo

```bash
make demo SCENARIO=broken_auth_cascade
```

This runs the three-phase story for one scenario:

1. blind commander
2. multi-agent council
3. hinted commander

Default output root:

- `outputs/council_pipeline/single_broken_auth_cascade/`

Convenience shortcuts:

```bash
make demo-easy
make demo-medium
make demo-hard
make demo-cache
```

### Full local council pipeline

```bash
make council-local
```

Default behavior:

- `COUNCIL_SCOPE=all`
- phases: `untrained,multi_agent,hinted`
- local commander model: `qwen2.5:3b`
- local multi-agent model: `qwen2.5:3b`
- specialist mode: `deterministic`

Default output root:

- `outputs/council_pipeline/`

### Run only one scenario in the council UI

```bash
make council-local COUNCIL_SCOPE=single SCENARIO=broken_auth_cascade
```

### Run only selected phases

```bash
make council-local COUNCIL_PHASES=untrained
make council-local COUNCIL_PHASES=multi_agent
make council-local COUNCIL_PHASES=hinted
```

### Hosted council variant

```bash
make council-hf
```

This keeps the untrained and hinted phases local, but switches the multi-agent council phase to the hosted HF commander stack.

## 6. Single-Agent Baselines

### Blind local commander

This is the clean baseline: one full-state commander with no hint help.

Before running it, make sure your local OpenAI-compatible endpoint exists.

For Ollama:

```bash
ollama serve
ollama list
```

Run the blind baseline:

```bash
make untrained SCENARIO=broken_auth_cascade LOCAL_MODEL=qwen2.5:1.5b
```

Default output folder:

- `outputs/untrained/`

### Direct single-agent freeform inference

```bash
make local-free SCENARIO=broken_auth_cascade LOCAL_MODEL=qwen2.5:1.5b
```

This goes through `inference.py` in `single_agent` mode without the council wrapper.

## 7. Multi-Agent Collection

This is the core benchmark artifact flow. One run can export:

- step-level JSONL data
- episode summary JSON
- raw traces
- a trace-derived hint pack

### Local no-HF smoke run

```bash
make multi-agent-local-smoke SCENARIO=broken_auth_cascade
```

Default local settings:

- commander provider: `openai`
- commander model: `qwen2.5:3b`
- specialist mode: `deterministic`
- output dir: `outputs/multi_agent/`

### Hosted or hybrid smoke run

```bash
make multi-agent-smoke SCENARIO=broken_auth_cascade COMMANDER_MODEL=Qwen/Qwen3-4B-Instruct-2507
```

Default hosted settings:

- commander provider: `hf`
- commander model: `Qwen/Qwen3-4B-Instruct-2507`
- commander HF provider: `nscale`
- specialist mode: `hybrid`

### Larger collection run

```bash
make multi-agent EPISODES=5 COMMANDER_MODEL=Qwen/Qwen3-4B-Instruct-2507
```

### Real hosted specialist collection

```bash
make real-collect-smoke
make real-collect EPISODES=5
```

These force `SPECIALIST_MODE=llm`.

### Local model matrix

```bash
make multi-agent-local-05b-smoke SCENARIO=broken_auth_cascade
make multi-agent-local-15b-smoke SCENARIO=broken_auth_cascade
make multi-agent-local-3b-smoke SCENARIO=broken_auth_cascade
make multi-agent-local-matrix SCENARIO=broken_auth_cascade
```

### Artifacts to inspect

After a multi-agent run, look at:

- `outputs/multi_agent/data.jsonl`
- `outputs/multi_agent/data.summary.json`
- `outputs/multi_agent/traces/`
- `outputs/multi_agent/hints.json`

### Specialist mode guidance

- `deterministic`: cheapest and most reproducible
- `hybrid`: hosted first, deterministic fallback on failure
- `llm`: strict hosted specialists only

Use `hybrid` when you want the pipeline to keep moving even if provider calls are brittle.

## 8. Hinted Replay

This re-runs a full-state local commander with the exported hint pack.

### Run the hinted commander

```bash
make hinted SCENARIO=broken_auth_cascade LOCAL_MODEL=qwen2.5:1.5b
```

Default output folder:

- `outputs/hinted/`

### Compare blind versus hinted

```bash
make hinted-check SCENARIO=broken_auth_cascade
```

Alias:

```bash
make hint-effect SCENARIO=broken_auth_cascade
```

Expected output folder:

- `outputs/hint_effect/`

This is the simplest way to test whether the trace-derived cheat sheet actually improves the local commander.

## 9. Hint Building And Dataset Smoke Tests

Build a hint pack directly from deterministic heuristic collection:

```bash
make build-hints EPISODES=5
```

Minimal dataset smoke test:

```bash
make dataset-smoke
```

This is useful when you want to validate artifact wiring without burning hosted credits.

## 10. GRPO Workflow

### Smoke test the training loop

```bash
make train-smoke
```

This does two things:

1. collects a tiny deterministic dataset
2. dry-runs `training/grpo_train.py` against it

### Dry-run training on an existing dataset

```bash
make grpo-dry-run DATASET_PATH=outputs/multi_agent/data.jsonl
```

### Real training run

```bash
make train DATASET_PATH=outputs/multi_agent/data.jsonl TRAIN_STEPS=20
```

Training output root:

- `outputs/grpo_runs/`

## 11. Direct Inference Commands

Single scenario:

```bash
make inference SCENARIO=database_sqli_outage
```

Pretty terminal rendering:

```bash
make inference-pretty SCENARIO=broken_auth_cascade
```

All scenarios:

```bash
make inference-all
```

The CLI itself supports:

```bash
./.venv/bin/python inference.py --help
```

Main choices:

- runtime: `local` or `remote`
- observation mode: `multi_agent` or `single_agent`
- specialist mode: `deterministic`, `hybrid`, or `llm`
- commander type: `heuristic`, `random`, `llm`, or `single-agent`

## 12. API Server And Remote Mode

### Start the local server

Terminal 1:

```bash
make server
```

For auto-reload:

```bash
make dev
```

### Run remote inference against the server

Terminal 2:

```bash
make remote REMOTE_SCENARIO=database_sqli_outage
make remote-pretty REMOTE_SCENARIO=database_sqli_outage
```

Default base URL:

- `http://127.0.0.1:8000`

The top-level server wrapper is `server/app.py`, and the real FastAPI/OpenEnv implementation is in `environments/pomir_env/server.py`.

## 13. Docker Flow

Build the image:

```bash
make docker-build
```

Run it:

```bash
make docker-run
```

The Docker entrypoint serves:

- `uvicorn server.app:app --host 0.0.0.0 --port 8000`

## 14. Validation Commands

### Tests

```bash
./.venv/bin/python -m pytest tests -q
```

Current local result:

- `23 passed`

### OpenEnv validation

```bash
./.venv/bin/openenv validate .
```

Current local result:

- not yet ready for multi-mode deployment because `uv.lock` is missing

The repo no longer fails validation on the server wrapper shape:

- `server = "server.app:main"` exists in `pyproject.toml`
- `server/app.py` exposes a callable `main()`

To clear the remaining issue:

```bash
uv lock
```

If `uv lock` needs network access in your environment, run it outside the restricted sandbox and then re-run:

```bash
./.venv/bin/openenv validate .
```

## 15. Recommended Paths

If you want the judge-facing story:

```bash
make demo
```

If you want the full local product flow:

```bash
make council-local
```

If you want benchmark artifacts without HF credits:

```bash
make multi-agent-local-smoke
```

If you want the strongest hosted multi-agent collection:

```bash
make multi-agent-smoke COMMANDER_MODEL=Qwen/Qwen3-4B-Instruct-2507
```

If you want the shortest training sanity check:

```bash
make train-smoke
```
