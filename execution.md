# SPECTRA Execution Guide

This is the easiest way to experience the repo as a product instead of a pile of scripts.

Project root:

```bash
cd /Users/madhav_189/Documents/Scalar_hackathon/Madhav_task
```

## The Main Flow

Think of SPECTRA in four user-visible steps:

1. `make untrained`: local commander, no help
2. `make multi-agent-smoke`: real environment run that creates traces, a cheat sheet, and GRPO data
3. `make hinted`: same local commander, now with the generated cheat sheet
4. `make grpo-dry-run`: verify the collected dataset is usable for GRPO replay

If you only remember one sequence, use this:

```bash
make doctor
make test
make untrained SCENARIO=broken_auth_cascade LOCAL_MODEL=qwen2.5:3b
make multi-agent-smoke SCENARIO=broken_auth_cascade COMMANDER_MODEL=Qwen/Qwen2.5-7B-Instruct
make hinted SCENARIO=broken_auth_cascade LOCAL_MODEL=qwen2.5:3b
make hinted-check SCENARIO=broken_auth_cascade
make grpo-dry-run DATASET_PATH=outputs/multi_agent/data.jsonl
```

## 1. Python Environment

There are two supported ways to run the repo.

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
/Users/madhav_189/Documents/Scalar_hackathon/openenv/.venv
```

### Check what the repo is using

```bash
make doctor
```

## 2. Optional `.env`

You can create a local `.env` from the template:

```bash
cp .env.example .env
```

Useful keys:

- `HF_TOKEN`
- `INFRA_SPECIALIST_MODEL`
- `LOG_SPECIALIST_MODEL`
- `SEC_SPECIALIST_MODEL`
- `COMMANDER_PROVIDER`
- `COMMANDER_MODEL`
- `API_BASE_URL`
- `OPENAI_API_KEY`

Important:

- `.env` is ignored by git
- specialist model IDs are configurable because hosted provider availability changes
- `SPECIALIST_MODE=hybrid` is the practical default when strict hosted LLM mode is brittle or credits are exhausted

## 3. First 5 Minutes

For a first-time user:

```bash
make doctor
make test
make scenarios
make demo SCENARIO=broken_auth_cascade
```

That verifies:

- Python environment
- test suite
- scenario catalog
- terminal walkthrough behavior

## 4. Run 1: Untrained Commander

This is the clean baseline: one local model with full-state access and no cheat sheet.

### Before you run it

Make sure your local OpenAI-compatible endpoint exists.

For Ollama:

```bash
ollama serve
ollama list
```

### Run it

```bash
make untrained SCENARIO=broken_auth_cascade LOCAL_MODEL=qwen2.5:3b
```

Default artifact folder:

- `outputs/untrained/`

This answers the question:

- what does the commander do when it knows nothing?

## 5. Run 2: Multi-Agent Collection

This is the core environment run. It creates both downstream products:

- a GRPO-ready dataset
- a cheat-sheet JSON distilled from successful traces

### Smoke run

```bash
make multi-agent-smoke SCENARIO=broken_auth_cascade COMMANDER_MODEL=Qwen/Qwen2.5-7B-Instruct
```

Default artifact folder:

- `outputs/multi_agent/`

Files to inspect:

- `outputs/multi_agent/data.jsonl`
- `outputs/multi_agent/data.summary.json`
- `outputs/multi_agent/traces/`
- `outputs/multi_agent/hints.json`

### Larger run

```bash
make multi-agent EPISODES=5 COMMANDER_MODEL=Qwen/Qwen2.5-7B-Instruct
```

### Strict vs practical live mode

- `SPECIALIST_MODE=llm`: strict hosted specialists only
- `SPECIALIST_MODE=hybrid`: hosted specialists first, deterministic fallback on failure

Use `hybrid` when you want the pipeline to keep moving despite provider instability.

Strict hosted example:

```bash
make multi-agent-smoke SCENARIO=broken_auth_cascade COMMANDER_MODEL=Qwen/Qwen2.5-7B-Instruct SPECIALIST_MODE=llm
```

### Manual version

```bash
./.venv/bin/python inference.py \
  --episodes 5 \
  --difficulty mixed \
  --commander llm \
  --commander-provider hf \
  --commander-model Qwen/Qwen2.5-7B-Instruct \
  --observation-mode multi_agent \
  --specialist-mode hybrid \
  --dataset-path outputs/my_run/data.jsonl \
  --summary-path outputs/my_run/data.summary.json \
  --trace-dir outputs/my_run/traces \
  --export-hint-file outputs/my_run/hints.json
```

## 6. Run 3: Hinted Commander

This re-runs the same local commander, but now it receives the cheat sheet produced by the multi-agent collection run.

### Run it

```bash
make hinted SCENARIO=broken_auth_cascade LOCAL_MODEL=qwen2.5:3b
```

Default artifact folder:

- `outputs/hinted/`

This answers the question:

- does the trace-derived cheat sheet improve the raw local commander?

### Measure blind vs hinted

```bash
make hinted-check SCENARIO=broken_auth_cascade
```

Outputs:

- `outputs/hint_effect/blind.jsonl`
- `outputs/hint_effect/hinted.jsonl`
- `outputs/hint_effect/comparison.json`

### Manual version

```bash
./.venv/bin/python inference.py \
  --scenario-id broken_auth_cascade \
  --commander llm \
  --commander-provider openai \
  --commander-base-url http://127.0.0.1:11434/v1 \
  --commander-model qwen2.5:3b \
  --observation-mode single_agent \
  --hint-file outputs/multi_agent/hints.json \
  --output-dir outputs/hinted
```

## 7. Run 4: GRPO Replay

This validates that the collected dataset is usable for environment-backed GRPO reward replay.

### Dry run

```bash
make grpo-dry-run DATASET_PATH=outputs/multi_agent/data.jsonl
```

Replay uses deterministic specialists by default, even if the original data was collected in `hybrid` or `llm` mode.

### End-to-end smoke

```bash
make train-smoke
```

### Full training entrypoint

```bash
make train DATASET_PATH=outputs/multi_agent/data.jsonl TRAIN_STEPS=20
```

Use this when you have the base checkpoint you actually want to train.

## 8. Explore The Project

### Try more scenarios

```bash
make demo SCENARIO=database_sqli_outage
make demo SCENARIO=api_gateway_xss
make demo SCENARIO=cache_poisoning_campaign
make demo SCENARIO=broken_auth_cascade
make demo SCENARIO=worker_supply_chain_compromise
```

### Save local runs

```bash
make inference
make inference-pretty
make inference-all
```

### Compare broader benchmark conditions

```bash
make compare
make trust-probe
```

## 9. Inspect Outputs

Useful folders:

- `outputs/untrained/`
- `outputs/multi_agent/`
- `outputs/hinted/`
- `outputs/hint_effect/`
- `outputs/grpo_runs/`
- `outputs/eval/`

Quick inspection commands:

```bash
sed -n '1,5p' outputs/multi_agent/data.jsonl
python -m json.tool outputs/multi_agent/data.summary.json
python -m json.tool outputs/multi_agent/hints.json
python -m json.tool outputs/hint_effect/comparison.json
```

## 10. Server / API Mode

Use this when you want the FastAPI/OpenEnv-style server path.

### Terminal 1

```bash
make server
```

### Terminal 2

```bash
make remote
```

Pretty remote mode:

```bash
make remote-pretty
```

## 11. Common Overrides

Examples:

```bash
make demo SCENARIO=cache_poisoning_campaign SEED=99
make multi-agent EPISODES=3 COMMANDER_MODEL=Qwen/Qwen2.5-7B-Instruct
make multi-agent-smoke SCENARIO=broken_auth_cascade SPECIALIST_MODE=llm
make untrained SCENARIO=worker_supply_chain_compromise LOCAL_MODEL=qwen2.5:3b
make hinted SCENARIO=worker_supply_chain_compromise LOCAL_MODEL=qwen2.5:3b
make server PORT=8010
make remote BASE_URL=http://127.0.0.1:8010
```

Useful variables:

- `SCENARIO`
- `REMOTE_SCENARIO`
- `SEED`
- `PORT`
- `BASE_URL`
- `LOCAL_MODEL`
- `LOCAL_PROVIDER`
- `COMMANDER_PROVIDER`
- `COMMANDER_MODEL`
- `COMMANDER_BASE_URL`
- `COMMANDER_HF_PROVIDER`
- `SPECIALIST_MODE`
- `HINT_FILE`
- `DATASET_PATH`
- `SUMMARY_PATH`
- `TRACE_DIR`
- `EPISODES`
- `TRAIN_STEPS`

## 12. Common Problems

### `argument --commander-hf-provider: expected one argument`

This used to happen when the CLI flag was emitted with an empty value. The runner now tolerates an empty optional provider, and the Make targets now omit the flag when it is blank.

### `make doctor` shows the wrong Python

Create a local environment:

```bash
make setup-venv
```

### `make untrained` or `make hinted` cannot reach the model

Usually:

- Ollama is not running
- the model is not installed
- the endpoint is not `http://127.0.0.1:11434/v1`

### Hosted specialists fail randomly

Use:

```bash
make multi-agent-smoke SPECIALIST_MODE=hybrid
```

This keeps the collection pipeline alive when hosted providers or credits are unstable.

### `make grpo-dry-run` fails with missing dataset

Create data first:

```bash
make multi-agent-smoke
```

### Hint file not found

Generate it first:

```bash
make multi-agent-smoke
```

## 13. Smallest Useful Demo

If you need the shortest convincing walkthrough:

```bash
make doctor
make test
make untrained SCENARIO=broken_auth_cascade LOCAL_MODEL=qwen2.5:3b
make multi-agent-smoke SCENARIO=broken_auth_cascade COMMANDER_MODEL=Qwen/Qwen2.5-7B-Instruct
make hinted SCENARIO=broken_auth_cascade LOCAL_MODEL=qwen2.5:3b
make hinted-check SCENARIO=broken_auth_cascade
make grpo-dry-run DATASET_PATH=outputs/multi_agent/data.jsonl
```

That sequence shows:

- the repo runs
- the raw commander baseline exists
- the multi-agent environment creates traces, hints, and GRPO data
- the hinted local run is a separate measurable step
- the training replay path works
