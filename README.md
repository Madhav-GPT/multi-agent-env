# SPECTRA

SPECTRA is a terminal-first OpenEnv benchmark for incident response under partial observability. The environment can now run in two honest modes:

- `multi_agent`: three specialists see different state partitions and the commander acts on their reports
- `single_agent`: the commander gets the full world state for local-model freedom and baseline comparison

For the quickest hands-on walkthrough, start with [execution.md](/Users/madhav_189/Documents/Scalar_hackathon/Madhav_task/execution.md).

The repo is built around three practical runs:

1. `make untrained`: run a local commander model on the full environment with no help
2. `make multi-agent`: collect step-level JSONL data plus raw episode traces and a cheat sheet from the agentic environment
3. `make hinted` plus `training/grpo_train.py`: test the cheat sheet on the same local commander, then replay the saved data for GRPO

## What Changed

The current repo is no longer just a terminal showcase:

- `inference.py` now writes a real step dataset instead of only a trace blob
- `inference.py` can also export raw per-episode trace JSON with specialist raw outputs and commander outputs
- `training/grpo_train.py` is now an environment-backed GRPO entrypoint instead of an A/B/C comparison stub
- per-step reward is clamped to `[0.0, 1.0]`, which makes it safe to use as a training signal
- a trace-derived hint-pack generator exports reusable JSON cheat sheets without oracle root-cause leakage
- single-agent and multi-agent observation modes are both first-class environment settings
- the observation contract now includes stage goals, required action fields, valid examples, and loop warnings inspired by the `openenv` reference repo
- specialist models are configurable by env vars, and `hybrid` mode is the practical real-world fallback when strict hosted LLM mode is brittle or provider credits are exhausted

## Architecture

```text
MasterSREEnv (hidden state)
  -> InfraEnv -> Infra specialist / extractor
  -> LogEnv   -> Log specialist / extractor
  -> SecEnv   -> Security specialist / extractor
  -> POMIREnv orchestrator
       -> multi_agent prompt from SpecialistReport objects
       -> single_agent prompt from full MasterSREState
       -> bounded reward + round history
```

The benchmark still keeps one hidden `MasterSREState` and strict typed partitions. The main difference is that the execution layer is now productized around dataset collection and training instead of only a pretty terminal loop.

## Run Modes

### 1. Untrained Local Commander

Run a local Ollama model on the full-state environment with no hint help:

```bash
make untrained SCENARIO=broken_auth_cascade LOCAL_MODEL=qwen2.5:3b
```

This uses:

- `observation_mode=single_agent`
- local OpenAI-compatible inference at `http://127.0.0.1:11434/v1`
- no hint pack

### 2. Multi-Agent Data Collection

Collect multi-agent JSONL data:

```bash
make multi-agent-smoke SCENARIO=broken_auth_cascade
```

Or collect a larger live set:

```bash
make multi-agent EPISODES=20 COMMANDER_MODEL=Qwen/Qwen2.5-7B-Instruct
```

Each JSONL row stores:

- `prompt`
- `completion`
- `reference_action`
- `reward`
- `reward_breakdown`
- `scenario_id`, `seed`, `step_index`
- `prior_actions`
- `observation_mode` and `specialist_mode`

That is enough to replay the exact state later during training.

This writes three artifacts:

- step-level GRPO dataset JSONL
- per-episode raw trace JSON
- trace-derived hint pack

Use `SPECIALIST_MODE=hybrid` for the practical live path and `SPECIALIST_MODE=llm` for strict hosted-only specialists.

### 3. Hinted Local Commander

Re-run the same local commander with the generated cheat sheet:

```bash
make hinted SCENARIO=broken_auth_cascade LOCAL_MODEL=qwen2.5:3b
```

Measure blind vs hinted improvement:

```bash
make hinted-check SCENARIO=broken_auth_cascade
```

### 4. GRPO Training

Validate the training dataset and reward replay:

```bash
make train-smoke
```

Dry-run the trainer on an existing dataset:

```bash
make grpo-dry-run DATASET_PATH=outputs/multi_agent/data.jsonl
```

Launch real GRPO training:

```bash
make train DATASET_PATH=outputs/multi_agent/data.jsonl TRAIN_STEPS=20
```

`training/grpo_train.py` replays the environment to the saved step and computes reward from the candidate completion's parsed action. It does not rely on a baked-in reward number from the dataset.
Replay uses deterministic specialists by default, even when the original dataset was collected in `hybrid` or `llm` mode.

## Specialist Modes

`POMIREnv` supports:

- `deterministic`: cheap extractor-backed specialists, good for reproducible training runs
- `hybrid`: try live LLM specialists and fall back to extractors when providers fail or credits run out
- `llm`: require live specialist calls

The default repo flow keeps `deterministic` as the practical training path and leaves `llm`/`hybrid` available when you want richer behavior.

## Scenarios

| Scenario | Difficulty | Root service | Core twist |
| --- | --- | --- | --- |
| `database_sqli_outage` | easy | `database` | clean SQLi causal chain |
| `api_gateway_xss` | medium | `api-gateway` | gateway recovery needs `scale_service` |
| `broken_auth_cascade` | hard | `auth_service` | infra is factually right about cache pain but wrong about cause |
| `worker_supply_chain_compromise` | hard | `worker` | poisoned worker release destabilizes downstream systems |
| `cache_poisoning_campaign` | medium | `cache` | logs mislead toward `api-gateway` |

## Useful Commands

```bash
make doctor
make test
make demo SCENARIO=broken_auth_cascade
make untrained SCENARIO=broken_auth_cascade LOCAL_MODEL=qwen2.5:3b
make multi-agent-smoke SCENARIO=broken_auth_cascade
make hinted SCENARIO=broken_auth_cascade LOCAL_MODEL=qwen2.5:3b
make hinted-check SCENARIO=broken_auth_cascade
make train-smoke
make compare
```

## Validation

Validated locally after the rebuild:

- `21` pytest checks passing
- 5-episode mixed multi-agent dataset collection with exported hint pack
- single-agent local Ollama run with the generated hint pack resolving `broken_auth_cascade`
- GRPO reward replay dry run on the collected JSONL dataset
- real/hybrid multi-agent collection on `broken_auth_cascade` with raw traces and trace-derived hints
- blind vs hinted comparison showing the hinted run improved from `1.229` reward / `5` steps to `1.443` reward / `4` steps on the same setup
