.DEFAULT_GOAL := help

.PHONY: help doctor setup-venv which-python scenarios test smoke demo demo-easy demo-medium demo-hard demo-cache \
	inference inference-pretty inference-all local-free untrained hinted build-hints dataset-smoke multi-agent-smoke multi-agent \
	real-collect-smoke real-collect hinted-check hint-effect server dev remote remote-pretty train grpo-dry-run train-smoke \
	compare trust-probe clean docker-build docker-run

LOCAL_VENV_PYTHON  := .venv/bin/python
SHARED_VENV_PYTHON := /Users/madhav_189/Documents/Scalar_hackathon/openenv/.venv/bin/python

ifneq ("$(wildcard $(LOCAL_VENV_PYTHON))","")
VENV_PYTHON := $(LOCAL_VENV_PYTHON)
PYTHON_SOURCE := local
else
VENV_PYTHON := $(SHARED_VENV_PYTHON)
PYTHON_SOURCE := shared-openenv
endif

PYTHON := PYTHONPATH=. $(VENV_PYTHON)

SCENARIO        ?= broken_auth_cascade
REMOTE_SCENARIO ?= database_sqli_outage
SEED            ?= 42
BASE_URL        ?= http://127.0.0.1:8000
PORT            ?= 8000
OUTPUT_DIR      ?= outputs/raw_traces
CONDITION       ?= C
EPISODES        ?= 10
COMPARE_EPISODES ?= 9
TRUST_EPISODES  ?= 5
TRAIN_STEPS     ?= 20
LOCAL_MODEL     ?= qwen2.5:1.5b
LOCAL_PROVIDER  ?= openai
COMMANDER_MODEL ?= Qwen/Qwen2.5-7B-Instruct
COMMANDER_PROVIDER ?= hf
COMMANDER_BASE_URL ?= http://127.0.0.1:11434/v1
COMMANDER_HF_PROVIDER ?=
SPECIALIST_MODE ?= hybrid
UNTRAINED_OUTPUT_DIR ?= outputs/untrained
HINTED_OUTPUT_DIR ?= outputs/hinted
MULTI_AGENT_OUTPUT_DIR ?= outputs/multi_agent
HINT_FILE       ?= $(MULTI_AGENT_OUTPUT_DIR)/hints.json
DATASET_PATH    ?= $(MULTI_AGENT_OUTPUT_DIR)/data.jsonl
SUMMARY_PATH    ?= $(MULTI_AGENT_OUTPUT_DIR)/data.summary.json
TRACE_DIR       ?= $(MULTI_AGENT_OUTPUT_DIR)/traces

COMMANDER_HF_PROVIDER_ARG := $(if $(strip $(COMMANDER_HF_PROVIDER)),--commander-hf-provider $(COMMANDER_HF_PROVIDER),)
HINT_FILE_ARG := $(if $(strip $(HINT_FILE)),--hint-file $(HINT_FILE),)

doctor:
	@echo "Project: /Users/madhav_189/Documents/Scalar_hackathon/Madhav_task"
	@echo "Python source: $(PYTHON_SOURCE)"
	@echo "Python path  : $(VENV_PYTHON)"
	@$(VENV_PYTHON) -V
	@$(PYTHON) -c "from environments.shared.scenarios import list_scenarios; print('Scenarios:', ', '.join(s.scenario_id for s in list_scenarios()))"

setup-venv:
	python3 -m venv .venv
	./.venv/bin/python -m pip install --upgrade pip
	./.venv/bin/pip install -r requirements.txt
	@echo "Local virtualenv created at /Users/madhav_189/Documents/Scalar_hackathon/Madhav_task/.venv"
	@echo "Make targets will now prefer the local .venv automatically."

which-python:
	@echo $(VENV_PYTHON)

scenarios:
	@$(PYTHON) -c "from environments.shared.scenarios import list_scenarios; print('SCENARIO ID'.ljust(34) + 'DIFF'.ljust(10) + 'ROOT'.ljust(16) + 'NAME'); print('-' * 88); [print(s.scenario_id.ljust(34) + s.difficulty.ljust(10) + s.root_cause_service.ljust(16) + s.name) for s in list_scenarios()]"

test:
	$(PYTHON) -m pytest tests -q

smoke: test
	$(PYTHON) inference.py --scenario-id $(SCENARIO) --seed $(SEED) --output-dir $(OUTPUT_DIR)

demo:
	$(PYTHON) run_demo.py --scenario-id $(SCENARIO) --seed $(SEED)

demo-easy:
	$(MAKE) demo SCENARIO=database_sqli_outage

demo-medium:
	$(MAKE) demo SCENARIO=api_gateway_xss

demo-hard:
	$(MAKE) demo SCENARIO=worker_supply_chain_compromise

demo-cache:
	$(MAKE) demo SCENARIO=cache_poisoning_campaign

inference:
	$(PYTHON) inference.py --scenario-id $(SCENARIO) --seed $(SEED) --output-dir $(OUTPUT_DIR)

inference-pretty:
	$(PYTHON) inference.py --scenario-id $(SCENARIO) --seed $(SEED) --output-dir $(OUTPUT_DIR) --pretty

local-free:
	$(PYTHON) inference.py --scenario-id $(SCENARIO) --seed $(SEED) --commander llm --observation-mode single_agent --commander-base-url http://127.0.0.1:11434/v1 --commander-model $(LOCAL_MODEL) $(HINT_FILE_ARG) --output-dir $(OUTPUT_DIR)

untrained:
	$(PYTHON) inference.py --scenario-id $(SCENARIO) --seed $(SEED) --commander llm --commander-provider $(LOCAL_PROVIDER) --observation-mode single_agent --commander-base-url $(COMMANDER_BASE_URL) --commander-model $(LOCAL_MODEL) --output-dir $(UNTRAINED_OUTPUT_DIR)

hinted:
	$(PYTHON) inference.py --scenario-id $(SCENARIO) --seed $(SEED) --commander llm --commander-provider $(LOCAL_PROVIDER) --observation-mode single_agent --commander-base-url $(COMMANDER_BASE_URL) --commander-model $(LOCAL_MODEL) $(HINT_FILE_ARG) --output-dir $(HINTED_OUTPUT_DIR)

build-hints:
	$(PYTHON) inference.py --episodes $(EPISODES) --difficulty mixed --commander heuristic --observation-mode multi_agent --specialist-mode deterministic --output-dir $(OUTPUT_DIR) --export-hint-file $(HINT_FILE)

dataset-smoke:
	$(PYTHON) inference.py --episodes 2 --difficulty mixed --commander heuristic --observation-mode multi_agent --specialist-mode deterministic --dataset-path $(DATASET_PATH) --summary-path $(SUMMARY_PATH) --output-dir $(OUTPUT_DIR)

real-collect-smoke:
	$(MAKE) multi-agent-smoke SCENARIO=broken_auth_cascade SPECIALIST_MODE=llm MULTI_AGENT_OUTPUT_DIR=outputs/real_smoke DATASET_PATH=outputs/real_smoke/smoke.jsonl SUMMARY_PATH=outputs/real_smoke/smoke.summary.json TRACE_DIR=outputs/real_smoke/traces HINT_FILE=outputs/real_smoke/hints.json

real-collect:
	$(MAKE) multi-agent SPECIALIST_MODE=llm

multi-agent-smoke:
	$(PYTHON) inference.py --episodes 1 --scenario-id $(SCENARIO) --commander llm --commander-provider $(COMMANDER_PROVIDER) --commander-model $(COMMANDER_MODEL) --commander-base-url $(COMMANDER_BASE_URL) $(COMMANDER_HF_PROVIDER_ARG) --observation-mode multi_agent --specialist-mode $(SPECIALIST_MODE) --dataset-path $(DATASET_PATH) --summary-path $(SUMMARY_PATH) --trace-dir $(TRACE_DIR) --export-hint-file $(HINT_FILE) --output-dir $(MULTI_AGENT_OUTPUT_DIR)

multi-agent:
	$(PYTHON) inference.py --episodes $(EPISODES) --difficulty mixed --commander llm --commander-provider $(COMMANDER_PROVIDER) --commander-model $(COMMANDER_MODEL) --commander-base-url $(COMMANDER_BASE_URL) $(COMMANDER_HF_PROVIDER_ARG) --observation-mode multi_agent --specialist-mode $(SPECIALIST_MODE) --dataset-path $(DATASET_PATH) --summary-path $(SUMMARY_PATH) --trace-dir $(TRACE_DIR) --export-hint-file $(HINT_FILE) --output-dir $(MULTI_AGENT_OUTPUT_DIR)

hint-effect:
	$(MAKE) hinted-check

hinted-check:
	$(PYTHON) eval/hint_effect.py --episodes 1 --scenario-id $(SCENARIO) --runtime local --observation-mode multi_agent --specialist-mode $(SPECIALIST_MODE) --hint-file $(HINT_FILE) --commander-provider $(COMMANDER_PROVIDER) --commander-model $(COMMANDER_MODEL) --commander-base-url $(COMMANDER_BASE_URL) $(COMMANDER_HF_PROVIDER_ARG) --output-dir outputs/hint_effect

inference-all:
	$(PYTHON) inference.py --scenario-id database_sqli_outage --seed $(SEED) --output-dir $(OUTPUT_DIR)
	$(PYTHON) inference.py --scenario-id api_gateway_xss --seed $(SEED) --output-dir $(OUTPUT_DIR)
	$(PYTHON) inference.py --scenario-id cache_poisoning_campaign --seed $(SEED) --output-dir $(OUTPUT_DIR)
	$(PYTHON) inference.py --scenario-id broken_auth_cascade --seed $(SEED) --output-dir $(OUTPUT_DIR)
	$(PYTHON) inference.py --scenario-id worker_supply_chain_compromise --seed $(SEED) --output-dir $(OUTPUT_DIR)

server:
	$(PYTHON) -m uvicorn server.app:app --host 127.0.0.1 --port $(PORT)

dev:
	$(PYTHON) -m uvicorn server.app:app --reload --host 127.0.0.1 --port $(PORT)

remote:
	$(PYTHON) inference.py --runtime remote --base-url $(BASE_URL) --scenario-id $(REMOTE_SCENARIO) --seed $(SEED) --output-dir outputs/remote_smoke

remote-pretty:
	$(PYTHON) inference.py --runtime remote --base-url $(BASE_URL) --scenario-id $(REMOTE_SCENARIO) --seed $(SEED) --output-dir outputs/remote_smoke --pretty

grpo-dry-run:
	$(PYTHON) training/grpo_train.py --dataset $(DATASET_PATH) --dry-run

train-smoke:
	$(PYTHON) inference.py --episodes 2 --difficulty mixed --commander heuristic --observation-mode multi_agent --specialist-mode deterministic --dataset-path outputs/grpo_smoke/smoke.jsonl --summary-path outputs/grpo_smoke/smoke.summary.json --output-dir outputs/grpo_smoke
	$(PYTHON) training/grpo_train.py --dataset outputs/grpo_smoke/smoke.jsonl --dry-run

train:
	$(PYTHON) training/grpo_train.py --dataset $(DATASET_PATH) --output-dir outputs/grpo_runs --max-steps $(TRAIN_STEPS)

compare:
	$(PYTHON) eval/compare_conditions.py --episodes $(COMPARE_EPISODES) --output-dir outputs/eval

trust-probe:
	$(PYTHON) eval/trust_probe.py --episodes $(TRUST_EPISODES)

docker-build:
	docker buildx build --platform linux/amd64 -t spectra:latest .

docker-run:
	docker run -p 8000:8000 spectra:latest

clean:
	rm -rf outputs __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

help:
	@echo ""
	@echo "SPECTRA quick commands"
	@echo ""
	@echo "Python environment behavior"
	@echo ""
	@echo "  1. If ./.venv exists in Madhav_task, Make uses that."
	@echo "  2. Otherwise it falls back to /Users/madhav_189/Documents/Scalar_hackathon/openenv/.venv."
	@echo "  3. Run 'make setup-venv' if you want a local first-time setup."
	@echo ""
	@echo "  make doctor"
	@echo "      Show Python path and available scenarios."
	@echo ""
	@echo "  make setup-venv"
	@echo "      Create a local .venv in Madhav_task and install requirements."
	@echo ""
	@echo "  make test"
	@echo "      Run the pytest suite."
	@echo ""
	@echo "  make demo SCENARIO=broken_auth_cascade"
	@echo "      Run the terminal demo for one scenario."
	@echo ""
	@echo "  make inference SCENARIO=worker_supply_chain_compromise"
	@echo "      Collect a dataset-backed run and save JSONL + summary."
	@echo ""
	@echo "  make untrained SCENARIO=broken_auth_cascade LOCAL_MODEL=qwen2.5:3b"
	@echo "      Run one local commander with full-state access and no cheat sheet."
	@echo ""
	@echo "  make multi-agent-smoke SCENARIO=broken_auth_cascade"
	@echo "      Run one multi-agent collection episode and save traces + GRPO data + hints."
	@echo ""
	@echo "  make hinted SCENARIO=broken_auth_cascade LOCAL_MODEL=qwen2.5:3b"
	@echo "      Re-run the local commander with the generated cheat sheet."
	@echo ""
	@echo "  make hinted-check SCENARIO=broken_auth_cascade"
	@echo "      Compare the same setup blind vs hinted."
	@echo ""
	@echo "  make local-free SCENARIO=broken_auth_cascade LOCAL_MODEL=qwen2.5:1.5b"
	@echo "      Run the single-agent full-state flow against a local Ollama model."
	@echo ""
	@echo "  make build-hints EPISODES=5"
	@echo "      Build a reusable hint pack from successful multi-agent heuristic runs."
	@echo ""
	@echo "  make real-collect-smoke COMMANDER_MODEL=Qwen/Qwen2.5-7B-Instruct"
	@echo "      Alias for strict hosted multi-agent smoke collection."
	@echo ""
	@echo "  make multi-agent EPISODES=5 COMMANDER_MODEL=Qwen/Qwen2.5-7B-Instruct"
	@echo "      Main collection path. Default specialist mode is hybrid."
	@echo ""
	@echo "  make real-collect EPISODES=5 COMMANDER_MODEL=Qwen/Qwen2.5-7B-Instruct"
	@echo "      Alias for strict hosted multi-agent collection."
	@echo ""
	@echo "  make dataset-smoke"
	@echo "      Collect a small multi-agent dataset for training validation."
	@echo ""
	@echo "  make hinted-check SCENARIO=broken_auth_cascade HINT_FILE=$(HINT_FILE)"
	@echo "      Compare the same commander blind vs hinted on the same scenario."
	@echo ""
	@echo "  make server"
	@echo "      Start the local FastAPI/OpenEnv server."
	@echo ""
	@echo "  make remote REMOTE_SCENARIO=database_sqli_outage"
	@echo "      Run inference against a live local server."
	@echo ""
	@echo "Useful variables"
	@echo ""
	@echo "  SCENARIO=<id>         Local demo/inference scenario"
	@echo "  REMOTE_SCENARIO=<id>  Remote inference scenario"
	@echo "  SEED=<n>              Episode seed"
	@echo "  PORT=<n>              Server port"
	@echo "  BASE_URL=<url>        Remote server URL"
	@echo "  LOCAL_PROVIDER=openai Local OpenAI-compatible provider for untrained/hinted"
	@echo "  COMMANDER_PROVIDER=hf|openai"
	@echo "  COMMANDER_MODEL=<id>  LLM commander model for real collection"
	@echo "  COMMANDER_BASE_URL=<url> OpenAI-compatible endpoint for local commander runs"
	@echo "  SPECIALIST_MODE=hybrid|llm|deterministic"
	@echo "  CONDITION=A|B|C       Training condition for make train"
	@echo "  EPISODES=<n>          Episode count for make train"
	@echo ""
	@echo "Scenario aliases"
	@echo ""
	@echo "  make demo-easy"
	@echo "  make demo-medium"
	@echo "  make demo-hard"
	@echo "  make demo-cache"
	@echo ""
	@echo "Benchmark commands"
	@echo ""
	@echo "  make smoke"
	@echo "  make untrained"
	@echo "  make multi-agent-smoke"
	@echo "  make hinted"
	@echo "  make hinted-check"
	@echo "  make real-collect-smoke"
	@echo "  make train-smoke"
	@echo "  make grpo-dry-run DATASET_PATH=outputs/multi_agent/data.jsonl"
	@echo "  make train DATASET_PATH=outputs/multi_agent/data.jsonl TRAIN_STEPS=20"
	@echo "  make hint-effect"
	@echo "  make compare"
	@echo "  make trust-probe"
	@echo ""
