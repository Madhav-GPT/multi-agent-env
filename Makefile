.DEFAULT_GOAL := help

.PHONY: help doctor setup-venv which-python scenarios test smoke demo demo-easy demo-medium demo-hard demo-cache \
	inference inference-pretty inference-all server dev remote remote-pretty \
	train train-smoke compare trust-probe clean docker-build docker-run

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

train-smoke:
	$(PYTHON) training/grpo_train.py --condition C --steps 2 --difficulty mixed --output-dir outputs/grpo_smoke

train:
	$(PYTHON) training/grpo_train.py --condition $(CONDITION) --steps $(EPISODES) --difficulty mixed --output-dir outputs/grpo_runs

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
	@echo "      Run local inference and save a trace."
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
	@echo "  make train-smoke"
	@echo "  make train CONDITION=C EPISODES=10"
	@echo "  make compare"
	@echo "  make trust-probe"
	@echo ""
