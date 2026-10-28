.PHONY: help install lint format test live bench catalog verify
.DEFAULT_GOAL := help
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

help: ## show available commands
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z_-]+:.*## / {printf "%-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## install the project and dev tools
	uv sync

lint: ## check style and formatting
	@! grep -oE 'https?://[^/"]+' uv.lock | grep -vxE 'https://(pypi\.org|files\.pythonhosted\.org)' || (echo "uv.lock points outside PyPI" && false)
	uv run ruff check mpc tests bench
	uv run ruff format --check mpc tests bench

format: ## apply formatting fixes
	uv run ruff check --fix mpc tests bench
	uv run ruff format mpc tests bench

test: ## offline tests
	uv run pytest

live: ## tests against the live Planetary Computer
	MPC_LIVE=1 uv run pytest tests/test_live.py

bench: ## time agents on bench/tasks.json, 4 runs each (about 16 sessions, $10); LABEL=...
	uv run python bench/run.py --repeat 4 --jobs 8 --label "$(LABEL)"

catalog: ## rebuild the catalog snapshot that ships in the package
	uv run python -c "from pathlib import Path; from mpc import catalog; catalog.refresh(None, out=Path('mpc/catalog.json'), log=print)"

verify: lint test ## run before handing over changes
