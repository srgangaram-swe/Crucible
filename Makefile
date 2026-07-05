# Crucible — developer entrypoints.
# All targets run offline with no external services.

PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: install lint format type test smoke clean hooks

$(VENV)/pyvenv.cfg:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

install: $(VENV)/pyvenv.cfg  ## Create venv and install package + dev tools
	$(BIN)/pip install -e ".[dev]"

hooks: install  ## Install pre-commit hooks
	$(BIN)/pre-commit install

lint:  ## Ruff + black (check only)
	$(BIN)/ruff check src tests
	$(BIN)/black --check src tests

format:  ## Auto-fix style
	$(BIN)/ruff check --fix src tests
	$(BIN)/black src tests

type:  ## mypy (strict)
	$(BIN)/mypy

test:  ## Unit + integration tests
	$(BIN)/pytest

smoke:  ## End-to-end pipeline on bundled tiny synthetic data (CPU, offline)
	$(BIN)/crucible smoke

clean:
	rm -rf $(VENV) build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
