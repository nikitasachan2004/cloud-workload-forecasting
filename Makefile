VENV ?= .venv
PYTHON := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
PYTEST := $(if $(wildcard $(VENV)/bin/pytest),$(VENV)/bin/pytest,pytest)

.PHONY: help install test clean

help:
	@echo "Cloud Workload Forecasting Challenge - Targets:"
	@echo "  install   : Install dependencies in editable mode"
	@echo "  test      : Run automated pytest suite"
	@echo "  clean     : Remove temporary caches and build artifacts"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTEST) -v

clean:
	rm -rf .pytest_cache build dist *.egg-info __pycache__ src/cwfc/__pycache__ tests/__pycache__
