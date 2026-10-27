VENV ?= .venv
PYTHON := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
PYTEST := $(if $(wildcard $(VENV)/bin/pytest),$(VENV)/bin/pytest,pytest)

.PHONY: help install test eda predict run-all clean

help:
	@echo "Cloud Workload Forecasting Challenge - Targets:"
	@echo "  install   : Install dependencies in editable mode"
	@echo "  test      : Run automated pytest suite including leakage and smoke tests"
	@echo "  eda       : Run exploratory data analysis and generate report figures"
	@echo "  predict   : Run end-to-end modeling pipeline and generate predictions.csv"
	@echo "  run-all   : Execute tests, EDA, experiments, and prediction generation"
	@echo "  clean     : Remove temporary caches and build artifacts"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTEST) -v

eda:
	$(PYTHON) scripts/run_eda.py

predict:
	$(PYTHON) -m cwfc.predict --data challenge_data --output predictions.csv

run-all: test eda
	$(PYTHON) scripts/run_experiments.py
	$(PYTHON) -m cwfc.predict --data challenge_data --output predictions.csv

clean:
	rm -rf .pytest_cache build dist *.egg-info __pycache__ src/cwfc/__pycache__ tests/__pycache__
