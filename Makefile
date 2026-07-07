PYTHON ?= python

.PHONY: build test lint validate

build:
	$(PYTHON) scripts/build_batched.py --inplace

test:
	$(PYTHON) scripts/run_pytest_batches.py

lint:
	ruff check .

validate: lint test
