.PHONY: test-all test-api test-unit test-integration test-ai-long coverage help

PYTEST  := .venv/bin/python -m pytest
COVARGS := --cov=api --cov=src/focus_group --cov-report=term-missing

# test_end_to_end.py imports focus_group.cli which does not exist yet
BROKEN  := --ignore=tests/integration/test_end_to_end.py

help:
	@echo "make test-all         Full suite (excludes broken end-to-end test)"
	@echo "make test-api         API layer only: auth, jobs, health (fast, no Ollama)"
	@echo "make test-unit        Simulation engine unit tests only"
	@echo "make test-integration Integration tests (no Ollama required)"
	@echo "make test-ai-long     Live AI smoke tests (requires Ollama running)"
	@echo "make coverage         Full suite + coverage report (term-missing)"

test-all:
	$(PYTEST) tests/ $(BROKEN) -v

test-api:
	$(PYTEST) tests/api/ -v

test-unit:
	$(PYTEST) tests/unit/ -v

test-integration:
	$(PYTEST) tests/integration/ $(BROKEN) -v

test-ai-long:
	$(PYTEST) tests/smoke/ --run-ollama -v

coverage:
	$(PYTEST) tests/ $(BROKEN) $(COVARGS)
