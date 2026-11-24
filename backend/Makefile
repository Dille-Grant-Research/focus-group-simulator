.PHONY: test-all test-api test-unit test-integration test-ai-long coverage help

PYTEST  := .venv/bin/python -m pytest
COVARGS := --cov=api --cov=src/focus_group --cov-report=term-missing


help:
	@echo "make test-all         Full suite"
	@echo "make test-api         API layer only: auth, jobs, health, worker (fast, no LLM)"
	@echo "make test-unit        Simulation engine unit tests only"
	@echo "make test-integration Integration tests (no LLM required)"
	@echo "make test-ai-long     Live AI smoke tests (uses LLM_* from .env, e.g. cruncher via tunnel)"
	@echo "make coverage         Full suite + coverage report (term-missing)"

test-all:
	$(PYTEST) tests/ -v

test-api:
	$(PYTEST) tests/api/ -v

test-unit:
	$(PYTEST) tests/unit/ -v

test-integration:
	$(PYTEST) tests/integration/ -v

test-ai-long:
	$(PYTEST) tests/smoke/ --run-ollama -v

coverage:
	$(PYTEST) tests/ $(COVARGS)
