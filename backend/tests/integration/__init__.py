"""Integration tests for the AI Focus Group Simulator.

This package contains integration tests that verify the complete system behavior,
including:
- Full session flow from config to output (test_end_to_end.py)
- CLI command validation and execution (test_cli.py)
- Component integration and error handling (test_components.py)

Tests use MockLLMProvider to eliminate dependency on actual LLM backends.

Usage:
    pytest tests/integration/ -v
    pytest tests/integration/ -m integration
    pytest tests/integration/ -k "session"
"""
