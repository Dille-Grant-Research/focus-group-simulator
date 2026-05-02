"""Pytest configuration and fixtures for the focus group simulator tests.

This module provides shared fixtures for all tests, including mock providers,
sample configurations, and temporary directories.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from focus_group.models.config import (
    CloudModelConfig,
    ContextConfig,
    LLMProviderType,
    OllamaModelConfig,
    OutputConfig,
    SessionConfig,
)
from focus_group.models.persona import Persona
from tests.mocks.mock_provider import MockLLMProvider, SequentialMockProvider

# ============================================================================
# Provider Fixtures
# ============================================================================


@pytest.fixture
def mock_provider() -> MockLLMProvider:
    """Provide a mock LLM provider for testing.

    Returns:
        A configured MockLLMProvider instance
    """
    return MockLLMProvider(
        model_name="mock-model-for-testing",
        max_context_tokens=8192,
        available=True,
    )


@pytest.fixture
def unavailable_provider() -> MockLLMProvider:
    """Provide a mock provider that appears unavailable.

    Returns:
        A MockLLMProvider configured to be unavailable
    """
    return MockLLMProvider(
        model_name="unavailable-model",
        max_context_tokens=8192,
        available=False,
    )


@pytest.fixture
def sequential_provider() -> SequentialMockProvider:
    """Provide a sequential mock provider.

    Returns:
        A SequentialMockProvider with some default responses
    """
    return SequentialMockProvider(
        responses=[
            '{"name": "Test Person", "age": 35, "background": "Test background", '
            '"personality_traits": ["friendly"], "communication_style": "Clear"}',
            "Hello, I am Test Person. Nice to meet everyone.",
            "leans forward with interest",
            '{"selected_speaker": "Test Person", "reasoning": "Appears eager", '
            '"someone_wants_to_speak": true}',
            "That is an interesting point to consider.",
        ],
        loop=True,
    )


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def sample_demographics() -> list[str]:
    """Provide sample demographic specifications.

    Returns:
        List of demographic strings
    """
    return [
        "Female, 30-40, tech industry professional",
        "Male, 50-60, retired teacher",
        "Non-binary, 25-35, graduate student in psychology",
    ]


@pytest.fixture
def sample_questions() -> list[str]:
    """Provide sample focus group questions.

    Returns:
        List of question strings
    """
    return [
        "What do you think about remote work and its impact on work-life balance?",
        "How do you stay productive when working from home?",
    ]


@pytest.fixture
def sample_model_config() -> OllamaModelConfig:
    """Provide a sample model configuration.

    Returns:
        OllamaModelConfig instance
    """
    return OllamaModelConfig(
        model_name="mock-model",
        base_url="http://localhost:11434",
        temperature=0.7,
        num_ctx=8192,
    )


@pytest.fixture
def sample_context_config() -> ContextConfig:
    """Provide a sample context configuration.

    Returns:
        ContextConfig instance
    """
    return ContextConfig(
        max_tokens=4096,
        summarize_after_tokens=3000,
        reserved_for_persona=500,
        reserved_for_question=200,
        reserved_for_recent_response=500,
    )


@pytest.fixture
def sample_output_config(tmp_path: Path) -> OutputConfig:
    """Provide a sample output configuration.

    Args:
        tmp_path: Pytest fixture for temporary directory

    Returns:
        OutputConfig instance pointing to temp directory
    """
    return OutputConfig(
        directory=str(tmp_path / "output"),
        save_debug_log=True,
        save_clean_transcript=True,
        log_format="json",
    )


@pytest.fixture
def sample_session_config(
    sample_demographics: list[str],
    sample_questions: list[str],
    sample_model_config: OllamaModelConfig,
    sample_context_config: ContextConfig,
    sample_output_config: OutputConfig,
) -> SessionConfig:
    """Provide a complete session configuration.

    Returns:
        SessionConfig instance with all nested configs
    """
    return SessionConfig(
        session_name="test_session",
        demographics=sample_demographics,
        questions=sample_questions,
        purpose="Test focus group discussion",
        closing_question="Any final thoughts?",
        model=sample_model_config,
        context=sample_context_config,
        output=sample_output_config,
        validate_personas=True,
    )


@pytest.fixture
def minimal_session_config(
    sample_model_config: OllamaModelConfig,
    tmp_path: Path,
) -> SessionConfig:
    """Provide a minimal session configuration with 1 participant.

    Useful for fast tests that don't need multiple participants.

    Returns:
        SessionConfig with minimal settings
    """
    return SessionConfig(
        session_name="minimal_test",
        demographics=["Test participant, 30-40, professional"],
        questions=["What is your opinion on testing?"],
        purpose="Minimal test",
        model=sample_model_config,
        output=OutputConfig(
            directory=str(tmp_path / "output"),
            save_debug_log=False,
            save_clean_transcript=False,
        ),
        validate_personas=False,
    )


@pytest.fixture
def sample_cloud_config() -> CloudModelConfig:
    """Provide a sample cloud model configuration.

    Returns:
        CloudModelConfig instance for testing
    """
    return CloudModelConfig(
        provider=LLMProviderType.OPENAI,
        model_name="deepseek-chat",
        api_key_env_var="TEST_API_KEY",
        base_url="https://api.deepseek.com",
        max_tokens=2048,
        temperature=0.7,
        top_p=0.9,
        num_ctx=8192,
    )


# ============================================================================
# Persona Fixtures
# ============================================================================


@pytest.fixture
def sample_persona() -> Persona:
    """Provide a sample persona.

    Returns:
        A valid Persona instance
    """
    return Persona(
        name="Sarah Chen",
        age=34,
        background=(
            "Senior software engineer at a major tech company with 10 years "
            "of experience. Leads a team of developers working on cloud infrastructure."
        ),
        personality_traits=["analytical", "collaborative", "detail-oriented"],
        communication_style="Clear and structured, tends to use examples",
        demographic_source="Female, 30-40, tech industry professional",
    )


@pytest.fixture
def sample_personas() -> list[Persona]:
    """Provide multiple sample personas.

    Returns:
        List of Persona instances
    """
    return [
        Persona(
            name="Sarah Chen",
            age=34,
            background="Senior software engineer with 10 years experience",
            personality_traits=["analytical", "collaborative"],
            communication_style="Clear and structured",
            demographic_source="Female, 30-40, tech",
        ),
        Persona(
            name="Robert Wilson",
            age=57,
            background="Retired high school teacher with 30 years in education",
            personality_traits=["patient", "reflective"],
            communication_style="Thoughtful and measured",
            demographic_source="Male, 50-60, retired teacher",
        ),
        Persona(
            name="Jordan Martinez",
            age=28,
            background="PhD candidate in organizational psychology",
            personality_traits=["curious", "articulate"],
            communication_style="Enthusiastic and well-informed",
            demographic_source="Non-binary, 25-35, graduate student",
        ),
    ]


# ============================================================================
# File Fixtures
# ============================================================================


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for config files.

    Args:
        tmp_path: Pytest fixture for temporary directory

    Returns:
        Path to a configs subdirectory
    """
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for output files.

    Args:
        tmp_path: Pytest fixture for temporary directory

    Returns:
        Path to an output subdirectory
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def sample_session_yaml(
    temp_config_dir: Path,
    sample_demographics: list[str],
    sample_questions: list[str],
    temp_output_dir: Path,
) -> Path:
    """Create a sample session YAML config file.

    Returns:
        Path to the created YAML file
    """
    config_data = {
        "session_name": "yaml_test_session",
        "demographics": sample_demographics,
        "questions": sample_questions,
        "purpose": "Test session from YAML",
        "closing_question": "Any final thoughts from the test?",
        "model": {
            "model_name": "mock-model",
            "base_url": "http://localhost:11434",
            "temperature": 0.7,
            "num_ctx": 8192,
        },
        "context": {
            "max_tokens": 4096,
            "summarize_after_tokens": 3000,
        },
        "output": {
            "directory": str(temp_output_dir),
            "save_debug_log": True,
            "save_clean_transcript": True,
            "log_format": "json",
        },
        "validate_personas": True,
    }

    yaml_path = temp_config_dir / "test_session.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, default_flow_style=False)

    return yaml_path


@pytest.fixture
def sample_demographics_yaml(
    temp_config_dir: Path,
    sample_demographics: list[str],
) -> Path:
    """Create a sample demographics YAML file.

    Returns:
        Path to the created YAML file
    """
    data = {"demographics": sample_demographics}

    yaml_path = temp_config_dir / "test_demographics.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)

    return yaml_path


# ============================================================================
# Fixtures for Mock Responses
# ============================================================================


@pytest.fixture
def mock_responses() -> dict[str, Any]:
    """Load the mock responses from the fixtures file.

    Returns:
        Dictionary of mock responses
    """
    fixtures_path = Path(__file__).parent / "fixtures" / "mock_responses.json"
    if fixtures_path.exists():
        with open(fixtures_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_ollama: marks tests that require Ollama running")


def pytest_collection_modifyitems(config, items):
    """Skip tests based on markers and environment."""
    # Add skip marker for tests that require Ollama when not available
    skip_ollama = pytest.mark.skip(reason="Ollama not available")
    for item in items:
        if "requires_ollama" in item.keywords:
            # Could add logic here to check if Ollama is running
            # For now, always skip unless explicitly requested
            if not config.getoption("--run-ollama", default=False):
                item.add_marker(skip_ollama)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-ollama",
        action="store_true",
        default=False,
        help="run tests that require Ollama",
    )
