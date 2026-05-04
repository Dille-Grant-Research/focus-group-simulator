"""Step 27: Live AI smoke test — persona generation.

Requires Ollama running with the configured model loaded.
Gate: --run-ollama flag (never fires in CI or make test-all).

Run with:
    make test-ai-long
    # or directly:
    pytest tests/smoke/ --run-ollama -v
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from focus_group.llm.ollama import OllamaProvider
from focus_group.models.config import OllamaModelConfig
from focus_group.persona.generator import PersonaGenerator

_SMOKE_DEMOGRAPHIC = "Female, age 38, rural Minnesota, registered nurse"


@pytest.mark.requires_ollama
async def test_first_persona_generated() -> None:
    """Generate exactly one persona and assert required fields are present.

    Exits as soon as the first persona is confirmed — does not run
    the full simulation.
    """
    host = os.environ.get("OLLAMA_HOST")
    port = int(os.environ.get("OLLAMA_PORT"))
    model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")

    config = OllamaModelConfig(
        model_name=model,
        base_url=f"http://{host}:{port}",
    )
    provider = OllamaProvider(config)

    if not await provider.is_available():
        pytest.skip(f"Ollama not reachable at http://{host}:{port}")

    generator = PersonaGenerator(provider)
    persona = await generator.generate_persona(_SMOKE_DEMOGRAPHIC)

    assert persona.name, "persona.name must be non-empty"
    assert isinstance(persona.age, int) and 18 <= persona.age <= 100, (
        f"persona.age must be an integer 18-100, got {persona.age!r}"
    )
    assert len(persona.background) >= 10, (
        f"persona.background too short ({len(persona.background)} chars)"
    )
    assert len(persona.personality_traits) >= 1, (
        "persona.personality_traits must have at least one trait"
    )
