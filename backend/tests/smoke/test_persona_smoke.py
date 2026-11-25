"""Live AI smoke test — persona generation.

Uses the same settings and provider as the job worker: LLM_PROVIDER, LLM_BASE_URL,
LLM_MODEL, LLM_REASONING_EFFORT etc. from backend/.env (e.g. cruncher via the SSH
tunnel at http://localhost:1234). Settings reads .env itself, so nothing leaks into
os.environ for the rest of the test run.

Gate: --run-ollama flag (never fires in CI or make test-all).

Run with:
    make test-ai-long
    # or directly:
    pytest tests/smoke/ --run-ollama -v
"""

from __future__ import annotations

import pytest

from api.config import Settings, build_llm_provider
from focus_group.persona.generator import PersonaGenerator

_SMOKE_DEMOGRAPHIC = "Female, age 38, rural Minnesota, registered nurse"


@pytest.mark.requires_ollama
async def test_first_persona_generated() -> None:
    """Generate exactly one persona and assert required fields are present.

    Exits as soon as the first persona is confirmed — does not run
    the full simulation.
    """
    settings = Settings()
    provider = build_llm_provider(settings)

    if not await provider.is_available():
        pytest.skip(f"{settings.LLM_MODEL} not reachable at {settings.llm_base_url}")

    try:
        persona = await PersonaGenerator(provider).generate_persona(_SMOKE_DEMOGRAPHIC)
    finally:
        await provider.aclose()

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
