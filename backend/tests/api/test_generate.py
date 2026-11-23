"""Tests for POST /api/v1/generate — session parameter auto-generation."""

from __future__ import annotations

import json

import pytest
from httpx import AsyncClient

from focus_group.llm.ollama import OllamaConnectionError
from focus_group.llm.openai_compat import OpenAIConnectionError
from tests.api.helpers import _TEST_USER_A, session_cookies
from tests.mocks.mock_provider import MockLLMProvider, SequentialMockProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_TOPIC = "telehealth adoption in rural Minnesota"

_DEMOGRAPHICS_JSON = json.dumps({
    "session_name": "Telehealth Study",
    "purpose": "We are exploring experiences with telehealth in rural areas.",
    "demographics": [
        "Female, age 42, Rural MN, registered nurse",
        "Male, age 58, Fargo-Moorhead Area, retired farmer, skeptical perspective",
        "Non-binary, age 31, Twin Cities Metro, software developer, casual communicator",
        "Female, age 67, Suburban MN, retired teacher",
    ],
})

_QUESTIONS_JSON = json.dumps([
    "What has been your experience with telehealth services so far?",
    "What barriers, if any, have prevented you from using telehealth?",
    "How do you think telehealth could better serve rural communities?",
])


def _make_provider(monkeypatch, responses: list[str]) -> None:
    """Patch build_llm_provider in the generate route to return a sequential mock."""
    provider = SequentialMockProvider(responses=responses, loop=False)
    monkeypatch.setattr("api.routes.generate.build_llm_provider", lambda _settings: provider)


def _make_error_provider(monkeypatch, error_type: type[Exception]) -> None:
    """Patch build_llm_provider to raise the given connection error on generate."""
    class _ErrorProvider(MockLLMProvider):
        async def generate(self, prompt: str, system_prompt=None) -> str:  # type: ignore[override]
            raise error_type("test: connection refused")

    monkeypatch.setattr(
        "api.routes.generate.build_llm_provider", lambda _settings: _ErrorProvider()
    )


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


async def test_generate_success(client: AsyncClient, monkeypatch) -> None:
    _make_provider(monkeypatch, [_DEMOGRAPHICS_JSON, _QUESTIONS_JSON])

    resp = await client.post(
        "/api/v1/generate",
        json={"research_topic": _VALID_TOPIC},
        cookies=session_cookies(_TEST_USER_A),
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["session_name"] == "Telehealth Study"
    assert "rural" in data["purpose"].lower()
    assert len(data["demographics"]) == 4
    assert len(data["questions"]) == 3
    assert all(isinstance(d, str) for d in data["demographics"])
    assert all(isinstance(q, str) for q in data["questions"])


async def test_generate_custom_counts(client: AsyncClient, monkeypatch) -> None:
    two_demo = json.dumps({
        "session_name": "S",
        "purpose": "P",
        "demographics": ["Female, age 30, Rural MN, nurse", "Male, age 50, Suburban MN, teacher"],
    })
    two_q = json.dumps(["Q1?", "Q2?"])
    _make_provider(monkeypatch, [two_demo, two_q])

    resp = await client.post(
        "/api/v1/generate",
        json={"research_topic": _VALID_TOPIC, "num_participants": 2, "num_questions": 2},
        cookies=session_cookies(_TEST_USER_A),
    )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["demographics"]) == 2
    assert len(data["questions"]) == 2


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


async def test_generate_requires_auth(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/generate",
        json={"research_topic": _VALID_TOPIC},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


async def test_generate_missing_topic_returns_422(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/generate",
        json={"research_topic": ""},
        cookies=session_cookies(_TEST_USER_A),
    )
    assert resp.status_code == 422


async def test_generate_topic_too_long_returns_422(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/generate",
        json={"research_topic": "x" * 501},
        cookies=session_cookies(_TEST_USER_A),
    )
    assert resp.status_code == 422


async def test_generate_num_participants_zero_returns_422(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/generate",
        json={"research_topic": _VALID_TOPIC, "num_participants": 0},
        cookies=session_cookies(_TEST_USER_A),
    )
    assert resp.status_code == 422


async def test_generate_num_participants_too_many_returns_422(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/generate",
        json={"research_topic": _VALID_TOPIC, "num_participants": 16},
        cookies=session_cookies(_TEST_USER_A),
    )
    assert resp.status_code == 422


async def test_generate_num_questions_zero_returns_422(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/generate",
        json={"research_topic": _VALID_TOPIC, "num_questions": 0},
        cookies=session_cookies(_TEST_USER_A),
    )
    assert resp.status_code == 422


async def test_generate_num_questions_too_many_returns_422(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/generate",
        json={"research_topic": _VALID_TOPIC, "num_questions": 21},
        cookies=session_cookies(_TEST_USER_A),
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Inference service unavailable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("error_type", [OllamaConnectionError, OpenAIConnectionError])
async def test_generate_llm_unreachable_returns_503(
    client: AsyncClient, monkeypatch, error_type: type[Exception]
) -> None:
    _make_error_provider(monkeypatch, error_type)

    resp = await client.post(
        "/api/v1/generate",
        json={"research_topic": _VALID_TOPIC},
        cookies=session_cookies(_TEST_USER_A),
    )

    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "SERVICE_UNAVAILABLE"


# ---------------------------------------------------------------------------
# Malformed LLM output
# ---------------------------------------------------------------------------


async def test_generate_bad_demographics_json_returns_500(
    client: AsyncClient, monkeypatch
) -> None:
    _make_provider(monkeypatch, ["this is not json at all", _QUESTIONS_JSON])

    resp = await client.post(
        "/api/v1/generate",
        json={"research_topic": _VALID_TOPIC},
        cookies=session_cookies(_TEST_USER_A),
    )

    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "INTERNAL_ERROR"


async def test_generate_bad_questions_json_returns_500(
    client: AsyncClient, monkeypatch
) -> None:
    _make_provider(monkeypatch, [_DEMOGRAPHICS_JSON, "not valid json"])

    resp = await client.post(
        "/api/v1/generate",
        json={"research_topic": _VALID_TOPIC},
        cookies=session_cookies(_TEST_USER_A),
    )

    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "INTERNAL_ERROR"
