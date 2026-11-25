"""Session parameter auto-generation endpoint.

Accepts a research topic and returns LLM-generated demographics, questions,
session name, and purpose. Uses the same inference provider as the job worker.
Auth required — the endpoint consumes inference resources.
"""

from __future__ import annotations

import json
import logging
import re

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api.config import Settings, build_llm_provider, get_settings
from focus_group.llm.base import BaseLLMProvider
from focus_group.llm.ollama import OllamaConnectionError
from focus_group.llm.openai_compat import OpenAIConnectionError
from focus_group.models.config import (
    MAX_DEMOGRAPHIC_CHARS,
    MAX_PURPOSE_CHARS,
    MAX_QUESTION_CHARS,
    MAX_SESSION_NAME_CHARS,
)

logger = logging.getLogger(__name__)

_LLM_CONNECTION_ERRORS = (OllamaConnectionError, OpenAIConnectionError)

router = APIRouter(prefix="/generate", tags=["generate"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    research_topic: str = Field(min_length=1, max_length=500)
    num_participants: int = Field(default=4, ge=1, le=15)
    num_questions: int = Field(default=3, ge=1, le=20)


class GenerateResponse(BaseModel):
    session_name: str
    purpose: str
    demographics: list[str]
    questions: list[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_fences(raw: str) -> str:
    """Remove markdown code fences from LLM output."""
    return re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()


def _parse_json_array(raw: str) -> list[str]:
    text = _strip_fences(raw)
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array")
    return [str(item) for item in parsed]


def _parse_json_object(raw: str) -> dict:
    text = _strip_fences(raw)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


_DEMOGRAPHICS_SYSTEM = (
    "You are a focus group research designer. "
    "You generate realistic, diverse participant profiles. "
    "Return only valid JSON — no explanation, no markdown prose."
)

_QUESTIONS_SYSTEM = (
    "You are a qualitative research expert specializing in focus group facilitation. "
    "You write neutral, open-ended questions that invite authentic diverse perspectives. "
    "Return only valid JSON — no explanation, no markdown prose."
)


def _demographics_prompt(topic: str, n: int) -> str:
    return f"""Research topic: "{topic}"

Generate exactly {n} demographically diverse focus group participants.
Return a JSON object with these keys:
- "session_name": a concise title for this session (max 60 chars)
- "purpose": a 1–2 sentence introductory statement for participants (neutral, factual)
- "demographics": a JSON array of exactly {n} strings, each in this exact format:
  "Gender, age N, Location, Occupation[, optional stance perspective][, optional register communicator]"

  Where:
  - Gender: Female | Male | Non-binary | Prefer not to say
  - age N: e.g. "age 34"
  - Location: Twin Cities Metro | Fargo-Moorhead Area | Suburban MN | Rural MN | Out of State
  - Occupation: free-text (e.g. "registered nurse", "retired teacher")
  - stance (optional, only if non-default): progressive perspective | conservative perspective | skeptical perspective | moderate perspective
  - register (optional, only if non-default): formal communicator | casual communicator | academic communicator

Ensure diversity across age, gender, location, and occupation. Return ONLY valid JSON."""


def _questions_prompt(topic: str, n: int) -> str:
    return f"""Research topic: "{topic}"

Generate exactly {n} open-ended focus group discussion questions.
Requirements:
- Neutral and non-leading
- Each invites genuine disagreement and diverse viewpoints
- Appropriate for qualitative research, not yes/no answerable
- Progressively deeper (start broad, end specific)

Return ONLY a JSON array of {n} question strings. No explanation."""


# ---------------------------------------------------------------------------
# Route handler
# ---------------------------------------------------------------------------


@router.post("", response_model=GenerateResponse)
async def generate_session_params(
    body: GenerateRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> GenerateResponse:
    provider = build_llm_provider(settings)
    try:
        return await _generate_params(provider, body)
    finally:
        await provider.aclose()


async def _generate_params(provider: BaseLLMProvider, body: GenerateRequest) -> GenerateResponse:
    """Run the two generation calls and parse them into a GenerateResponse."""
    # ── Call 1: demographics + session meta ──────────────────────────────────
    demo_prompt = _demographics_prompt(body.research_topic, body.num_participants)
    try:
        demo_raw = await provider.generate(demo_prompt, system_prompt=_DEMOGRAPHICS_SYSTEM)
    except _LLM_CONNECTION_ERRORS as exc:
        logger.warning("Inference service unreachable during generate: %s", exc)
        raise HTTPException(
            status_code=503,
            detail={
                "code": "SERVICE_UNAVAILABLE",
                "message": "Inference service is not reachable. Try again in a moment.",
                "details": {},
            },
        )

    try:
        meta = _parse_json_object(demo_raw)
        session_name = str(meta.get("session_name", ""))
        purpose = str(meta.get("purpose", ""))
        demographics_raw = meta.get("demographics", [])
        if not isinstance(demographics_raw, list):
            raise ValueError("demographics must be a list")
        demographics = [str(d) for d in demographics_raw]
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Demographics parse failure: %s | raw=%r", exc, demo_raw[:300])
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": "Could not parse LLM response for participants. Please try again.",
                "details": {},
            },
        )

    # ── Call 2: questions ────────────────────────────────────────────────────
    q_prompt = _questions_prompt(body.research_topic, body.num_questions)
    try:
        q_raw = await provider.generate(q_prompt, system_prompt=_QUESTIONS_SYSTEM)
    except _LLM_CONNECTION_ERRORS as exc:
        logger.warning("Inference service unreachable during generate (questions): %s", exc)
        raise HTTPException(
            status_code=503,
            detail={
                "code": "SERVICE_UNAVAILABLE",
                "message": "Inference service is not reachable. Try again in a moment.",
                "details": {},
            },
        )

    try:
        questions = _parse_json_array(q_raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Questions parse failure: %s | raw=%r", exc, q_raw[:300])
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": "Could not parse LLM response for questions. Please try again.",
                "details": {},
            },
        )

    # Clip to the SessionConfig caps so the UI can always submit what we return.
    return GenerateResponse(
        session_name=session_name[:MAX_SESSION_NAME_CHARS],
        purpose=purpose[:MAX_PURPOSE_CHARS],
        demographics=[d[:MAX_DEMOGRAPHIC_CHARS] for d in demographics],
        questions=[q[:MAX_QUESTION_CHARS] for q in questions],
    )
