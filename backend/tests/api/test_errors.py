"""Tests for information exposure: API docs gating and unhandled-error messages."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport

from api.config import get_settings
from api.main import create_app
from tests.api.helpers import _TEST_USER_A, session_headers

HTTP_OK = 200
HTTP_NOT_FOUND = 404
HTTP_INTERNAL_ERROR = 500
GENERIC_MESSAGE = "An unexpected error occurred."
SECRET_DETAIL = "secret-internal-detail"


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
@pytest.mark.parametrize("enabled, expected", [("false", HTTP_NOT_FOUND), ("true", HTTP_OK)])
async def test_api_docs_gated_by_setting(
    monkeypatch: pytest.MonkeyPatch,
    isolate_settings: str,
    path: str,
    enabled: str,
    expected: int,
) -> None:
    monkeypatch.setenv("API_DOCS_ENABLED", enabled)
    get_settings.cache_clear()
    docs_app = create_app()

    async with httpx.AsyncClient(
        transport=ASGITransport(app=docs_app), base_url="http://test"
    ) as http:
        resp = await http.get(path)

    assert resp.status_code == expected


async def test_unhandled_error_hides_exception_text(app: FastAPI) -> None:
    async def boom() -> None:
        raise RuntimeError(SECRET_DETAIL)

    app.add_api_route("/api/v1/boom", boom, methods=["GET"])

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as http:
        resp = await http.get("/api/v1/boom", headers=session_headers(_TEST_USER_A))

    assert resp.status_code == HTTP_INTERNAL_ERROR
    assert resp.json()["error"]["message"] == GENERIC_MESSAGE
    assert SECRET_DETAIL not in resp.text
