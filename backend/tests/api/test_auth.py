"""Step 12: Authentication route tests.

Covers: login redirect, callback (success + error cases), logout,
and auth middleware rejection on protected routes.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient

from tests.api.helpers import _TEST_USER_A, cookie_header, make_flow_cookie, make_session_cookie

# ---------------------------------------------------------------------------
# MSAL mock data
# ---------------------------------------------------------------------------

_MOCK_FLOW = {
    "auth_uri": "https://login.microsoftonline.com/test-tenant/oauth2/v2.0/authorize?state=abc",
    "state": "abc",
    "nonce": "test-nonce",
    "scope": ["openid", "profile", "email"],
}

_MOCK_TOKEN_OK = {
    "id_token_claims": {
        "oid": _TEST_USER_A,
        "name": "Test User A",
        "preferred_username": "usera@test.msum.edu",
    }
}

_MOCK_TOKEN_ERROR = {
    "error": "invalid_grant",
    "error_description": "The authorization code is invalid or has expired.",
}


@pytest.fixture
def mock_msal(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace ConfidentialClientApplication with a controllable mock."""
    mock_app = MagicMock()
    mock_app.initiate_auth_code_flow.return_value = _MOCK_FLOW
    mock_app.acquire_token_by_auth_code_flow.return_value = _MOCK_TOKEN_OK
    monkeypatch.setattr(
        "api.auth.ConfidentialClientApplication",
        MagicMock(return_value=mock_app),
    )
    return mock_app


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


async def test_login_redirects_to_entra(client: AsyncClient, mock_msal: MagicMock) -> None:
    resp = await client.get("/api/v1/auth/login", follow_redirects=False)

    assert resp.status_code == 302
    location = resp.headers["location"]
    assert location.startswith("https://login.microsoftonline.com/")
    assert "oauth_flow" in resp.cookies


# ---------------------------------------------------------------------------
# Callback — success
# ---------------------------------------------------------------------------


async def test_callback_success_sets_session_cookie(
    client: AsyncClient, mock_msal: MagicMock
) -> None:
    flow_cookie = make_flow_cookie(_MOCK_FLOW)

    resp = await client.get(
        "/api/v1/auth/callback",
        params={"code": "test-auth-code", "state": "abc"},
        headers=cookie_header({"oauth_flow": flow_cookie}),
        follow_redirects=False,
    )

    assert resp.status_code == 302
    assert resp.headers["location"] == "http://localhost"
    assert "session" in resp.cookies


# ---------------------------------------------------------------------------
# Callback — error cases
# ---------------------------------------------------------------------------


async def test_callback_missing_flow_cookie_redirects_with_error(
    client: AsyncClient, mock_msal: MagicMock
) -> None:
    resp = await client.get(
        "/api/v1/auth/callback",
        params={"code": "test-code"},
        follow_redirects=False,
    )

    assert resp.status_code == 302
    assert "auth_error" in resp.headers["location"]


async def test_callback_msal_error_redirects_with_error(
    client: AsyncClient, mock_msal: MagicMock
) -> None:
    mock_msal.acquire_token_by_auth_code_flow.return_value = _MOCK_TOKEN_ERROR
    flow_cookie = make_flow_cookie(_MOCK_FLOW)

    resp = await client.get(
        "/api/v1/auth/callback",
        params={"code": "bad-code", "state": "abc"},
        headers=cookie_header({"oauth_flow": flow_cookie}),
        follow_redirects=False,
    )

    assert resp.status_code == 302
    assert "auth_error" in resp.headers["location"]


async def test_callback_tampered_flow_cookie_redirects_with_error(
    client: AsyncClient, mock_msal: MagicMock
) -> None:
    resp = await client.get(
        "/api/v1/auth/callback",
        params={"code": "test-code", "state": "abc"},
        headers=cookie_header({"oauth_flow": "this.is.not.a.valid.signed.cookie"}),
        follow_redirects=False,
    )

    assert resp.status_code == 302
    assert "auth_error" in resp.headers["location"]


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------


async def test_logout_redirects_to_frontend(client: AsyncClient) -> None:
    resp = await client.get(
        "/api/v1/auth/logout",
        headers=cookie_header({"session": make_session_cookie(_TEST_USER_A)}),
        follow_redirects=False,
    )

    assert resp.status_code == 302
    assert resp.headers["location"] == "http://localhost"


# ---------------------------------------------------------------------------
# Auth middleware — protected routes
# ---------------------------------------------------------------------------


async def test_protected_route_no_session_returns_401(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/jobs")

    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "UNAUTHENTICATED"


async def test_protected_route_tampered_session_returns_401(client: AsyncClient) -> None:
    resp = await client.get(
        "/api/v1/jobs",
        headers=cookie_header({"session": "this.is.definitely.not.valid"}),
    )

    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "UNAUTHENTICATED"


async def test_health_is_exempt_from_auth(client: AsyncClient, monkeypatch) -> None:
    """Sanity check: /health does not require a session cookie."""
    import httpx
    from unittest.mock import AsyncMock, MagicMock as MM

    mock_resp = MM()
    mock_resp.status_code = 200
    mock_cli = AsyncMock()
    mock_cli.get = AsyncMock(return_value=mock_resp)
    mock_ctx = MM()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_cli)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(httpx, "AsyncClient", MM(return_value=mock_ctx))

    resp = await client.get("/health")
    assert resp.status_code in (200, 503)  # not 401
    assert "error" not in resp.json() or resp.json().get("error", {}).get("code") != "UNAUTHENTICATED"
