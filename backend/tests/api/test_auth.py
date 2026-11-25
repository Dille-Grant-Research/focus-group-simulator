"""Step 12: Authentication route tests.

Covers: login redirect, callback (success + error cases), logout,
and auth middleware rejection on protected routes.
"""

from __future__ import annotations

import time
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient

from tests.api.helpers import (
    _TEST_USER_A,
    _TEST_USER_B,
    cookie_header,
    make_flow_cookie,
    make_session_cookie,
    session_headers,
)

_LOGOUT_AGE_SECONDS = 10  # how far in the past the simulated logout happened

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


_SESSION_MAX_AGE_SECONDS = 12 * 3600


async def test_callback_sets_session_max_age(client: AsyncClient, mock_msal: MagicMock) -> None:
    resp = await client.get(
        "/api/v1/auth/callback",
        params={"code": "test-auth-code", "state": "abc"},
        headers=cookie_header({"oauth_flow": make_flow_cookie(_MOCK_FLOW)}),
        follow_redirects=False,
    )

    session_header = next(
        h for h in resp.headers.get_list("set-cookie") if h.startswith("session=")
    )
    assert f"Max-Age={_SESSION_MAX_AGE_SECONDS}" in session_header


@pytest.mark.parametrize(
    ("issued_ago", "expected"),
    [
        (timedelta(hours=11), 200),
        (timedelta(hours=13), 401),
    ],
)
async def test_session_lifetime_enforced(
    client: AsyncClient, issued_ago: timedelta, expected: int
) -> None:
    headers = cookie_header({"session": make_session_cookie(_TEST_USER_A, issued_ago)})

    assert (await client.get("/api/v1/jobs", headers=headers)).status_code == expected


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------


async def test_logout_revokes_existing_session(client: AsyncClient) -> None:
    headers = session_headers(_TEST_USER_A)
    assert (await client.get("/api/v1/jobs", headers=headers)).status_code == 200

    resp = await client.post("/api/v1/auth/logout", headers=headers, follow_redirects=False)

    assert resp.status_code == 303
    assert resp.headers["location"] == "http://localhost"
    assert (await client.get("/api/v1/jobs", headers=headers)).status_code == 401


async def test_session_issued_after_logout_is_accepted(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # itsdangerous rejects future-dated cookies, so move the logout into the past instead.
    logout_time = time.time() - _LOGOUT_AGE_SECONDS
    monkeypatch.setattr("api.routes.auth.time", SimpleNamespace(time=lambda: logout_time))
    await client.post("/api/v1/auth/logout", headers=session_headers(_TEST_USER_A))

    resp = await client.get("/api/v1/jobs", headers=session_headers(_TEST_USER_A))
    assert resp.status_code == 200

    older = cookie_header(
        {"session": make_session_cookie(_TEST_USER_A, timedelta(seconds=2 * _LOGOUT_AGE_SECONDS))}
    )
    assert (await client.get("/api/v1/jobs", headers=older)).status_code == 401


async def test_logout_does_not_revoke_other_users(client: AsyncClient) -> None:
    await client.post("/api/v1/auth/logout", headers=session_headers(_TEST_USER_A))

    resp = await client.get("/api/v1/jobs", headers=session_headers(_TEST_USER_B))

    assert resp.status_code == 200


@pytest.mark.parametrize(
    "cookie",
    [None, "garbage", make_session_cookie(_TEST_USER_A, timedelta(hours=13))],
    ids=["missing", "garbage", "expired"],
)
async def test_logout_with_bad_cookie_still_redirects(
    client: AsyncClient, cookie: str | None
) -> None:
    headers = cookie_header({"session": cookie}) if cookie else {}

    resp = await client.post("/api/v1/auth/logout", headers=headers, follow_redirects=False)

    assert resp.status_code == 303
    assert resp.headers["location"] == "http://localhost"
    assert 'session=""' in resp.headers["set-cookie"]


async def test_logout_clears_cookie_even_when_revocation_fails(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("db unavailable")

    monkeypatch.setattr("api.routes.auth.revoke_user_sessions", boom)

    resp = await client.post(
        "/api/v1/auth/logout", headers=session_headers(_TEST_USER_A), follow_redirects=False
    )

    assert resp.status_code == 303
    assert resp.headers["location"] == "http://localhost"
    assert 'session=""' in resp.headers["set-cookie"]


async def test_revoke_user_sessions_without_users_row_still_records(tmp_path) -> None:
    from api.database import get_db, get_sessions_valid_after, init_db, revoke_user_sessions

    db_path = str(tmp_path / "revoke.db")
    await init_db(db_path)
    revoked_at = 1_700_000_000

    async with get_db(db_path) as db:
        await revoke_user_sessions(db, "ghost-user", revoked_at)
        assert await get_sessions_valid_after(db, "ghost-user") == revoked_at


async def test_get_logout_not_allowed(client: AsyncClient) -> None:
    assert (await client.get("/api/v1/auth/logout")).status_code == 405


async def test_init_db_adds_sessions_valid_after_column_to_old_schema(tmp_path) -> None:
    import aiosqlite

    from api.database import _CREATE_USERS, init_db

    old_schema = _CREATE_USERS.replace(",\n    sessions_valid_after INTEGER", "")
    assert "sessions_valid_after" not in old_schema

    db_path = str(tmp_path / "old.db")
    async with aiosqlite.connect(db_path) as db:
        await db.execute(old_schema)
        await db.execute("INSERT INTO users VALUES ('u1', 'a', 'b')")
        await db.commit()

    await init_db(db_path)
    await init_db(db_path)  # idempotent

    async with aiosqlite.connect(db_path) as db:
        info = await (await db.execute("PRAGMA table_info(users)")).fetchall()
        columns = {row[1] for row in info}
        row = await (await db.execute("SELECT user_id, sessions_valid_after FROM users")).fetchone()
    assert "sessions_valid_after" in columns
    assert row == ("u1", None)


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
