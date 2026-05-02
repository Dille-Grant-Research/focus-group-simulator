"""Shared constants and pure helpers for API layer tests.

Imported by both conftest.py and test modules. Kept separate from conftest.py
so test files can import these symbols directly (conftest.py is not importable
as a regular Python module in all pytest configurations).
"""

from __future__ import annotations

_TEST_SECRET = "test-session-secret-key-abc123xyz"
_TEST_USER_A = "oid-test-user-aaaa-0001"
_TEST_USER_B = "oid-test-user-bbbb-0002"

_MINIMAL_JOB_CONFIG: dict = {
    "config": {
        "demographics": ["Female, 30-40, professional"],
        "questions": ["What do you think about AI?"],
        "validate_personas": False,
    }
}


def make_session_cookie(user_id: str) -> str:
    """Return a signed session cookie value for the given user."""
    from itsdangerous import URLSafeTimedSerializer

    return URLSafeTimedSerializer(_TEST_SECRET, salt="session").dumps({"user_id": user_id})


def session_cookies(user_id: str) -> dict[str, str]:
    """Return a cookies dict suitable for authenticated requests."""
    return {"session": make_session_cookie(user_id)}


def make_flow_cookie(flow: dict) -> str:
    """Return a signed oauth_flow cookie value from an MSAL flow dict."""
    from itsdangerous import URLSafeTimedSerializer

    return URLSafeTimedSerializer(_TEST_SECRET, salt="oauth-flow").dumps(flow)
