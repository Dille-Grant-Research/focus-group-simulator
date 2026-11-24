"""Tests for Settings validation."""

import pytest
from pydantic import ValidationError

from api.config import Settings

LOCAL_REDIRECT_URI = "http://localhost:8000/api/v1/auth/callback"
PRODUCTION_REDIRECT_URI = "https://focusgroup.mnstate.edu/api/v1/auth/callback"


@pytest.mark.parametrize(
    "redirect_uri, allowed",
    [
        ("http://localhost:8000/api/v1/auth/callback", True),
        ("https://localhost:8000/auth/callback", True),
        ("http://127.0.0.1:8000/api/v1/auth/callback", True),
        ("http://[::1]:8000/api/v1/auth/callback", True),
        ("https://focusgroup.mnstate.edu/api/v1/auth/callback", False),
        ("https://localhost.attacker.example/api/v1/auth/callback", False),
    ],
)
def test_dev_bypass_only_allowed_on_localhost(
    monkeypatch: pytest.MonkeyPatch, redirect_uri: str, allowed: bool
) -> None:
    monkeypatch.setenv("DEV_AUTH_BYPASS", "true")
    monkeypatch.setenv("ENTRA_REDIRECT_URI", redirect_uri)
    if allowed:
        assert Settings(_env_file=None).DEV_AUTH_BYPASS is True
    else:
        with pytest.raises(ValidationError, match="DEV_AUTH_BYPASS"):
            Settings(_env_file=None)


def test_bypass_off_accepts_any_redirect_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEV_AUTH_BYPASS", "false")
    monkeypatch.setenv("ENTRA_REDIRECT_URI", PRODUCTION_REDIRECT_URI)
    assert Settings(_env_file=None).DEV_AUTH_BYPASS is False
