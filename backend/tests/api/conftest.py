"""Shared fixtures for the API layer test suite.

Module-level env var defaults are set before any api.* imports so that the
module-level `app = create_app()` in api/main.py can succeed even when no
real .env file is present (e.g., in CI).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import AsyncIterator, Generator

# ---------------------------------------------------------------------------
# Bootstrap env vars BEFORE importing api.* (which triggers app = create_app())
# ---------------------------------------------------------------------------
_IMPORT_DEFAULTS = {
    "OLLAMA_HOST": "localhost",
    "OLLAMA_PORT": "11434",
    "OLLAMA_MODEL": "test-model",
    "ENTRA_TENANT_ID": "test-tenant-id", # We won't actully hard code this and cuase an error is not set
    "ENTRA_CLIENT_ID": "test-client-id", # We won't actully hard code this and cuase an error is not set
    "ENTRA_CLIENT_SECRET": "test-client-secret-placeholder",
    "ENTRA_REDIRECT_URI": "http://localhost/api/v1/auth/callback",
    "FRONTEND_URL": "http://localhost",
    "SESSION_SECRET_KEY": "test-session-key-placeholder", # We won't actully hard code this and cuase an error is not set
    "DB_PATH": "/tmp/focus_group_api_test_import.db", # Why not /mnt or actually temp works for now
    "MAX_JOBS_PER_HOUR": "3", # Should be modified based on the VRAM also taking KV cache into account
}
for _k, _v in _IMPORT_DEFAULTS.items():
    os.environ.setdefault(_k, _v)

import pytest
from httpx import ASGITransport, AsyncClient

from api.config import get_settings
from api.database import init_db, startup_recovery
from api.main import create_app, lifespan as _lifespan

from tests.api.helpers import _TEST_SECRET, _TEST_USER_A, _TEST_USER_B


# ---------------------------------------------------------------------------
# Per-test settings isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Generator[str, None, None]:
    """Reset the settings LRU cache and inject test env vars before each test.

    Yields the db_path so dependent fixtures (app) can use it directly.
    """
    db_path = str(tmp_path / "test.db")
    test_envs = {
        "OLLAMA_HOST": "localhost",
        "OLLAMA_PORT": "11434",
        "OLLAMA_MODEL": "test-model",
        "ENTRA_TENANT_ID": "test-tenant-id",
        "ENTRA_CLIENT_ID": "test-client-id",
        "ENTRA_CLIENT_SECRET": "test-client-secret",
        "ENTRA_REDIRECT_URI": "http://localhost/api/v1/auth/callback",
        "FRONTEND_URL": "http://localhost",
        "SESSION_SECRET_KEY": _TEST_SECRET,
        "DB_PATH": db_path,
        "MAX_JOBS_PER_HOUR": "3",
        "LOG_LEVEL": "WARNING",
        "DEV_AUTH_BYPASS": "false",
    }
    for k, v in test_envs.items():
        monkeypatch.setenv(k, v)

    get_settings.cache_clear()
    monkeypatch.setattr("api.auth._msal_app", None)

    yield db_path

    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# App + client fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def app(monkeypatch: pytest.MonkeyPatch, isolate_settings: str):
    """FastAPI app with noop worker and a fresh isolated DB per test."""

    async def _noop_worker(settings, stop_event):
        await stop_event.wait()

    monkeypatch.setattr("api.main.job_worker", _noop_worker)

    db_path = isolate_settings
    await init_db(db_path)
    await startup_recovery(db_path)

    return create_app()


@pytest.fixture
async def client(app) -> AsyncIterator[AsyncClient]:
    """Unauthenticated async client with the app lifespan fully started.

    The lifespan starts the (noop) worker task so app.state.worker_task is
    set — health check tests that verify worker liveness work correctly.
    """
    async with _lifespan(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
