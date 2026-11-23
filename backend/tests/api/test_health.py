"""Step 14: Health check and metrics endpoint tests.

Both routes are auth-exempt (no session cookie required).
Worker liveness depends on app.state.worker_task, which is set by the
lifespan fixture in conftest — a noop coroutine that stays alive until
stop_event is signalled at teardown.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from httpx import AsyncClient

from api.config import get_settings
from api.database import get_db, update_job
from tests.api.helpers import _MINIMAL_JOB_CONFIG, _TEST_USER_A, session_cookies


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


async def test_health_all_components_ok(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_llm_reachable(monkeypatch)

    resp = await client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["components"]["database"] == "ok"
    assert data["components"]["worker"] == "alive"
    assert data["components"]["llm"] == "reachable"
    assert "timestamp" in data


async def test_health_degraded_when_llm_unreachable(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_llm_down(monkeypatch)

    resp = await client.get("/health")

    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["components"]["llm"] == "unreachable"


async def test_health_degraded_when_worker_dead(
    app, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_llm_reachable(monkeypatch)

    # Replace the running worker task with a completed future to simulate crash
    loop = asyncio.get_running_loop()
    done_future: asyncio.Future = loop.create_future()
    done_future.set_result(None)
    app.state.worker_task = done_future

    resp = await client.get("/health")

    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["components"]["worker"] == "dead"


async def test_health_returns_503_when_database_unreachable(
    app, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_llm_reachable(monkeypatch)

    # Point settings at a non-existent path to simulate DB failure
    from api.config import Settings

    bad_settings = Settings(
        LLM_PROVIDER="ollama",
        LLM_BASE_URL="http://localhost:11434",
        LLM_MODEL="test-model",
        ENTRA_TENANT_ID="test-tenant-id",
        ENTRA_CLIENT_ID="test-client-id",
        ENTRA_CLIENT_SECRET="test-secret",
        ENTRA_REDIRECT_URI="http://localhost/callback",
        FRONTEND_URL="http://localhost",
        SESSION_SECRET_KEY="test-key",
        DB_PATH="/nonexistent/path/that/cannot/be/created/focus.db",
    )
    monkeypatch.setattr("api.routes.health.get_settings", lambda: bad_settings)

    resp = await client.get("/health")

    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["components"]["database"] == "error"


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


async def test_metrics_returns_expected_fields(client: AsyncClient) -> None:
    resp = await client.get("/metrics")

    assert resp.status_code == 200
    data = resp.json()
    assert "queue_depth" in data
    assert "jobs_completed_24h" in data
    assert "jobs_failed_24h" in data
    assert "avg_duration_seconds" in data
    assert "worker_status" in data
    assert "current_job_id" in data


async def test_metrics_queue_depth_reflects_queued_jobs(client: AsyncClient) -> None:
    await client.post(
        "/api/v1/jobs", json=_MINIMAL_JOB_CONFIG, cookies=session_cookies(_TEST_USER_A)
    )

    resp = await client.get("/metrics")

    assert resp.status_code == 200
    assert resp.json()["queue_depth"] == 1


async def test_metrics_worker_status_idle_when_no_running_job(client: AsyncClient) -> None:
    resp = await client.get("/metrics")

    data = resp.json()
    assert data["worker_status"] == "idle"
    assert data["current_job_id"] is None


async def test_metrics_counts_completed_jobs(client: AsyncClient) -> None:
    import json
    from datetime import datetime, timezone

    submit = await client.post(
        "/api/v1/jobs", json=_MINIMAL_JOB_CONFIG, cookies=session_cookies(_TEST_USER_A)
    )
    job_id = submit.json()["job_id"]

    now = datetime.now(timezone.utc).isoformat()
    settings = get_settings()
    async with get_db(settings.DB_PATH) as db:
        await update_job(
            db,
            job_id,
            status="completed",
            result_json=json.dumps({"transcript": []}),
            started_at=now,
            completed_at=now,
        )

    resp = await client.get("/metrics")

    data = resp.json()
    assert data["jobs_completed_24h"] == 1
    assert data["queue_depth"] == 0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _patch_llm_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_cli = AsyncMock()
    mock_cli.get = AsyncMock(return_value=mock_resp)
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_cli)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_ctx))


def _patch_llm_down(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_cli = AsyncMock()
    mock_cli.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_cli)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_ctx))
