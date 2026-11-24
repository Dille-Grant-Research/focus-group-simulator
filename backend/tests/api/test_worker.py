"""Tests for the async job worker (api/worker.py)."""

from __future__ import annotations

import asyncio
import json

import pytest
from httpx import AsyncClient

from api.config import get_settings
from api.database import get_db
from api.worker import job_worker
from focus_group.core.session import SessionResult
from tests.api.helpers import _MINIMAL_JOB_CONFIG, _TEST_USER_A
from tests.mocks.mock_provider import MockLLMProvider

_FAST_POLL_SECONDS = 0.01
_WAIT_FOR_STATUS_SECONDS = 5.0
_INVALID_CONFIG_JSON = "{}"  # SessionConfig requires demographics and questions


async def _insert_job(job_id: str, config_json: str) -> None:
    from datetime import datetime, timedelta, timezone

    from api.database import create_job, upsert_user

    expires_at = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    async with get_db(get_settings().DB_PATH) as db:
        await upsert_user(db, _TEST_USER_A)
        await create_job(
            db,
            job_id=job_id,
            user_id=_TEST_USER_A,
            config_json=config_json,
            idempotency_key=None,
            expires_at=expires_at,
        )


async def _run_worker_until(job_id: str, terminal: set[str]) -> dict:
    """Run the worker until the job reaches a terminal status; return its row."""
    stop = asyncio.Event()
    task = asyncio.create_task(job_worker(get_settings(), stop))
    try:
        async with asyncio.timeout(_WAIT_FOR_STATUS_SECONDS):
            while True:
                async with get_db(get_settings().DB_PATH) as db:
                    cursor = await db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
                    row = dict(await cursor.fetchone())
                if row["status"] in terminal:
                    return row
                await asyncio.sleep(_FAST_POLL_SECONDS)
    finally:
        stop.set()
        await task


@pytest.fixture(autouse=True)
def fast_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("api.worker._POLL_INTERVAL_SECONDS", _FAST_POLL_SECONDS)


async def test_failed_job_records_completed_at(client: AsyncClient) -> None:
    """A failed job gets completed_at so /metrics can count it in jobs_failed_24h."""
    await _insert_job("job-bad-config", _INVALID_CONFIG_JSON)

    row = await _run_worker_until("job-bad-config", {"failed", "completed"})

    assert row["status"] == "failed"
    assert row["completed_at"] is not None

    data = (await client.get("/metrics")).json()
    assert data["jobs_failed_24h"] == 1


# ---------------------------------------------------------------------------
# Lifecycle with a fake session (no LLM)
# ---------------------------------------------------------------------------

_VALID_CONFIG_JSON = json.dumps(_MINIMAL_JOB_CONFIG["config"])
_SHORT_TIMEOUT_SECONDS = "1"
_HANG_SECONDS = 30
_TERMINAL = {"failed", "completed"}


class _TrackingProvider(MockLLMProvider):
    closed = False

    async def aclose(self) -> None:
        self.closed = True


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch, *, errors: list[str] | None = None, hang: bool = False
) -> _TrackingProvider:
    """Replace the session and provider the worker builds. Returns the provider."""
    provider = _TrackingProvider()

    class _FakeSession:
        def __init__(self, provider, config) -> None:  # noqa: ANN001
            pass

        async def run(self) -> SessionResult:
            if hang:
                await asyncio.sleep(_HANG_SECONDS)
            return SessionResult(errors=list(errors or []))

    monkeypatch.setattr("api.worker.FocusGroupSession", _FakeSession)
    monkeypatch.setattr("api.worker.build_llm_provider", lambda *_args, **_kw: provider)
    return provider


async def test_successful_job_completes(client: AsyncClient, monkeypatch) -> None:
    provider = _install_fakes(monkeypatch)
    await _insert_job("job-ok", _VALID_CONFIG_JSON)

    row = await _run_worker_until("job-ok", _TERMINAL)

    assert row["status"] == "completed"
    assert row["result_json"] is not None
    assert row["error_message"] is None
    assert provider.closed is True


async def test_session_errors_mark_job_failed(client: AsyncClient, monkeypatch) -> None:
    errors = ["Question 1 failed: boom", "Question 2 failed: bang"]
    provider = _install_fakes(monkeypatch, errors=errors)
    await _insert_job("job-errors", _VALID_CONFIG_JSON)

    row = await _run_worker_until("job-errors", _TERMINAL)

    assert row["status"] == "failed"
    assert row["completed_at"] is not None
    for error in errors:
        assert error in row["error_message"]
    # The partial transcript is kept, not thrown away with the failure
    assert json.loads(row["result_json"])["errors"] == errors
    assert provider.closed is True


async def test_job_timeout_marks_job_failed(client: AsyncClient, monkeypatch) -> None:
    monkeypatch.setenv("JOB_TIMEOUT_SECONDS", _SHORT_TIMEOUT_SECONDS)
    get_settings.cache_clear()
    provider = _install_fakes(monkeypatch, hang=True)
    await _insert_job("job-hang", _VALID_CONFIG_JSON)

    row = await _run_worker_until("job-hang", _TERMINAL)

    assert row["status"] == "failed"
    assert "timed out" in row["error_message"]
    assert provider.closed is True


# ---------------------------------------------------------------------------
# Submitted sampling settings are honoured and recorded
# ---------------------------------------------------------------------------

_SERVER_MAX_TOKENS = "1024"


def _capture_sampling(monkeypatch: pytest.MonkeyPatch) -> list:
    """Install fakes and record the sampling params the worker builds the provider with."""
    _install_fakes(monkeypatch)
    captured: list = []
    provider = _TrackingProvider()

    def _build(settings, sampling=None):  # noqa: ANN001
        captured.append(sampling)
        return provider

    monkeypatch.setattr("api.worker.build_llm_provider", _build)
    return captured


def _config_with_model(model: dict) -> str:
    return json.dumps({**_MINIMAL_JOB_CONFIG["config"], "model": model})


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        pytest.param(
            {"temperature": 0.2, "top_p": 0.5, "seed": 7},
            {"temperature": 0.2, "top_p": 0.5, "seed": 7},
            id="submitted",
        ),
        pytest.param({"max_tokens": 9000}, {"max_tokens": 1024}, id="max-tokens-capped"),
        pytest.param({"max_tokens": 300}, {"max_tokens": 300}, id="max-tokens-under-cap"),
    ],
)
async def test_submitted_sampling_is_honoured(
    client: AsyncClient, monkeypatch, model: dict, expected: dict
) -> None:
    monkeypatch.setenv("LLM_MAX_TOKENS", _SERVER_MAX_TOKENS)
    get_settings.cache_clear()
    captured = _capture_sampling(monkeypatch)
    await _insert_job("job-sampling", _config_with_model(model))

    row = await _run_worker_until("job-sampling", _TERMINAL)

    assert row["status"] == "completed"
    sampling = captured[0]
    for key, value in expected.items():
        assert getattr(sampling, key) == value


async def test_server_chooses_and_records_seed(client: AsyncClient, monkeypatch) -> None:
    captured = _capture_sampling(monkeypatch)
    await _insert_job("job-seed", _VALID_CONFIG_JSON)

    row = await _run_worker_until("job-seed", _TERMINAL)

    seed = captured[0].seed
    assert isinstance(seed, int)
    metadata = json.loads(row["run_metadata_json"])
    assert metadata["params"]["seed"] == seed
    assert metadata["model"] == get_settings().LLM_MODEL
    assert metadata["duration_seconds"] >= 0


async def test_failed_job_still_records_metadata(client: AsyncClient, monkeypatch) -> None:
    _install_fakes(monkeypatch, errors=["Question 1 failed: boom"])
    await _insert_job("job-fail-meta", _VALID_CONFIG_JSON)

    row = await _run_worker_until("job-fail-meta", _TERMINAL)

    assert row["status"] == "failed"
    assert json.loads(row["run_metadata_json"])["params"]["seed"] is not None


_SERVER_MAX_TOKENS_HIGH = "4096"


@pytest.mark.parametrize(
    "model",
    [
        pytest.param({"temperature": 0.2}, id="ollama-shaped"),
        pytest.param(
            {"provider": "openai", "model_name": "x", "temperature": 0.2}, id="cloud-shaped"
        ),
    ],
)
async def test_unsubmitted_fields_use_server_defaults_via_route(
    client: AsyncClient, monkeypatch, model: dict
) -> None:
    """Submitting through the real route must not turn model defaults into 'submitted' values."""
    from tests.api.helpers import session_headers

    monkeypatch.setenv("LLM_MAX_TOKENS", _SERVER_MAX_TOKENS_HIGH)
    get_settings.cache_clear()
    captured = _capture_sampling(monkeypatch)
    body = {"config": {**_MINIMAL_JOB_CONFIG["config"], "model": model}}
    resp = await client.post("/api/v1/jobs", json=body, headers=session_headers(_TEST_USER_A))
    job_id = resp.json()["job_id"]

    row = await _run_worker_until(job_id, _TERMINAL)

    assert row["status"] == "completed"
    assert captured[0].temperature == 0.2
    assert captured[0].max_tokens == int(_SERVER_MAX_TOKENS_HIGH)
