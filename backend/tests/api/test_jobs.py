"""Step 13: Job lifecycle tests.

Covers: submit, idempotency, rate limiting, status (with ownership enforcement),
result retrieval (pending + completed), delete, and list.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from httpx import AsyncClient

from api.config import get_settings
from api.database import delete_expired_jobs, get_db, get_job, update_job
from tests.api.helpers import _MINIMAL_JOB_CONFIG, _TEST_USER_A, _TEST_USER_B, session_headers


def _config(*, session_name: str | None = None) -> dict:
    """Return a minimal job submit body, optionally with a distinct session_name."""
    cfg = dict(_MINIMAL_JOB_CONFIG["config"])
    if session_name is not None:
        cfg["session_name"] = session_name
    return {"config": cfg}


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


async def test_submit_valid_job(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/jobs",
        json=_MINIMAL_JOB_CONFIG,
        headers=session_headers(_TEST_USER_A),
    )

    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"
    assert data["estimated_position"] >= 1


async def test_submit_unauthenticated_returns_401(client: AsyncClient) -> None:
    resp = await client.post("/api/v1/jobs", json=_MINIMAL_JOB_CONFIG)

    assert resp.status_code == 401


async def test_submit_invalid_body_returns_422(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/jobs",
        json={"config": {"demographics": []}},  # demographics min_length=1 violated
        headers=session_headers(_TEST_USER_A),
    )

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


async def test_idempotent_submission_returns_same_job(client: AsyncClient) -> None:
    resp1 = await client.post(
        "/api/v1/jobs", json=_MINIMAL_JOB_CONFIG, headers=session_headers(_TEST_USER_A)
    )
    resp2 = await client.post(
        "/api/v1/jobs", json=_MINIMAL_JOB_CONFIG, headers=session_headers(_TEST_USER_A)
    )

    assert resp1.status_code == 202
    assert resp2.status_code == 202
    assert resp1.json()["job_id"] == resp2.json()["job_id"]


async def test_different_configs_create_separate_jobs(client: AsyncClient) -> None:
    resp1 = await client.post(
        "/api/v1/jobs", json=_config(session_name="session-alpha"), headers=session_headers(_TEST_USER_A)
    )
    resp2 = await client.post(
        "/api/v1/jobs", json=_config(session_name="session-beta"), headers=session_headers(_TEST_USER_A)
    )

    assert resp1.json()["job_id"] != resp2.json()["job_id"]


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


async def test_rate_limit_enforced(client: AsyncClient) -> None:
    for i in range(3):  # MAX_JOBS_PER_HOUR = 3 in test settings
        resp = await client.post(
            "/api/v1/jobs",
            json=_config(session_name=f"rl-test-{i}"),
            headers=session_headers(_TEST_USER_A),
        )
        assert resp.status_code == 202

    resp = await client.post(
        "/api/v1/jobs",
        json=_config(session_name="rl-overflow"),
        headers=session_headers(_TEST_USER_A),
    )

    assert resp.status_code == 429
    error = resp.json()["error"]
    assert error["code"] == "RATE_LIMIT_EXCEEDED"
    assert error["details"]["limit"] == 3


async def test_rate_limit_is_per_user(client: AsyncClient) -> None:
    """Submitting up to the cap as user A should not block user B."""
    for i in range(3):
        await client.post(
            "/api/v1/jobs",
            json=_config(session_name=f"user-a-{i}"),
            headers=session_headers(_TEST_USER_A),
        )

    resp = await client.post(
        "/api/v1/jobs",
        json=_config(session_name="user-b-job"),
        headers=session_headers(_TEST_USER_B),
    )

    assert resp.status_code == 202


# ---------------------------------------------------------------------------
# Time windows
#
# Rows store Python ISO timestamps ("2026-11-23T01:00:00+00:00"). Comparing them
# as text against SQLite's datetime('now') ("2026-11-23 03:00:00") treats any
# time earlier the same UTC day as "inside" the window, because 'T' > ' '.
# ---------------------------------------------------------------------------

_OUTSIDE_RATE_WINDOW = timedelta(hours=2)
_OUTSIDE_IDEMPOTENCY_WINDOW = timedelta(minutes=11)
_PAST_EXPIRY = timedelta(hours=1)


def _iso_ago(delta: timedelta) -> str:
    return (datetime.now(timezone.utc) - delta).isoformat()


async def _backdate_created_at(job_id: str, delta: timedelta) -> None:
    async with get_db(get_settings().DB_PATH) as db:
        await update_job(db, job_id, created_at=_iso_ago(delta))


async def test_rate_limit_ignores_jobs_older_than_window(client: AsyncClient) -> None:
    """Jobs submitted 2h ago (earlier the same UTC day) don't count toward the hourly cap."""
    for i in range(3):  # MAX_JOBS_PER_HOUR = 3 in test settings
        resp = await client.post(
            "/api/v1/jobs",
            json=_config(session_name=f"old-{i}"),
            headers=session_headers(_TEST_USER_A),
        )
        await _backdate_created_at(resp.json()["job_id"], _OUTSIDE_RATE_WINDOW)

    resp = await client.post(
        "/api/v1/jobs",
        json=_config(session_name="fresh"),
        headers=session_headers(_TEST_USER_A),
    )

    assert resp.status_code == 202


async def test_idempotency_expires_after_window(client: AsyncClient) -> None:
    """Resubmitting the same config 11 minutes later creates a new job."""
    first = await client.post(
        "/api/v1/jobs", json=_MINIMAL_JOB_CONFIG, headers=session_headers(_TEST_USER_A)
    )
    await _backdate_created_at(first.json()["job_id"], _OUTSIDE_IDEMPOTENCY_WINDOW)

    second = await client.post(
        "/api/v1/jobs", json=_MINIMAL_JOB_CONFIG, headers=session_headers(_TEST_USER_A)
    )

    assert second.status_code == 202
    assert second.json()["job_id"] != first.json()["job_id"]


async def test_delete_expired_jobs_removes_job_expired_earlier_today(
    client: AsyncClient,
) -> None:
    resp = await client.post(
        "/api/v1/jobs", json=_MINIMAL_JOB_CONFIG, headers=session_headers(_TEST_USER_A)
    )
    job_id = resp.json()["job_id"]

    async with get_db(get_settings().DB_PATH) as db:
        await update_job(db, job_id, expires_at=_iso_ago(_PAST_EXPIRY))
        await delete_expired_jobs(db)
        assert await get_job(db, job_id, _TEST_USER_A) is None


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


async def test_get_status_queued(client: AsyncClient) -> None:
    submit = await client.post(
        "/api/v1/jobs", json=_MINIMAL_JOB_CONFIG, headers=session_headers(_TEST_USER_A)
    )
    job_id = submit.json()["job_id"]

    resp = await client.get(
        f"/api/v1/jobs/{job_id}/status", headers=session_headers(_TEST_USER_A)
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job_id
    assert data["status"] == "queued"
    assert data["queue_position"] >= 1


async def test_status_ownership_enforced(client: AsyncClient) -> None:
    submit = await client.post(
        "/api/v1/jobs", json=_MINIMAL_JOB_CONFIG, headers=session_headers(_TEST_USER_A)
    )
    job_id = submit.json()["job_id"]

    resp = await client.get(
        f"/api/v1/jobs/{job_id}/status", headers=session_headers(_TEST_USER_B)
    )

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "JOB_NOT_FOUND"


async def test_status_unknown_job_returns_404(client: AsyncClient) -> None:
    resp = await client.get(
        "/api/v1/jobs/nonexistent-job-id/status",
        headers=session_headers(_TEST_USER_A),
    )

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


async def test_result_returns_409_when_not_complete(client: AsyncClient) -> None:
    submit = await client.post(
        "/api/v1/jobs", json=_MINIMAL_JOB_CONFIG, headers=session_headers(_TEST_USER_A)
    )
    job_id = submit.json()["job_id"]

    resp = await client.get(
        f"/api/v1/jobs/{job_id}/result", headers=session_headers(_TEST_USER_A)
    )

    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "JOB_NOT_COMPLETE"


async def test_result_returns_payload_when_completed(client: AsyncClient) -> None:
    submit = await client.post(
        "/api/v1/jobs", json=_MINIMAL_JOB_CONFIG, headers=session_headers(_TEST_USER_A)
    )
    job_id = submit.json()["job_id"]

    result_data = {"transcript": [{"speaker": "Alice", "text": "Hello everyone"}]}
    now = datetime.now(timezone.utc).isoformat()
    settings = get_settings()
    async with get_db(settings.DB_PATH) as db:
        await update_job(
            db,
            job_id,
            status="completed",
            result_json=json.dumps(result_data),
            started_at=now,
            completed_at=now,
            participant_count=1,
            total_turns=1,
            consensus_level="high",
        )

    resp = await client.get(
        f"/api/v1/jobs/{job_id}/result", headers=session_headers(_TEST_USER_A)
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job_id
    assert data["status"] == "completed"
    assert data["result"] == result_data
    assert data["metadata"]["consensus_level"] == "high"


async def test_result_ownership_enforced(client: AsyncClient) -> None:
    submit = await client.post(
        "/api/v1/jobs", json=_MINIMAL_JOB_CONFIG, headers=session_headers(_TEST_USER_A)
    )
    job_id = submit.json()["job_id"]

    resp = await client.get(
        f"/api/v1/jobs/{job_id}/result", headers=session_headers(_TEST_USER_B)
    )

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


async def test_delete_own_job(client: AsyncClient) -> None:
    submit = await client.post(
        "/api/v1/jobs", json=_MINIMAL_JOB_CONFIG, headers=session_headers(_TEST_USER_A)
    )
    job_id = submit.json()["job_id"]

    del_resp = await client.delete(
        f"/api/v1/jobs/{job_id}", headers=session_headers(_TEST_USER_A)
    )
    assert del_resp.status_code == 204

    get_resp = await client.get(
        f"/api/v1/jobs/{job_id}/status", headers=session_headers(_TEST_USER_A)
    )
    assert get_resp.status_code == 404


async def test_delete_other_users_job_returns_404(client: AsyncClient) -> None:
    submit = await client.post(
        "/api/v1/jobs", json=_MINIMAL_JOB_CONFIG, headers=session_headers(_TEST_USER_A)
    )
    job_id = submit.json()["job_id"]

    resp = await client.delete(
        f"/api/v1/jobs/{job_id}", headers=session_headers(_TEST_USER_B)
    )

    assert resp.status_code == 404


async def test_delete_running_job_returns_409(client: AsyncClient) -> None:
    submit = await client.post(
        "/api/v1/jobs", json=_MINIMAL_JOB_CONFIG, headers=session_headers(_TEST_USER_A)
    )
    job_id = submit.json()["job_id"]

    settings = get_settings()
    async with get_db(settings.DB_PATH) as db:
        await update_job(db, job_id, status="running")

    resp = await client.delete(
        f"/api/v1/jobs/{job_id}", headers=session_headers(_TEST_USER_A)
    )

    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "JOB_RUNNING"


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


async def test_list_jobs_returns_all_user_jobs(client: AsyncClient) -> None:
    for i in range(2):
        await client.post(
            "/api/v1/jobs",
            json=_config(session_name=f"list-test-{i}"),
            headers=session_headers(_TEST_USER_A),
        )

    resp = await client.get("/api/v1/jobs", headers=session_headers(_TEST_USER_A))

    assert resp.status_code == 200
    jobs = resp.json()
    assert len(jobs) == 2


async def test_list_jobs_isolates_by_user(client: AsyncClient) -> None:
    await client.post(
        "/api/v1/jobs",
        json=_config(session_name="user-a-only"),
        headers=session_headers(_TEST_USER_A),
    )

    resp = await client.get("/api/v1/jobs", headers=session_headers(_TEST_USER_B))

    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_jobs_empty_for_new_user(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/jobs", headers=session_headers(_TEST_USER_A))

    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------


async def test_result_includes_run_metadata(client: AsyncClient) -> None:
    submit = await client.post(
        "/api/v1/jobs", json=_MINIMAL_JOB_CONFIG, headers=session_headers(_TEST_USER_A)
    )
    job_id = submit.json()["job_id"]
    run = {"provider": "openai", "model": "m", "params": {"seed": 3}}
    now = datetime.now(timezone.utc).isoformat()
    async with get_db(get_settings().DB_PATH) as db:
        await update_job(
            db,
            job_id,
            status="completed",
            result_json=json.dumps({"transcript": []}),
            completed_at=now,
            run_metadata_json=json.dumps(run),
        )

    resp = await client.get(f"/api/v1/jobs/{job_id}/result", headers=session_headers(_TEST_USER_A))

    assert resp.json()["metadata"]["run"] == run


async def test_init_db_adds_run_metadata_column_to_old_schema(tmp_path) -> None:
    import aiosqlite

    from api.database import _CREATE_JOBS, init_db

    # The schema as shipped before run_metadata_json existed
    old_schema = _CREATE_JOBS.replace("consensus_level   TEXT,\n", "consensus_level   TEXT\n")
    old_schema = old_schema.replace("    run_metadata_json TEXT\n", "")
    assert "run_metadata_json" not in old_schema

    db_path = str(tmp_path / "old.db")
    async with aiosqlite.connect(db_path) as db:
        await db.execute(old_schema)
        await db.commit()

    await init_db(db_path)
    await init_db(db_path)  # idempotent

    async with aiosqlite.connect(db_path) as db:
        columns = {row[1] for row in await (await db.execute("PRAGMA table_info(jobs)")).fetchall()}
    assert "run_metadata_json" in columns
