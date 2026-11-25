"""Step 13: Job lifecycle tests.

Covers: submit, idempotency, rate limiting, status (with ownership enforcement),
result retrieval (pending + completed), delete, and list.
"""

from __future__ import annotations

import asyncio
import copy
import json
from datetime import datetime, timedelta, timezone

import aiosqlite
import pytest
from httpx import AsyncClient

from api import database
from api.config import get_settings
from api.database import delete_expired_jobs, get_db, get_job, update_job
from focus_group.models.config import (
    MAX_DEMOGRAPHIC_CHARS,
    MAX_PARTICIPANT_NAME_CHARS,
    MAX_PURPOSE_CHARS,
    MAX_QUESTION_CHARS,
    MAX_SESSION_NAME_CHARS,
)
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
_OUTSIDE_RETENTION = timedelta(hours=25)


_BACKFILL_JOB_AGE_MINUTES = 10
_QUOTA = 3  # MAX_JOBS_PER_HOUR in test settings
_BARRIER_TIMEOUT_SECONDS = 0.2


def _install_count_barrier(monkeypatch: pytest.MonkeyPatch, expected_arrivals: int) -> None:
    """Hold each rate-limit count until all requests have counted (or a short timeout).

    Pre-fix, every request counts before any inserts, so all pass the check. Post-fix, the
    count runs inside the write transaction: the first request times out waiting, and the
    others are blocked on BEGIN IMMEDIATE, so they run one at a time.
    """
    real_count = database.count_user_jobs_in_window
    arrivals = 0
    all_arrived = asyncio.Event()

    async def counting_with_barrier(*args: object, **kwargs: object) -> int:
        nonlocal arrivals
        count = await real_count(*args, **kwargs)
        arrivals += 1
        if arrivals >= expected_arrivals:
            all_arrived.set()
        try:
            await asyncio.wait_for(all_arrived.wait(), timeout=_BARRIER_TIMEOUT_SECONDS)
        except TimeoutError:
            pass
        return count

    monkeypatch.setattr(database, "count_user_jobs_in_window", counting_with_barrier)


async def test_concurrent_submissions_cannot_exceed_rate_limit(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_count_barrier(monkeypatch, 2 * _QUOTA)
    posts = [
        client.post(
            "/api/v1/jobs",
            json=_config(session_name=f"race-{i}"),
            headers=session_headers(_TEST_USER_A),
        )
        for i in range(2 * _QUOTA)
    ]
    statuses = sorted(r.status_code for r in await asyncio.gather(*posts))

    assert statuses == [202] * _QUOTA + [429] * _QUOTA


async def test_concurrent_submissions_from_two_users_each_get_full_quota(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_count_barrier(monkeypatch, 2 * _QUOTA)
    posts = [
        client.post(
            "/api/v1/jobs",
            json=_config(session_name=f"{label}-{i}"),
            headers=session_headers(user),
        )
        for label, user in (("a", _TEST_USER_A), ("b", _TEST_USER_B))
        for i in range(_QUOTA)
    ]
    responses = await asyncio.gather(*posts)

    assert [r.status_code for r in responses] == [202] * (2 * _QUOTA)


async def test_duplicate_submission_at_limit_returns_existing_job(client: AsyncClient) -> None:
    first_ids = []
    for i in range(_QUOTA):
        resp = await client.post(
            "/api/v1/jobs",
            json=_config(session_name=f"dup-{i}"),
            headers=session_headers(_TEST_USER_A),
        )
        assert resp.status_code == 202
        first_ids.append(resp.json()["job_id"])

    resp = await client.post(
        "/api/v1/jobs",
        json=_config(session_name="dup-0"),
        headers=session_headers(_TEST_USER_A),
    )

    assert resp.status_code == 202
    assert resp.json()["job_id"] == first_ids[0]


async def _submit_distinct(client: AsyncClient, count: int, prefix: str) -> list[str]:
    job_ids = []
    for i in range(count):
        resp = await client.post(
            "/api/v1/jobs",
            json=_config(session_name=f"{prefix}-{i}"),
            headers=session_headers(_TEST_USER_A),
        )
        assert resp.status_code == 202
        job_ids.append(resp.json()["job_id"])
    return job_ids


async def _assert_quota_exhausted(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/jobs",
        json=_config(session_name="over-quota"),
        headers=session_headers(_TEST_USER_A),
    )
    assert resp.status_code == 429


async def test_deleting_job_does_not_restore_quota(client: AsyncClient) -> None:
    job_ids = await _submit_distinct(client, _QUOTA, "del")

    resp = await client.delete(f"/api/v1/jobs/{job_ids[0]}", headers=session_headers(_TEST_USER_A))
    assert resp.status_code == 204

    await _assert_quota_exhausted(client)


async def test_failed_jobs_count_toward_quota(client: AsyncClient) -> None:
    job_ids = await _submit_distinct(client, _QUOTA, "fail")
    async with get_db(get_settings().DB_PATH) as db:
        for job_id in job_ids:
            await update_job(db, job_id, status="failed")

    await _assert_quota_exhausted(client)


async def test_init_db_adds_job_submissions_table_to_old_schema(tmp_path) -> None:
    db_path = str(tmp_path / "old.db")
    async with aiosqlite.connect(db_path) as db:
        await db.execute(database._CREATE_USERS)
        await db.execute(database._CREATE_JOBS)
        await db.execute(
            "INSERT INTO users (user_id, first_seen, last_seen) VALUES ('u', 'x', 'x')"
        )
        await db.execute(
            "INSERT INTO jobs (job_id, user_id, config_json, created_at, expires_at)"
            " VALUES ('j1', 'u', '{}', 'x', 'x')"
        )
        await db.commit()

    await database.init_db(db_path)

    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE name = 'job_submissions'"
        )
        assert await cursor.fetchone() is not None
        cursor = await db.execute("SELECT job_id FROM jobs")
        assert [row[0] for row in await cursor.fetchall()] == ["j1"]


async def test_init_db_backfills_submissions_once_from_recent_jobs(tmp_path) -> None:
    db_path = str(tmp_path / "old.db")
    recent = _iso_ago(timedelta(minutes=_BACKFILL_JOB_AGE_MINUTES))
    ancient = _iso_ago(_OUTSIDE_RETENTION)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(database._CREATE_USERS)
        await db.execute(database._CREATE_JOBS)
        await db.execute(
            "INSERT INTO users (user_id, first_seen, last_seen) VALUES ('u', 'x', 'x')"
        )
        for job_id, created in (("recent", recent), ("ancient", ancient)):
            await db.execute(
                "INSERT INTO jobs (job_id, user_id, config_json, created_at, expires_at)"
                " VALUES (?, 'u', '{}', ?, 'x')",
                (job_id, created),
            )
        await db.commit()

    await database.init_db(db_path)
    await database.init_db(db_path)  # second startup must not duplicate

    async with get_db(db_path) as db:
        assert await database.count_user_jobs_in_window(db, "u", hours=1) == 1
        cursor = await db.execute("SELECT COUNT(*) FROM job_submissions")
        assert (await cursor.fetchone())[0] == 1


async def test_delete_expired_jobs_prunes_old_submissions(client: AsyncClient) -> None:
    await _submit_distinct(client, 1, "prune")
    assert await _count_submissions() == 1

    async with get_db(get_settings().DB_PATH) as db:
        await delete_expired_jobs(db)
    assert await _count_submissions() == 1  # recent submission is kept

    await _backdate_submissions(_TEST_USER_A, _OUTSIDE_RETENTION)
    async with get_db(get_settings().DB_PATH) as db:
        await delete_expired_jobs(db)
    assert await _count_submissions() == 0


def _iso_ago(delta: timedelta) -> str:
    return (datetime.now(timezone.utc) - delta).isoformat()


async def _backdate_created_at(job_id: str, delta: timedelta) -> None:
    async with get_db(get_settings().DB_PATH) as db:
        await update_job(db, job_id, created_at=_iso_ago(delta))


async def _backdate_submissions(user_id: str, delta: timedelta) -> None:
    async with get_db(get_settings().DB_PATH) as db:
        await db.execute(
            "UPDATE job_submissions SET submitted_at = ? WHERE user_id = ?",
            (_iso_ago(delta), user_id),
        )
        await db.commit()


async def _count_submissions() -> int:
    async with get_db(get_settings().DB_PATH) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM job_submissions")
        return (await cursor.fetchone())[0]


async def test_rate_limit_ignores_jobs_older_than_window(client: AsyncClient) -> None:
    """Submissions 2h ago (earlier the same UTC day) don't count toward the hourly cap."""
    await _submit_distinct(client, _QUOTA, "old")
    await _backdate_submissions(_TEST_USER_A, _OUTSIDE_RATE_WINDOW)

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


# ---------------------------------------------------------------------------
# Text field length caps
# ---------------------------------------------------------------------------

_OVERSIZED_FIELDS = [
    pytest.param("demographics", ["x" * (MAX_DEMOGRAPHIC_CHARS + 1)], id="demographics"),
    pytest.param(
        "participant_names", ["x" * (MAX_PARTICIPANT_NAME_CHARS + 1)], id="participant_names"
    ),
    pytest.param("questions", ["x" * (MAX_QUESTION_CHARS + 1)], id="questions"),
    pytest.param("closing_question", "x" * (MAX_QUESTION_CHARS + 1), id="closing_question"),
    pytest.param("purpose", "x" * (MAX_PURPOSE_CHARS + 1), id="purpose"),
    pytest.param("session_name", "x" * (MAX_SESSION_NAME_CHARS + 1), id="session_name"),
]


@pytest.mark.parametrize(("field", "value"), _OVERSIZED_FIELDS)
async def test_oversized_text_field_returns_422(
    client: AsyncClient, field: str, value: object
) -> None:
    body = copy.deepcopy(_MINIMAL_JOB_CONFIG)
    body["config"][field] = value

    resp = await client.post("/api/v1/jobs", json=body, headers=session_headers(_TEST_USER_A))

    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert field in error["details"]["errors"][0]["loc"]


async def test_text_fields_at_exact_cap_are_accepted(client: AsyncClient) -> None:
    body = copy.deepcopy(_MINIMAL_JOB_CONFIG)
    body["config"].update(
        demographics=["x" * MAX_DEMOGRAPHIC_CHARS],
        participant_names=["x" * MAX_PARTICIPANT_NAME_CHARS],
        questions=["x" * MAX_QUESTION_CHARS],
        closing_question="x" * MAX_QUESTION_CHARS,
        purpose="x" * MAX_PURPOSE_CHARS,
        session_name="x" * MAX_SESSION_NAME_CHARS,
    )

    resp = await client.post("/api/v1/jobs", json=body, headers=session_headers(_TEST_USER_A))

    assert resp.status_code == 202
