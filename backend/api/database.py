"""SQLite data layer — WAL mode, schema management, and helper queries.

All queries that touch the jobs table include a user_id filter except for
get_next_queued_job (used by the worker, which sees all jobs) and the
cleanup / startup functions. This enforces job ownership at the database
layer, not just the route layer.
"""

import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

import aiosqlite


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_USERS = """
CREATE TABLE IF NOT EXISTS users (
    user_id    TEXT PRIMARY KEY,
    first_seen TEXT NOT NULL,
    last_seen  TEXT NOT NULL,
    sessions_valid_after INTEGER
)
"""

_CREATE_JOBS = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id            TEXT PRIMARY KEY,
    user_id           TEXT NOT NULL REFERENCES users(user_id),
    status            TEXT NOT NULL DEFAULT 'queued',
    config_json       TEXT NOT NULL,
    idempotency_key   TEXT,
    result_json       TEXT,
    error_message     TEXT,
    created_at        TEXT NOT NULL,
    expires_at        TEXT NOT NULL,
    started_at        TEXT,
    completed_at      TEXT,
    participant_count INTEGER,
    total_turns       INTEGER,
    consensus_level   TEXT,
    run_metadata_json TEXT
)
"""

# Append-only ledger of accepted submissions, used for the hourly rate limit. It is
# separate from jobs so deleting or failing a job cannot hand quota back.
_CREATE_JOB_SUBMISSIONS = """
CREATE TABLE IF NOT EXISTS job_submissions (
    user_id      TEXT NOT NULL,
    submitted_at TEXT NOT NULL
)
"""

# Ledger rows older than this are pruned; must exceed any rate-limit window.
_SUBMISSION_RETENTION_HOURS = 24

# Columns added after the first release. init_db adds any that an existing
# database file is missing, so deployed volumes upgrade in place.
_ADDED_JOB_COLUMNS = {
    "run_metadata_json": "TEXT",
}

_ADDED_USER_COLUMNS = {
    "sessions_valid_after": "INTEGER",
}

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_jobs_user        ON jobs(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_status_time ON jobs(status, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_idempotency ON jobs(idempotency_key)",
    "CREATE INDEX IF NOT EXISTS idx_submissions_user_time "
    "ON job_submissions(user_id, submitted_at)",
]


# ---------------------------------------------------------------------------
# Initialization & recovery
# ---------------------------------------------------------------------------


async def init_db(db_path: str) -> None:
    """Initialize the database: enable WAL mode and create schema."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute(_CREATE_USERS)
        await db.execute(_CREATE_JOBS)
        submissions_existed = await _table_exists(db, "job_submissions")
        await db.execute(_CREATE_JOB_SUBMISSIONS)
        await _add_missing_columns(db, "users", _ADDED_USER_COLUMNS)
        await _add_missing_columns(db, "jobs", _ADDED_JOB_COLUMNS)
        for stmt in _CREATE_INDEXES:
            await db.execute(stmt)
        if not submissions_existed:
            await _backfill_submissions(db)
        await db.commit()


async def _table_exists(db: aiosqlite.Connection, table: str) -> bool:
    cursor = await db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    )
    return await cursor.fetchone() is not None


async def _backfill_submissions(db: aiosqlite.Connection) -> None:
    """Seed the ledger from existing jobs so upgrading does not reset anyone's quota.

    Runs only when init_db has just created the table.
    """
    await db.execute(
        """
        INSERT INTO job_submissions (user_id, submitted_at)
        SELECT user_id, created_at FROM jobs
        WHERE julianday(created_at) >= julianday('now', ? || ' hours')
        """,
        (f"-{_SUBMISSION_RETENTION_HOURS}",),
    )


async def _add_missing_columns(
    db: aiosqlite.Connection, table: str, added_columns: dict[str, str]
) -> None:
    cursor = await db.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in await cursor.fetchall()}
    for name, column_type in added_columns.items():
        if name not in existing:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {column_type}")


async def startup_recovery(db_path: str) -> None:
    """Reset any jobs stuck in 'running' back to 'queued' after a crash."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE jobs SET status = 'queued', started_at = NULL WHERE status = 'running'"
        )
        await db.commit()


# ---------------------------------------------------------------------------
# Connection context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def get_db(db_path: str) -> AsyncIterator[aiosqlite.Connection]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        yield db


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# Timestamps are stored as Python ISO strings ("2026-11-23T01:00:00+00:00") but
# SQLite's datetime('now') is "2026-11-23 03:00:00". Compared as text, 'T' > ' '
# makes any time earlier the same UTC day look newer than 'now'. Always compare
# through julianday(), which parses both formats (and the offset) to a number.


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------


async def upsert_user(db: aiosqlite.Connection, user_id: str) -> None:
    now = _utcnow()
    await db.execute(
        """
        INSERT INTO users (user_id, first_seen, last_seen) VALUES (?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET last_seen = excluded.last_seen
        """,
        (user_id, now, now),
    )
    await db.commit()


async def revoke_user_sessions(db: aiosqlite.Connection, user_id: str, at_epoch: int) -> None:
    """Reject every session for this user issued at or before at_epoch."""
    now = _utcnow()
    await db.execute(
        """
        INSERT INTO users (user_id, first_seen, last_seen, sessions_valid_after)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET sessions_valid_after = excluded.sessions_valid_after
        """,
        (user_id, now, now, at_epoch),
    )
    await db.commit()


async def get_sessions_valid_after(db: aiosqlite.Connection, user_id: str) -> int | None:
    cursor = await db.execute(
        "SELECT sessions_valid_after FROM users WHERE user_id = ?", (user_id,)
    )
    row = await cursor.fetchone()
    return row[0] if row else None


async def create_job(
    db: aiosqlite.Connection,
    *,
    job_id: str,
    user_id: str,
    config_json: str,
    idempotency_key: str | None,
    expires_at: str,
) -> None:
    """Insert a queued job WITHOUT the rate-limit check or ledger row.

    For tests and internal use only; request handlers use create_job_within_limit.
    """
    await _insert_job(
        db,
        job_id=job_id,
        user_id=user_id,
        config_json=config_json,
        idempotency_key=idempotency_key,
        expires_at=expires_at,
        created_at=_utcnow(),
    )
    await db.commit()


async def _insert_job(
    db: aiosqlite.Connection,
    *,
    job_id: str,
    user_id: str,
    config_json: str,
    idempotency_key: str | None,
    expires_at: str,
    created_at: str,
) -> None:
    await db.execute(
        """
        INSERT INTO jobs
            (job_id, user_id, status, config_json, idempotency_key, created_at, expires_at)
        VALUES (?, ?, 'queued', ?, ?, ?, ?)
        """,
        (job_id, user_id, config_json, idempotency_key, created_at, expires_at),
    )


async def create_job_within_limit(
    db: aiosqlite.Connection,
    *,
    job_id: str,
    user_id: str,
    config_json: str,
    idempotency_key: str,
    expires_at: str,
    limit: int,
    window_hours: int,
) -> bool:
    """Insert a queued job unless the user is at the limit; the count and insert are atomic.

    BEGIN IMMEDIATE takes SQLite's write lock before counting, so concurrent submissions
    are serialized. The job and its ledger row are written together. Returns True when
    inserted, False (nothing written) at the limit.
    """
    now = _utcnow()
    await db.execute("BEGIN IMMEDIATE")
    try:
        if await count_user_jobs_in_window(db, user_id, hours=window_hours) >= limit:
            await db.rollback()
            return False
        await _insert_job(
            db,
            job_id=job_id,
            user_id=user_id,
            config_json=config_json,
            idempotency_key=idempotency_key,
            expires_at=expires_at,
            created_at=now,
        )
        await db.execute(
            "INSERT INTO job_submissions (user_id, submitted_at) VALUES (?, ?)",
            (user_id, now),
        )
        await db.commit()
    except BaseException:
        await db.rollback()
        raise
    return True


async def update_job(db: aiosqlite.Connection, job_id: str, **fields: object) -> None:
    """Update arbitrary columns on a job row. Field names must be valid column names."""
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = [*fields.values(), job_id]
    await db.execute(f"UPDATE jobs SET {set_clause} WHERE job_id = ?", values)
    await db.commit()


async def delete_expired_jobs(db: aiosqlite.Connection) -> None:
    await db.execute("DELETE FROM jobs WHERE julianday(expires_at) < julianday('now')")
    await db.execute(
        "DELETE FROM job_submissions "
        "WHERE julianday(submitted_at) < julianday('now', ? || ' hours')",
        (f"-{_SUBMISSION_RETENTION_HOURS}",),
    )
    await db.commit()


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


async def get_job(
    db: aiosqlite.Connection, job_id: str, user_id: str
) -> sqlite3.Row | None:
    """Fetch a job, enforcing ownership via user_id filter."""
    cursor = await db.execute(
        "SELECT * FROM jobs WHERE job_id = ? AND user_id = ?",
        (job_id, user_id),
    )
    return await cursor.fetchone()


async def get_job_by_idempotency_key(
    db: aiosqlite.Connection, key: str, user_id: str
) -> sqlite3.Row | None:
    """Return a non-failed job submitted within the last 10 minutes with this key."""
    cursor = await db.execute(
        """
        SELECT * FROM jobs
        WHERE idempotency_key = ?
          AND user_id = ?
          AND status != 'failed'
          AND julianday(created_at) > julianday('now', '-10 minutes')
        """,
        (key, user_id),
    )
    return await cursor.fetchone()


async def get_next_queued_job(db: aiosqlite.Connection) -> sqlite3.Row | None:
    """FIFO: return the oldest queued job regardless of user."""
    cursor = await db.execute(
        "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1"
    )
    return await cursor.fetchone()


async def get_queue_position(db: aiosqlite.Connection, job_id: str) -> int:
    """Return 1-based queue position (1 = next to run)."""
    cursor = await db.execute(
        """
        SELECT COUNT(*) FROM jobs
        WHERE status = 'queued'
          AND created_at < (SELECT created_at FROM jobs WHERE job_id = ?)
        """,
        (job_id,),
    )
    row = await cursor.fetchone()
    return (row[0] + 1) if row else 1


async def count_user_jobs_in_window(
    db: aiosqlite.Connection, user_id: str, hours: int = 1
) -> int:
    """Count submissions by user in the last N hours (for rate limiting).

    Reads the job_submissions ledger, so deleted and failed jobs still count.
    """
    cursor = await db.execute(
        """
        SELECT COUNT(*) FROM job_submissions
        WHERE user_id = ?
          AND julianday(submitted_at) > julianday('now', ? || ' hours')
        """,
        (user_id, f"-{hours}"),
    )
    row = await cursor.fetchone()
    return row[0] if row else 0


async def list_user_jobs(db: aiosqlite.Connection, user_id: str) -> list[sqlite3.Row]:
    """Return all jobs for a user, newest first."""
    cursor = await db.execute(
        "SELECT * FROM jobs WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    )
    return await cursor.fetchall()
