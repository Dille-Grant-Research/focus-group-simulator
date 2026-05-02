"""Job submission, status, result, list, and delete endpoints.

All routes require authentication — request.state.user_id is set by AuthMiddleware.
Ownership is enforced at the database query layer (get_job includes a user_id filter),
so URL manipulation cannot expose another user's jobs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator
from uuid import uuid4

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.config import Settings, get_settings
from api.database import (
    count_user_jobs_in_window,
    create_job,
    get_db,
    get_job,
    get_job_by_idempotency_key,
    get_queue_position,
    list_user_jobs,
)
from api.middleware.rate_limit import check_rate_limit
from focus_group.models.config import SessionConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])

_JOB_RETENTION_DAYS = 30


# ---------------------------------------------------------------------------
# Shared dependency
# ---------------------------------------------------------------------------


async def _get_db(settings: Settings = Depends(get_settings)) -> AsyncIterator[aiosqlite.Connection]:
    async with get_db(settings.DB_PATH) as db:
        yield db


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class JobSubmitRequest(BaseModel):
    config: SessionConfig


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    estimated_position: int | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    started_at: str | None
    completed_at: str | None
    expires_at: str
    queue_position: int | None


class JobSummary(BaseModel):
    job_id: str
    status: str
    created_at: str
    expires_at: str
    participant_count: int | None
    total_turns: int | None
    consensus_level: str | None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", status_code=202, response_model=JobSubmitResponse)
async def submit_job(
    body: JobSubmitRequest,
    request: Request,
    db: aiosqlite.Connection = Depends(_get_db),
    settings: Settings = Depends(get_settings),
) -> JobSubmitResponse:
    user_id: str = request.state.user_id
    config_json = body.config.model_dump_json()
    idempotency_key = hashlib.sha256(f"{user_id}:{config_json}".encode()).hexdigest()

    existing = await get_job_by_idempotency_key(db, idempotency_key, user_id)
    if existing is not None:
        position = (
            await get_queue_position(db, existing["job_id"])
            if existing["status"] == "queued"
            else None
        )
        return JobSubmitResponse(
            job_id=existing["job_id"],
            status=existing["status"],
            estimated_position=position,
        )

    await check_rate_limit(db, user_id, settings)

    job_id = str(uuid4())
    expires_at = (datetime.now(timezone.utc) + timedelta(days=_JOB_RETENTION_DAYS)).isoformat()
    await create_job(
        db,
        job_id=job_id,
        user_id=user_id,
        config_json=config_json,
        idempotency_key=idempotency_key,
        expires_at=expires_at,
    )
    position = await get_queue_position(db, job_id)
    return JobSubmitResponse(job_id=job_id, status="queued", estimated_position=position)


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    request: Request,
    db: aiosqlite.Connection = Depends(_get_db),
) -> JobStatusResponse:
    user_id: str = request.state.user_id
    row = await get_job(db, job_id, user_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "JOB_NOT_FOUND", "message": "Job not found.", "details": {}},
        )
    position = (
        await get_queue_position(db, job_id) if row["status"] == "queued" else None
    )
    return JobStatusResponse(
        job_id=row["job_id"],
        status=row["status"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        expires_at=row["expires_at"],
        queue_position=position,
    )


@router.get("/{job_id}/result")
async def get_job_result(
    job_id: str,
    request: Request,
    db: aiosqlite.Connection = Depends(_get_db),
) -> JSONResponse:
    user_id: str = request.state.user_id
    row = await get_job(db, job_id, user_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "JOB_NOT_FOUND", "message": "Job not found.", "details": {}},
        )
    if row["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "JOB_NOT_COMPLETE",
                "message": "Job has not completed yet.",
                "details": {"status": row["status"]},
            },
        )
    return JSONResponse(
        content={
            "job_id": row["job_id"],
            "status": row["status"],
            "config": json.loads(row["config_json"]),
            "result": json.loads(row["result_json"]),
            "metadata": {
                "participant_count": row["participant_count"],
                "total_turns": row["total_turns"],
                "consensus_level": row["consensus_level"],
                "created_at": row["created_at"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "expires_at": row["expires_at"],
            },
        }
    )


@router.get("", response_model=list[JobSummary])
async def list_jobs(
    request: Request,
    db: aiosqlite.Connection = Depends(_get_db),
) -> list[JobSummary]:
    user_id: str = request.state.user_id
    rows = await list_user_jobs(db, user_id)
    return [
        JobSummary(
            job_id=row["job_id"],
            status=row["status"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            participant_count=row["participant_count"],
            total_turns=row["total_turns"],
            consensus_level=row["consensus_level"],
        )
        for row in rows
    ]


@router.delete("/{job_id}", status_code=204)
async def delete_job(
    job_id: str,
    request: Request,
    db: aiosqlite.Connection = Depends(_get_db),
) -> None:
    user_id: str = request.state.user_id
    row = await get_job(db, job_id, user_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "JOB_NOT_FOUND", "message": "Job not found.", "details": {}},
        )
    if row["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "JOB_RUNNING",
                "message": "Cannot delete a job that is currently running.",
                "details": {},
            },
        )
    await db.execute("DELETE FROM jobs WHERE job_id = ? AND user_id = ?", (job_id, user_id))
    await db.commit()
