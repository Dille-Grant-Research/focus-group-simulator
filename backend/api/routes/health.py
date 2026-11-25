"""Health check and operational metrics endpoints.

/health  — deep check of all critical components; returns 503 if any are down.
/metrics — lightweight SQLite aggregation queries; no external calls.

Both routes are auth-exempt (listed in AuthMiddleware._EXEMPT_PREFIXES) and mount
at the API root (no /api/v1 prefix) so Docker healthchecks can reach them simply.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.config import Settings, build_llm_provider, get_settings
from api.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ops"])


@router.get("/health")
async def health_check(request: Request) -> JSONResponse:
    settings: Settings = get_settings()
    components: dict[str, str] = {}

    # 1. Database
    try:
        async with get_db(settings.DB_PATH) as db:
            await db.execute("SELECT 1")
        components["database"] = "ok"
    except Exception:
        logger.exception("Health check: database error")
        components["database"] = "error"

    # 2. Worker — worker_task is stored in app.state by the lifespan function
    worker_task: asyncio.Task | None = getattr(request.app.state, "worker_task", None)
    if worker_task is None or worker_task.done():
        components["worker"] = "dead"
    else:
        components["worker"] = "alive"

    # 3. Inference service (Ollama or OpenAI-compatible, per LLM_PROVIDER)
    try:
        available = await build_llm_provider(settings).is_available()
        components["llm"] = "reachable" if available else "unreachable"
    except Exception:
        logger.exception("Health check: inference provider error")
        components["llm"] = "error"

    all_ok = (
        components["database"] == "ok"
        and components["worker"] == "alive"
        and components["llm"] == "reachable"
    )
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={
            "status": "healthy" if all_ok else "degraded",
            "components": components,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@router.get("/metrics")
async def metrics(request: Request) -> JSONResponse:
    settings: Settings = get_settings()

    async with get_db(settings.DB_PATH) as db:
        queue_depth_row = await (
            await db.execute("SELECT COUNT(*) FROM jobs WHERE status = 'queued'")
        ).fetchone()

        completed_24h_row = await (
            await db.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = 'completed'"
                " AND julianday(completed_at) > julianday('now', '-24 hours')"
            )
        ).fetchone()

        failed_24h_row = await (
            await db.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = 'failed'"
                " AND julianday(completed_at) > julianday('now', '-24 hours')"
            )
        ).fetchone()

        avg_duration_row = await (
            await db.execute(
                "SELECT AVG(CAST((julianday(completed_at) - julianday(started_at)) * 86400 AS INTEGER))"
                " FROM jobs WHERE status = 'completed'"
                " AND julianday(completed_at) > julianday('now', '-24 hours')"
            )
        ).fetchone()

        running_row = await (
            await db.execute(
                "SELECT job_id FROM jobs WHERE status = 'running' LIMIT 1"
            )
        ).fetchone()

    current_job_id = running_row["job_id"] if running_row else None
    worker_task: asyncio.Task | None = getattr(request.app.state, "worker_task", None)
    worker_status = "dead" if (worker_task is None or worker_task.done()) else (
        "processing" if current_job_id else "idle"
    )

    return JSONResponse(
        content={
            "queue_depth": queue_depth_row[0] if queue_depth_row else 0,
            "jobs_completed_24h": completed_24h_row[0] if completed_24h_row else 0,
            "jobs_failed_24h": failed_24h_row[0] if failed_24h_row else 0,
            "avg_duration_seconds": avg_duration_row[0] if avg_duration_row else None,
            "worker_status": worker_status,
        }
    )
