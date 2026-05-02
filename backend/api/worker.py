"""Async job worker — single FIFO background task.

One worker, one job at a time. The worker polls SQLite for the next queued
job, runs the simulation, writes the result back, and loops. It shares the
process with the FastAPI server via asyncio tasks.

Crash recovery: any job stuck in 'running' at startup is reset to 'queued'
by startup_recovery() in database.py before this worker begins.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

from api.config import Settings
from api.database import delete_expired_jobs, get_db, get_next_queued_job, update_job
from focus_group.core.session import FocusGroupSession
from focus_group.llm.factory import create_provider
from focus_group.models.config import OllamaModelConfig, SessionConfig

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 2
_CLEANUP_EVERY_N_ITERATIONS = 10


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_ollama_config(settings: Settings) -> OllamaModelConfig:
    return OllamaModelConfig(
        model_name=settings.OLLAMA_MODEL,
        base_url=settings.ollama_base_url,
    )


def _extract_consensus_level(result_dict: dict) -> str | None:
    """Best-effort extraction of an overall consensus level from the result dict."""
    # When memory is enabled, the last question's consensus lives in participant memory dumps.
    # Fall back to None if unavailable — this column is optional summary metadata.
    dumps = result_dict.get("participant_memory_dumps")
    if dumps:
        try:
            last_q = dumps[0].get("completed_questions", [])
            if last_q:
                return last_q[-1].get("consensus_level")
        except (IndexError, AttributeError, KeyError):
            pass
    return None


async def job_worker(settings: Settings, stop_event: asyncio.Event) -> None:
    """Main worker loop. Runs until stop_event is set."""
    logger.info("Job worker started")
    iteration = 0

    while not stop_event.is_set():
        async with get_db(settings.DB_PATH) as db:
            if iteration % _CLEANUP_EVERY_N_ITERATIONS == 0:
                await delete_expired_jobs(db)

            job = await get_next_queued_job(db)

            if job is None:
                iteration += 1
                await asyncio.sleep(_POLL_INTERVAL_SECONDS)
                continue

            job_id: str = job["job_id"]
            logger.info("Starting job %s", job_id)

            try:
                await update_job(db, job_id, status="running", started_at=_utcnow())

                config = SessionConfig.model_validate_json(job["config_json"])
                provider = create_provider(_build_ollama_config(settings))
                session = FocusGroupSession(provider=provider, config=config)

                result = await session.run()
                result_dict = result.to_dict()

                await update_job(
                    db,
                    job_id,
                    status="completed",
                    completed_at=_utcnow(),
                    result_json=json.dumps(result_dict),
                    participant_count=len(result.participants),
                    total_turns=result.total_responses,
                    consensus_level=_extract_consensus_level(result_dict),
                )
                logger.info(
                    "Job %s completed (%d participants, %d turns)",
                    job_id,
                    len(result.participants),
                    result.total_responses,
                )

            except Exception as exc:
                logger.exception("Job %s failed", job_id)
                await update_job(db, job_id, status="failed", error_message=str(exc))

        iteration += 1

    logger.info("Job worker stopped")
