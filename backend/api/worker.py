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
import time
from datetime import datetime, timezone

from api.config import SamplingParams, Settings, build_llm_provider, resolve_sampling
from api.database import delete_expired_jobs, get_db, get_next_queued_job, update_job
from focus_group.core.session import FocusGroupSession
from focus_group.llm.base import BaseLLMProvider
from focus_group.models.config import SessionConfig

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 2
_CLEANUP_EVERY_N_ITERATIONS = 10


class JobFailedError(Exception):
    """The session finished but reported errors, so its result is incomplete.

    result_json carries the partial result so the transcript isn't lost with the failure.
    """

    def __init__(self, message: str, result_json: str | None = None) -> None:
        super().__init__(message)
        self.result_json = result_json


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _run_metadata(
    settings: Settings,
    sampling: SamplingParams,
    provider: BaseLLMProvider | None,
    started: float,
) -> str:
    """Record what ran: provider, model, sampling params, token usage, duration."""
    usage = provider.usage_totals if provider is not None else None
    return json.dumps(
        {
            "provider": settings.LLM_PROVIDER.value,
            "model": settings.LLM_MODEL,
            "params": {
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "max_tokens": sampling.max_tokens,
                "seed": sampling.seed,
                "reasoning_effort": settings.LLM_REASONING_EFFORT,
            },
            "usage": (
                {
                    "calls": usage.calls,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "max_prompt_tokens": usage.max_prompt_tokens,
                }
                if usage is not None
                else None
            ),
            "duration_seconds": round(time.monotonic() - started, 1),
        }
    )


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
            provider: BaseLLMProvider | None = None
            sampling: SamplingParams | None = None
            started = time.monotonic()

            try:
                await update_job(db, job_id, status="running", started_at=_utcnow())

                config = SessionConfig.model_validate_json(job["config_json"])
                sampling = resolve_sampling(settings, config.model)
                provider = build_llm_provider(settings, sampling)
                session = FocusGroupSession(provider=provider, config=config)

                try:
                    result = await asyncio.wait_for(
                        session.run(), timeout=settings.JOB_TIMEOUT_SECONDS
                    )
                except TimeoutError as exc:
                    raise JobFailedError(
                        f"Job timed out after {settings.JOB_TIMEOUT_SECONDS}s "
                        "(JOB_TIMEOUT_SECONDS)"
                    ) from exc

                if config.exclude_nonverbals:
                    for qr in result.question_results:
                        qr.nonverbal_reactions = []
                    if result.closing_result:
                        result.closing_result.nonverbal_reactions = []
                    if result.introduction_result:
                        result.introduction_result.reactions = []

                result_dict = result.to_dict()

                if result.errors:
                    raise JobFailedError(
                        "; ".join(result.errors), result_json=json.dumps(result_dict)
                    )

                await update_job(
                    db,
                    job_id,
                    status="completed",
                    completed_at=_utcnow(),
                    result_json=json.dumps(result_dict),
                    participant_count=len(result.participants),
                    total_turns=result.total_responses,
                    consensus_level=_extract_consensus_level(result_dict),
                    run_metadata_json=_run_metadata(settings, sampling, provider, started),
                )
                logger.info(
                    "Job %s completed (%d participants, %d turns)",
                    job_id,
                    len(result.participants),
                    result.total_responses,
                )

            except Exception as exc:
                logger.exception("Job %s failed", job_id)
                await update_job(
                    db,
                    job_id,
                    status="failed",
                    completed_at=_utcnow(),
                    error_message=str(exc),
                    result_json=exc.result_json if isinstance(exc, JobFailedError) else None,
                    run_metadata_json=(
                        _run_metadata(settings, sampling, provider, started)
                        if sampling is not None
                        else None
                    ),
                )

            finally:
                if provider is not None:
                    await provider.aclose()

        iteration += 1

    logger.info("Job worker stopped")
