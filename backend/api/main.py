"""FastAPI application factory.

Wires together: lifespan (DB init + worker task), CORS middleware,
and the versioned API router. Authentication and rate-limiting middleware
are added in Phase 2 and Phase 3 respectively.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import Settings, get_settings
from api.database import init_db, startup_recovery
from api.middleware.auth import AuthMiddleware
from api.routes import api_router
from api.routes.health import router as health_router
from api.worker import job_worker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    settings: Settings = get_settings()

    logging.basicConfig(level=settings.LOG_LEVEL.upper())

    await init_db(settings.DB_PATH)
    await startup_recovery(settings.DB_PATH)
    logger.info("Database ready at %s", settings.DB_PATH)

    stop_event = asyncio.Event()
    worker_task = asyncio.create_task(job_worker(settings, stop_event))
    app.state.stop_event = stop_event
    app.state.worker_task = worker_task

    yield

    # Graceful shutdown: signal worker and wait up to JOB_TIMEOUT_SECONDS
    stop_event.set()
    try:
        await asyncio.wait_for(worker_task, timeout=float(settings.JOB_TIMEOUT_SECONDS))
        logger.info("Worker shut down cleanly")
    except asyncio.TimeoutError:
        worker_task.cancel()
        logger.warning(
            "Worker did not finish within %ds — cancelled (job will be requeued on next start)",
            settings.JOB_TIMEOUT_SECONDS,
        )


# ---------------------------------------------------------------------------
# Exception handler — standard error envelope
# ---------------------------------------------------------------------------


async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict):
        return JSONResponse(status_code=exc.status_code, content={"error": detail})
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": "HTTP_ERROR", "message": str(detail), "details": {}}},
    )


async def _validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed.",
                "details": {"errors": exc.errors()},
            }
        },
    )


async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_ERROR", "message": str(exc), "details": {}}},
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Focus Group Simulator API",
        version="1.0.0",
        lifespan=lifespan,
        # Disable default 422 detail exposure in production if needed
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Starlette applies middleware LIFO: last registered = outermost layer.
    # AuthMiddleware is added first (innermost); CORSMiddleware is added last (outermost).
    # This ensures CORS headers appear on 401 responses from AuthMiddleware —
    # without this, browsers see a CORS error instead of the actual 401.
    app.add_middleware(AuthMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    app.include_router(api_router, prefix="/api/v1")
    app.include_router(health_router)

    app.add_exception_handler(HTTPException, _http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _global_exception_handler)  # type: ignore[arg-type]

    return app


app = create_app()
