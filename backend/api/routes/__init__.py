"""API route registry.

Import and include sub-routers here as they are built in Phase 2 and 3.
"""

from fastapi import APIRouter

from api.routes.auth import router as auth_router

api_router = APIRouter()
api_router.include_router(auth_router)

# Phase 3: from api.routes.jobs import router as jobs_router; api_router.include_router(...)
# Phase 3: from api.routes.health import router as health_router; api_router.include_router(...)
