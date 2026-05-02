"""API route registry.

Versioned routes (/api/v1/*) are collected into api_router and mounted in main.py.
The health router (/ and /metrics) is registered separately in main.py without a prefix.
"""

from fastapi import APIRouter

from api.routes.auth import router as auth_router
from api.routes.jobs import router as jobs_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(jobs_router)
