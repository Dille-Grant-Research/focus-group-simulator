"""Authentication middleware — validates session cookies on every protected request.

Exempt paths (/auth/*, /health, /metrics, /docs, /redoc, /openapi.json) pass through
without authentication. All other paths require a valid signed session cookie and have
request.state.user_id set for downstream route handlers.
"""

from __future__ import annotations

import logging

from itsdangerous import BadSignature, SignatureExpired
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from api.auth import SESSION_COOKIE_MAX_AGE, get_session_serializer
from api.config import get_settings
from api.database import get_db, upsert_user

logger = logging.getLogger(__name__)

_EXEMPT_PREFIXES = (
    "/api/v1/auth/",
    "/health",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path

        if any(path.startswith(prefix) for prefix in _EXEMPT_PREFIXES):
            return await call_next(request)

        settings = get_settings()
        session_cookie = request.cookies.get("session")
        if not session_cookie:
            return _unauthenticated("No session cookie present.")

        serializer = get_session_serializer(settings)
        try:
            payload = serializer.loads(session_cookie, max_age=SESSION_COOKIE_MAX_AGE)
        except SignatureExpired:
            return _unauthenticated("Session expired. Please log in again.")
        except BadSignature:
            return _unauthenticated("Invalid session cookie.")

        user_id = payload.get("user_id")
        if not user_id:
            return _unauthenticated("Malformed session payload.")

        # Update last_seen on each authenticated request; a failure here is non-fatal.
        try:
            async with get_db(settings.DB_PATH) as db:
                await upsert_user(db, user_id)
        except Exception:
            logger.exception("Failed to upsert user %s in auth middleware", user_id)

        request.state.user_id = user_id
        return await call_next(request)


def _unauthenticated(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"error": {"code": "UNAUTHENTICATED", "message": message, "details": {}}},
    )
