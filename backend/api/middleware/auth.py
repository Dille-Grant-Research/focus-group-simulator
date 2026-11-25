"""Authentication middleware — validates session cookies on every protected request.

Exempt paths (/auth/*, /health, /metrics, /docs, /redoc, /openapi.json) pass through
without authentication. All other paths require a valid signed session cookie and have
request.state.user_id set for downstream route handlers.
"""

from __future__ import annotations

import logging

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from api.auth import AuthError, read_session
from api.config import get_settings
from api.database import get_db, get_sessions_valid_after, upsert_user

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

        # WARNING: DEV_AUTH_BYPASS disables authentication entirely; every request
        # runs as "dev-user-local". Settings refuses to load this flag unless
        # ENTRA_REDIRECT_URI points at localhost. Never remove that guard.
        if settings.DEV_AUTH_BYPASS:
            request.state.user_id = "dev-user-local"
            try:
                async with get_db(settings.DB_PATH) as db:
                    await upsert_user(db, "dev-user-local")
            except Exception:
                logger.exception("DEV_AUTH_BYPASS: failed to upsert dev user")
            return await call_next(request)

        session_cookie = request.cookies.get("session")
        if not session_cookie:
            return _unauthenticated("No session cookie present.")

        try:
            session = read_session(settings, session_cookie)
        except AuthError as exc:
            return _unauthenticated(exc.message)
        user_id = session.user_id

        # Update last_seen on each authenticated request; a failure here is non-fatal.
        # The revocation lookup is not: if it fails, the request errors rather than
        # letting a possibly-revoked session through.
        async with get_db(settings.DB_PATH) as db:
            try:
                await upsert_user(db, user_id)
            except Exception:
                logger.exception("Failed to upsert user %s in auth middleware", user_id)
            valid_after = await get_sessions_valid_after(db, user_id)

        # Cookie timestamps and valid_after are whole seconds. `<=` means a cookie signed in
        # the same second as the logout is rejected; a re-login within that second needs one
        # more round trip (a second later) before its cookie is accepted.
        if valid_after is not None and session.issued_at <= valid_after:
            return _unauthenticated("Session was signed out. Please log in again.")

        request.state.user_id = user_id
        return await call_next(request)


def _unauthenticated(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"error": {"code": "UNAUTHENTICATED", "message": message, "details": {}}},
    )
