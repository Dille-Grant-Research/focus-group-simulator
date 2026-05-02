"""Authentication routes — OIDC Authorization Code Flow with Microsoft Entra.

Routes mount under /api/v1/auth/* (prefix stacks with the /api/v1 mount in main.py).
"""

from __future__ import annotations

import logging
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse, Response

from api.auth import (
    SESSION_COOKIE_MAX_AGE,
    AuthError,
    build_auth_url,
    exchange_code_for_token,
    get_session_serializer,
)
from api.config import Settings, get_settings
from api.database import get_db, upsert_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


# ---------------------------------------------------------------------------
# GET /auth/login
# ---------------------------------------------------------------------------


@router.get("/login")
async def login(settings: Settings = Depends(get_settings)) -> Response:
    auth_uri, signed_flow = build_auth_url(settings)
    response = RedirectResponse(url=auth_uri, status_code=302)
    response.set_cookie(
        key="oauth_flow",
        value=signed_flow,
        httponly=True,
        secure=True,
        samesite="lax",  # "strict" would drop the cookie on the Entra callback redirect
        max_age=600,
        path="/api/v1/auth",  # scoped to /auth only — not sent on API calls
    )
    return response


# ---------------------------------------------------------------------------
# GET /auth/callback
# ---------------------------------------------------------------------------


@router.get("/callback")
async def callback(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> Response:
    signed_flow = request.cookies.get("oauth_flow")
    if not signed_flow:
        return _auth_error_redirect(settings, "Login state missing. Please try again.")

    try:
        user_id = exchange_code_for_token(
            settings=settings,
            signed_flow_cookie=signed_flow,
            auth_response=dict(request.query_params),
        )
    except AuthError as exc:
        logger.warning("Auth callback failed [%s]: %s", exc.code, exc.message)
        return _auth_error_redirect(settings, exc.message)

    async with get_db(settings.DB_PATH) as db:
        await upsert_user(db, user_id)

    session_value = get_session_serializer(settings).dumps({"user_id": user_id})

    response = RedirectResponse(url=settings.FRONTEND_URL, status_code=302)
    response.set_cookie(
        key="session",
        value=session_value,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=SESSION_COOKIE_MAX_AGE,
        path="/",
    )
    # Clean up the transient flow cookie
    response.delete_cookie(key="oauth_flow", path="/api/v1/auth")
    return response


# ---------------------------------------------------------------------------
# GET /auth/logout
# ---------------------------------------------------------------------------


@router.get("/logout")
async def logout(settings: Settings = Depends(get_settings)) -> Response:
    response = RedirectResponse(url=settings.FRONTEND_URL, status_code=302)
    response.delete_cookie(key="session", path="/")
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auth_error_redirect(settings: Settings, message: str) -> RedirectResponse:
    """Redirect to the frontend with an auth_error query param.

    Used instead of a JSON 401 because the browser is mid-redirect during the
    OAuth dance — a JSON response would produce a blank page, not an error UI.
    The redirect destination is always FRONTEND_URL (server config), never a
    user-supplied URL, preventing open redirect attacks.
    """
    params = urlencode({"auth_error": message})
    return RedirectResponse(url=f"{settings.FRONTEND_URL}?{params}", status_code=302)
