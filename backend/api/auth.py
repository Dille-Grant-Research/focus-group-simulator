"""MSAL authentication utilities for Microsoft Entra OIDC.

All MSAL interaction is isolated here — no FastAPI dependencies.
Callers (routes, middleware) import build_auth_url, exchange_code_for_token,
get_session_serializer, SESSION_COOKIE_MAX_AGE, and AuthError.
"""

from __future__ import annotations

import logging

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from msal import ConfidentialClientApplication

from api.config import Settings

logger = logging.getLogger(__name__)

_SCOPES = ["openid", "profile", "email"]
_FLOW_COOKIE_MAX_AGE = 600  # seconds — time window to complete the login redirect
SESSION_COOKIE_MAX_AGE = 86400 * 7  # 7 days — exported for use by routes and middleware

# Module-level singleton. Safe because --workers 1 means single process.
_msal_app: ConfidentialClientApplication | None = None


# ---------------------------------------------------------------------------
# AuthError
# ---------------------------------------------------------------------------


class AuthError(Exception):
    """Raised when authentication cannot proceed. Carries HTTP metadata."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# MSAL singleton
# ---------------------------------------------------------------------------


def get_msal_app(settings: Settings) -> ConfidentialClientApplication:
    global _msal_app
    if _msal_app is None:
        _msal_app = ConfidentialClientApplication(
            client_id=settings.ENTRA_CLIENT_ID,
            client_credential=settings.ENTRA_CLIENT_SECRET,
            authority=settings.entra_authority,
        )
    return _msal_app


# ---------------------------------------------------------------------------
# Cookie serializers
# ---------------------------------------------------------------------------
# Two distinct salts create domain separation: a stolen oauth_flow cookie cannot
# be deserialized as a session cookie even though both use SESSION_SECRET_KEY.


def get_flow_serializer(settings: Settings) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(settings.SESSION_SECRET_KEY, salt="oauth-flow")


def get_session_serializer(settings: Settings) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(settings.SESSION_SECRET_KEY, salt="session")


# ---------------------------------------------------------------------------
# Auth code flow
# ---------------------------------------------------------------------------


def build_auth_url(settings: Settings) -> tuple[str, str]:
    """Initiate an MSAL auth code flow.

    Returns:
        (auth_uri, signed_flow_cookie_value)

    The caller must set the oauth_flow cookie to signed_flow_cookie_value and
    redirect the user to auth_uri.
    """
    msal_app = get_msal_app(settings)
    flow = msal_app.initiate_auth_code_flow(
        scopes=_SCOPES,
        redirect_uri=settings.ENTRA_REDIRECT_URI,
    )
    signed_flow = get_flow_serializer(settings).dumps(flow)
    return flow["auth_uri"], signed_flow


def exchange_code_for_token(
    settings: Settings,
    signed_flow_cookie: str,
    auth_response: dict[str, str],
) -> str:
    """Complete the auth code flow and return the stable user_id (oid claim).

    Args:
        signed_flow_cookie: Raw value from the oauth_flow cookie set by /auth/login.
        auth_response: dict(request.query_params) from the /auth/callback request.

    Returns:
        user_id — the Azure Object ID (oid claim). Stable across email/username changes.

    Raises:
        AuthError on any authentication failure.
    """
    # 1. Deserialize and verify the flow cookie (checks signature + timestamp)
    serializer = get_flow_serializer(settings)
    try:
        flow = serializer.loads(signed_flow_cookie, max_age=_FLOW_COOKIE_MAX_AGE)
    except SignatureExpired:
        raise AuthError(401, "SESSION_EXPIRED", "Login session expired. Please try again.")
    except BadSignature:
        raise AuthError(401, "INVALID_SESSION", "Invalid session state. Please try again.")

    # 2. Exchange code for token (MSAL validates state + nonce internally)
    msal_app = get_msal_app(settings)
    try:
        result = msal_app.acquire_token_by_auth_code_flow(
            auth_code_flow=flow,
            auth_response=auth_response,
        )
    except ValueError:
        # MSAL raises ValueError on CSRF state mismatch or protocol errors
        raise AuthError(401, "AUTH_STATE_MISMATCH", "Security check failed. Please try again.")

    # 3. Check for Entra-reported error
    if "error" in result:
        description = result.get("error_description", result["error"])
        logger.warning("Entra auth error: %s", description)
        raise AuthError(401, "AUTH_FAILED", "Authentication failed. Please try again.")

    # 4. Extract the stable Azure Object ID as user_id
    claims = result.get("id_token_claims") or {}
    user_id = claims.get("oid")
    if not user_id:
        logger.error("id_token_claims missing 'oid' claim. Claims present: %s", list(claims))
        raise AuthError(500, "MISSING_OID_CLAIM", "Identity token is missing required claim.")

    return user_id
