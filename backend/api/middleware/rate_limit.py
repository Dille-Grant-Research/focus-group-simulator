"""Rate limiting helper for job submission.

Implemented as a plain async function (not ASGI middleware) so that the
POST /jobs route handler can check idempotency first — duplicate submissions
bypass the rate limit and return the existing job regardless of quota.
"""

import aiosqlite
from fastapi import HTTPException

from api.config import Settings
from api.database import count_user_jobs_in_window


async def check_rate_limit(
    db: aiosqlite.Connection,
    user_id: str,
    settings: Settings,
) -> None:
    """Raise 429 if the user has hit the per-hour job submission cap."""
    count = await count_user_jobs_in_window(db, user_id, hours=1)
    if count >= settings.MAX_JOBS_PER_HOUR:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "RATE_LIMIT_EXCEEDED",
                "message": (
                    f"Maximum {settings.MAX_JOBS_PER_HOUR} simulations per hour. "
                    "Try again later or contact an administrator to raise your limit."
                ),
                "details": {"limit": settings.MAX_JOBS_PER_HOUR, "window_hours": 1},
            },
        )
