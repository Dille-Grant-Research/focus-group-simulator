"""Rate limiting helper for job submission.

The count-and-insert is atomic in api.database.create_job_within_limit; this module only
builds the 429 error. Plain function (not ASGI middleware) so the POST /jobs handler can
check idempotency first — duplicate submissions bypass the limit and return the existing job.
"""

from fastapi import HTTPException

from api.config import Settings

RATE_LIMIT_WINDOW_HOURS = 1


def rate_limit_exceeded(settings: Settings) -> HTTPException:
    """Build the 429 raised when the user has hit the per-hour job submission cap."""
    return HTTPException(
        status_code=429,
        detail={
            "code": "RATE_LIMIT_EXCEEDED",
            "message": (
                f"Maximum {settings.MAX_JOBS_PER_HOUR} simulations per hour. "
                "Try again later or contact an administrator to raise your limit."
            ),
            "details": {
                "limit": settings.MAX_JOBS_PER_HOUR,
                "window_hours": RATE_LIMIT_WINDOW_HOURS,
            },
        },
    )
