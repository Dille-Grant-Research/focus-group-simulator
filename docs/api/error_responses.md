# Error Response Reference

All non-2xx responses from the API follow a standard error envelope. This document covers every error code the API can return and the recommended handling for each.

---

## Error Envelope Schema

Every error response has this shape:

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Maximum 5 simulations per hour. Try again later or contact an administrator to raise your limit.",
    "details": {
      "limit": 5,
      "window_hours": 1
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `error.code` | string | Machine-readable error identifier. Use this for branching logic, not `error.message`. |
| `error.message` | string | Human-readable explanation. Safe to display to end users. |
| `error.details` | object | Additional structured context. May be empty `{}` or contain field-level information. |

> **Note for C# developers:** Always branch on `error.code`, not on the HTTP status code alone. Two different codes can share the same HTTP status (e.g., `409` is used for `DUPLICATE_SUBMISSION`, `JOB_NOT_COMPLETE`, and `JOB_RUNNING`).

---

## Error Code Reference

### 400 — INVALID_CONFIG

**When:** The `SessionConfig` body passes Pydantic type validation but fails a business-logic constraint (e.g., `demographics` list is empty after stripping whitespace).

**Response:**
```json
{
  "error": {
    "code": "INVALID_CONFIG",
    "message": "Session configuration is invalid.",
    "details": {}
  }
}
```

**Recommended handling:** Show the user a validation error message. Do not retry automatically.

---

### 401 — UNAUTHENTICATED

**When:** The `session` cookie is missing, expired, has an invalid signature, or was signed out (session revoked by logout). A signed-out session returns the message "Session was signed out. Please log in again."

**Response:**
```json
{
  "error": {
    "code": "UNAUTHENTICATED",
    "message": "Session expired. Please log in again.",
    "details": {}
  }
}
```

**Recommended handling:** Redirect the user to `GET /api/v1/auth/login`. Do not attempt a silent retry — the session must be re-established through the full login flow.

---

### 403 — FORBIDDEN

**When:** The authenticated user attempts to access a job that belongs to a different user.

**Response:**
```json
{
  "error": {
    "code": "FORBIDDEN",
    "message": "You do not have permission to access this job.",
    "details": {}
  }
}
```

**Recommended handling:** Treat this identically to `404 JOB_NOT_FOUND` in the UI. Showing a generic "not found" message prevents information leakage about other users' job IDs.

---

### 404 — JOB_NOT_FOUND

**When:** The requested `job_id` does not exist, belongs to another user, or the 30-day retention window has expired and the job was purged.

**Response:**
```json
{
  "error": {
    "code": "JOB_NOT_FOUND",
    "message": "Job not found.",
    "details": {}
  }
}
```

**Recommended handling:** Show the user a "no results found" state. If they bookmarked the URL, explain that results are only available for 30 days.

---

### 409 — DUPLICATE_SUBMISSION

**When:** The exact same `SessionConfig` was submitted by the same user within the 10-minute idempotency window and the previous job did not fail. The existing job is returned instead of creating a new one.

**Response:**
```json
{
  "error": {
    "code": "DUPLICATE_SUBMISSION",
    "message": "A job with this configuration already exists.",
    "details": {
      "existing_job_id": "3f8a2b1c-e7d4-4a2f-9b1e-0c5d6f7a8b9e"
    }
  }
}
```

**Recommended handling:** Read `details.existing_job_id` and redirect the user to the status page for that job. This is not an error condition — it protects against accidental double-submission.

---

### 409 — JOB_NOT_COMPLETE

**When:** `GET /api/v1/jobs/{job_id}/result` is called while the job is still `queued` or `running`.

**Response:**
```json
{
  "error": {
    "code": "JOB_NOT_COMPLETE",
    "message": "Job is not yet complete.",
    "details": {}
  }
}
```

**Recommended handling:** Continue polling `GET /api/v1/jobs/{job_id}/status`. Only call `/result` once `status` is `"completed"`.

---

### 409 — JOB_RUNNING

**When:** `DELETE /api/v1/jobs/{job_id}` is called while the job is in `running` status.

**Response:**
```json
{
  "error": {
    "code": "JOB_RUNNING",
    "message": "Cannot delete a running job.",
    "details": {}
  }
}
```

**Recommended handling:** Inform the user that the simulation is in progress and cannot be cancelled mid-run. Offer to delete it after it completes or fails.

---

### 422 — VALIDATION_ERROR

**When:** The request body fails Pydantic field-level validation (wrong type, out-of-range value, missing required field). This is generated automatically by FastAPI.

**Response:**
```json
{
  "detail": [
    {
      "type": "too_short",
      "loc": ["body", "demographics"],
      "msg": "List should have at least 1 item after validation, not 0",
      "input": [],
      "ctx": {"min_length": 1}
    }
  ]
}
```

> **Note:** `422` responses use FastAPI's native `detail` array format, not the standard `error` envelope. Parse `detail[*].loc` to identify which field failed and `detail[*].msg` for the error message.

**Recommended handling:** Map field paths from `loc` to the corresponding form fields and display inline validation errors to the user.

---

### 429 — RATE_LIMIT_EXCEEDED

**When:** The user has submitted `MAX_JOBS_PER_HOUR` or more jobs in the past hour (default: 5). Every accepted submission in the last hour counts, including jobs later deleted or failed. Duplicate submissions that hit the idempotency check do not count toward this limit.

**Response:**
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Maximum 5 simulations per hour. Try again later or contact an administrator to raise your limit.",
    "details": {
      "limit": 5,
      "window_hours": 1
    }
  }
}
```

**Recommended handling:** Disable the submit button and display a message like "You have reached your hourly limit of 5 simulations. You can submit again in less than 1 hour." The limit can be raised by changing `MAX_JOBS_PER_HOUR` in `.env` — contact IT if the default is too restrictive for your research session.

---

### 500 — INTERNAL_ERROR

**When:** An unexpected server-side error occurred.

**Response:**
```json
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "An unexpected error occurred.",
    "details": {}
  }
}
```

**Recommended handling:** Log the error client-side (include `job_id` if available), show a generic error message to the user, and notify IT. Do not expose raw error details to end users.

---

### 503 — SERVICE_UNAVAILABLE

**When:** `POST /api/v1/generate` cannot reach the inference server. (`GET /health` also returns HTTP 503 when a component is down, but with its own body: `status: "degraded"` and a `components` map. See endpoints.md.) Job submission does not return 503: jobs queue normally and a job that cannot reach the inference server ends as `failed`.

**Response:**
```json
{
  "error": {
    "code": "SERVICE_UNAVAILABLE",
    "message": "Inference service is not reachable. Try again in a moment.",
    "details": {}
  }
}
```

**Recommended handling:** Show a maintenance banner. Retry the operation after a delay (suggest 60 seconds). If the error persists, contact IT — it usually means the inference server is unreachable (`llm` component `unreachable` in `/health`) or the Docker container crashed (`worker` component `dead`).

---

## Quick Reference Table

| HTTP Status | Code | Retry? | Recommended Action |
|-------------|------|--------|--------------------|
| 400 | `INVALID_CONFIG` | No | Show validation error to user |
| 401 | `UNAUTHENTICATED` | No | Redirect to `/api/v1/auth/login` |
| 403 | `FORBIDDEN` | No | Show "not found" (security — don't reveal ownership) |
| 404 | `JOB_NOT_FOUND` | No | Show "no results" or "expired" state |
| 409 | `DUPLICATE_SUBMISSION` | No | Use `details.existing_job_id` to navigate to existing job |
| 409 | `JOB_NOT_COMPLETE` | Yes (keep polling) | Continue polling `/status` |
| 409 | `JOB_RUNNING` | No | Inform user; offer to delete after completion |
| 422 | `VALIDATION_ERROR` | No | Show inline field errors using `detail[*].loc` |
| 429 | `RATE_LIMIT_EXCEEDED` | Yes (after 1 hour) | Show hourly limit message |
| 500 | `INTERNAL_ERROR` | Maybe | Log, notify IT, show generic error |
| 503 | `SERVICE_UNAVAILABLE` | Yes (after 60s) | Show maintenance banner; notify IT if persistent |
