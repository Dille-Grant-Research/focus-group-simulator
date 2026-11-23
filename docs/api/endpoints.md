# API Endpoint Reference

**Base URL:** `http(s)://<server>:8000`

All job endpoints are prefixed with `/api/v1`. Auth endpoints are prefixed with `/api/v1/auth`. Health and metrics endpoints are at the root.

All authenticated endpoints require the signed `session` cookie set by the login flow. See [authentication.md](authentication.md) for details.

---

## Table of Contents

1. [Job Endpoints](#1-job-endpoints)
   - [POST /api/v1/jobs — Submit a simulation](#post-apiv1jobs)
   - [GET /api/v1/jobs/{job_id}/status — Poll status](#get-apiv1jobsjob_idstatus)
   - [GET /api/v1/jobs/{job_id}/result — Fetch result](#get-apiv1jobsjob_idresult)
   - [GET /api/v1/jobs — List jobs](#get-apiv1jobs)
   - [DELETE /api/v1/jobs/{job_id} — Delete a job](#delete-apiv1jobsjob_id)
2. [Auth Endpoints](#2-auth-endpoints)
   - [GET /api/v1/auth/login](#get-apiv1authlogin)
   - [GET /api/v1/auth/callback](#get-apiv1authcallback)
   - [GET /api/v1/auth/logout](#get-apiv1authlogout)
3. [System Endpoints](#3-system-endpoints)
   - [GET /health](#get-health)
   - [GET /metrics](#get-metrics)

---

## 1. Job Endpoints

### POST /api/v1/jobs

Submit a new focus group simulation for processing.

**Auth required:** Yes

**Idempotency:** Submitting the same configuration within 10 minutes returns the existing `job_id` instead of creating a new job. This prevents duplicate simulations from double-clicks or retries.

**Rate limit:** `MAX_JOBS_PER_HOUR` submissions per user per rolling hour (default: 5). Duplicate submissions (idempotency hits) do not count against the limit.

#### Request

```
POST /api/v1/jobs
Content-Type: application/json
Cookie: session=<value>
```

**Body:** A `SessionConfig` object. See [configuration.md](../configuration.md) for the full field reference.

Minimal example:
```json
{
  "demographics": [
    "Female, age 34, suburban Minnesota, works in healthcare",
    "Male, age 52, rural Minnesota, retired farmer"
  ],
  "questions": [
    "What are your biggest concerns about adopting telehealth services?"
  ]
}
```

#### Response — 202 Accepted

```json
{
  "job_id": "3f8a2b1c-e7d4-4a2f-9b1e-0c5d6f7a8b9e",
  "status": "queued",
  "estimated_position": 1
}
```

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | UUID for this simulation. Store this to poll status and fetch the result. |
| `status` | string | Always `"queued"` on initial submission. |
| `estimated_position` | integer or null | Position in the queue (1 = next to run). |

#### Error Cases

| Status | Code | When |
|--------|------|------|
| 400 | `INVALID_CONFIG` | `SessionConfig` fields fail business-logic validation |
| 409 | `DUPLICATE_SUBMISSION` | Identical config submitted within the idempotency window |
| 422 | `VALIDATION_ERROR` | Pydantic field type or constraint violation |
| 429 | `RATE_LIMIT_EXCEEDED` | Hourly submission cap reached |

---

### GET /api/v1/jobs/{job_id}/status

Poll the current status of a submitted job.

**Auth required:** Yes — users can only poll their own jobs.

**Recommended poll intervals:**

| Status | Interval |
|--------|----------|
| `queued` | Every 10 seconds |
| `running` | Every 5 seconds |
| `completed` or `failed` | Stop polling; fetch result or show error |

#### Request

```
GET /api/v1/jobs/3f8a2b1c-e7d4-4a2f-9b1e-0c5d6f7a8b9e/status
Cookie: session=<value>
```

#### Response — 200 OK (queued)

```json
{
  "job_id": "3f8a2b1c-e7d4-4a2f-9b1e-0c5d6f7a8b9e",
  "status": "queued",
  "created_at": "2026-05-02T14:00:00Z",
  "started_at": null,
  "completed_at": null,
  "expires_at": "2026-06-01T14:00:00Z",
  "queue_position": 2
}
```

#### Response — 200 OK (running)

```json
{
  "job_id": "3f8a2b1c-e7d4-4a2f-9b1e-0c5d6f7a8b9e",
  "status": "running",
  "created_at": "2026-05-02T14:00:00Z",
  "started_at": "2026-05-02T14:01:30Z",
  "completed_at": null,
  "expires_at": "2026-06-01T14:00:00Z",
  "queue_position": null
}
```

#### Response — 200 OK (completed)

```json
{
  "job_id": "3f8a2b1c-e7d4-4a2f-9b1e-0c5d6f7a8b9e",
  "status": "completed",
  "created_at": "2026-05-02T14:00:00Z",
  "started_at": "2026-05-02T14:01:30Z",
  "completed_at": "2026-05-02T14:07:45Z",
  "expires_at": "2026-06-01T14:00:00Z",
  "queue_position": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | Job identifier |
| `status` | string | `queued`, `running`, `completed`, or `failed` |
| `created_at` | string | ISO 8601 timestamp — when job was submitted |
| `started_at` | string or null | ISO 8601 timestamp — when the worker picked it up |
| `completed_at` | string or null | ISO 8601 timestamp — when simulation finished |
| `expires_at` | string | ISO 8601 timestamp — job and result deleted after this date (30 days) |
| `queue_position` | integer or null | Position in the queue if `queued`; null otherwise |

#### Error Cases

| Status | Code | When |
|--------|------|------|
| 404 | `JOB_NOT_FOUND` | Job does not exist, belongs to another user, or has expired |

---

### GET /api/v1/jobs/{job_id}/result

Fetch the full simulation transcript and analysis once a job is `completed`.

**Auth required:** Yes — users can only fetch their own results.

#### Request

```
GET /api/v1/jobs/3f8a2b1c-e7d4-4a2f-9b1e-0c5d6f7a8b9e/result
Cookie: session=<value>
```

#### Response — 200 OK

```json
{
  "job_id": "3f8a2b1c-e7d4-4a2f-9b1e-0c5d6f7a8b9e",
  "status": "completed",
  "config": {
    "demographics": [
      "Female, age 34, suburban Minnesota, works in healthcare",
      "Male, age 52, rural Minnesota, retired farmer"
    ],
    "questions": [
      "What are your biggest concerns about adopting telehealth services?"
    ],
    "purpose": null,
    "session_name": "Telehealth Study Pilot"
  },
  "result": {
    "transcript": [
      {
        "turn": 1,
        "speaker": "Participant 1",
        "verbal": "My main concern is the reliability of the internet connection. Out here we have spotty service and I'd hate to lose connection mid-appointment.",
        "nonverbals": {
          "Participant 2": "nods slowly, leans forward with hands clasped"
        },
        "timestamp": "2026-05-02T14:02:10Z"
      }
    ],
    "analysis": {
      "stance_matrix": {
        "Participant 1": {"telehealth_adoption": "skeptical"},
        "Participant 2": {"telehealth_adoption": "cautiously_supportive"}
      },
      "themes": [
        "internet reliability concerns",
        "trust in technology for healthcare",
        "rural vs suburban access disparities"
      ],
      "consensus_level": "low",
      "notable_interactions": [
        "Participant 2 initially agreed with Participant 1 but shifted stance when asked about specialist access"
      ]
    }
  },
  "metadata": {
    "participant_count": 2,
    "total_turns": 14,
    "consensus_level": "low",
    "created_at": "2026-05-02T14:00:00Z",
    "started_at": "2026-05-02T14:01:30Z",
    "completed_at": "2026-05-02T14:07:45Z",
    "expires_at": "2026-06-01T14:00:00Z"
  }
}
```

**Note:** The `result` object contains the raw simulation output. Shape may vary depending on whether memory/stance tracking was enabled in the session config.

#### Error Cases

| Status | Code | When |
|--------|------|------|
| 404 | `JOB_NOT_FOUND` | Job does not exist, belongs to another user, or has expired |
| 409 | `JOB_NOT_COMPLETE` | Job is still `queued` or `running`; continue polling `/status` |

---

### GET /api/v1/jobs

List all jobs for the authenticated user, most recent first.

**Auth required:** Yes

#### Request

```
GET /api/v1/jobs
Cookie: session=<value>
```

#### Response — 200 OK

```json
[
  {
    "job_id": "3f8a2b1c-e7d4-4a2f-9b1e-0c5d6f7a8b9e",
    "status": "completed",
    "created_at": "2026-05-02T14:00:00Z",
    "expires_at": "2026-06-01T14:00:00Z",
    "participant_count": 2,
    "total_turns": 14,
    "consensus_level": "low"
  },
  {
    "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "status": "queued",
    "created_at": "2026-05-02T15:30:00Z",
    "expires_at": "2026-06-01T15:30:00Z",
    "participant_count": null,
    "total_turns": null,
    "consensus_level": null
  }
]
```

Summary fields (`participant_count`, `total_turns`, `consensus_level`) are `null` for jobs that have not yet completed.

Returns an empty array `[]` if the user has no jobs.

---

### DELETE /api/v1/jobs/{job_id}

Delete a job and permanently remove its result.

**Auth required:** Yes — users can only delete their own jobs.

**Constraint:** A job in `running` status cannot be deleted. Wait for it to complete or fail first.

#### Request

```
DELETE /api/v1/jobs/3f8a2b1c-e7d4-4a2f-9b1e-0c5d6f7a8b9e
Cookie: session=<value>
```

#### Response — 204 No Content

Empty body on success.

#### Error Cases

| Status | Code | When |
|--------|------|------|
| 404 | `JOB_NOT_FOUND` | Job does not exist or belongs to another user |
| 409 | `JOB_RUNNING` | Job is currently running; deletion not allowed |

---

## 2. Auth Endpoints

### GET /api/v1/auth/login

Initiates the Microsoft Entra OIDC login flow.

**Auth required:** No

**Usage:** Redirect the user's browser to this URL. Do not call it via `fetch` or `XMLHttpRequest` — the response is a browser redirect.

```
GET /api/v1/auth/login
```

**Response:** `302` redirect to the Microsoft Entra login page.

---

### GET /api/v1/auth/callback

The OAuth callback endpoint. Microsoft Entra redirects here after the user authenticates. This endpoint is called by Entra automatically — you do not call it directly.

**Auth required:** No

**Response:** `302` redirect to `FRONTEND_URL` with the `session` cookie set on success, or `302` redirect to `FRONTEND_URL?auth_error=<message>` on failure.

---

### GET /api/v1/auth/logout

Logs the user out by clearing the session cookie.

**Auth required:** No (but has no effect if the user is not logged in)

```
GET /api/v1/auth/logout
```

**Response:** `302` redirect to `FRONTEND_URL` with the session cookie deleted.

---

## 3. System Endpoints

### GET /health

Deep health check verifying all critical system components. Used by Docker's built-in `healthcheck` directive and suitable for external uptime monitoring.

**Auth required:** No

#### Response — 200 OK (all healthy)

```json
{
  "status": "healthy",
  "components": {
    "database": "ok",
    "worker": "alive",
    "llm": "reachable"
  },
  "timestamp": "2026-05-02T14:00:00Z"
}
```

#### Response — 503 Service Unavailable (degraded)

```json
{
  "status": "degraded",
  "components": {
    "database": "ok",
    "worker": "alive",
    "llm": "unreachable"
  },
  "timestamp": "2026-05-02T14:00:00Z"
}
```

| Component | Values | What it checks |
|-----------|--------|---------------|
| `database` | `"ok"` / `"error"` | SQLite responds to `SELECT 1` |
| `worker` | `"alive"` / `"dead"` | Background job worker task is running |
| `llm` | `"reachable"` / `"unreachable"` / `"error"` | Inference service is up and serves `LLM_MODEL`. `LLM_PROVIDER=openai`: `GET <LLM_BASE_URL>/v1/models` lists the model. `LLM_PROVIDER=ollama`: `GET <LLM_BASE_URL>/api/tags` returns 200 |

Returns `503` if **any** component is not in its healthy state. Returns `200` only when all three are healthy.

---

### GET /metrics

Operational statistics for the job queue. Always returns 200.

**Auth required:** No

#### Response — 200 OK

```json
{
  "queue_depth": 2,
  "jobs_completed_24h": 7,
  "jobs_failed_24h": 1,
  "avg_duration_seconds": 287.4,
  "worker_status": "processing",
  "current_job_id": "3f8a2b1c-e7d4-4a2f-9b1e-0c5d6f7a8b9e"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `queue_depth` | integer | Number of jobs currently in `queued` status |
| `jobs_completed_24h` | integer | Jobs that reached `completed` in the last 24 hours |
| `jobs_failed_24h` | integer | Jobs that reached `failed` in the last 24 hours |
| `avg_duration_seconds` | float or null | Average runtime of completed jobs in the last 24 hours |
| `worker_status` | string | `"dead"`, `"idle"`, or `"processing"` |
| `current_job_id` | string or null | Job ID currently being processed; null if idle |
