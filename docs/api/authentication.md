# Authentication Guide

**Audience:** MSUM IT team and the C# frontend developers.

This document covers everything required to wire up authentication between the C# frontend, the FastAPI backend, and Microsoft Entra ID.

---

## Overview

The system uses the **OAuth 2.0 Authorization Code Flow** with Microsoft Entra ID (formerly Azure Active Directory). The backend owns the token exchange entirely — the C# frontend never sees or handles Entra tokens directly. It only sends a signed session cookie that the backend issues after a successful login.

---

## Flow Diagram

```
C# Frontend                  FastAPI Backend              Microsoft Entra
─────────────                ─────────────────            ───────────────
User clicks "Login"
    │
    ▼
GET /api/v1/auth/login ──────► Generates auth URL
                               Sets oauth_flow cookie
                               302 → Entra login URL ──────► Entra login page
                                                             User authenticates
                                                             302 → /api/v1/auth/callback?code=...
                                                                           │
GET /api/v1/auth/callback ◄───────────────────────────────────────────────┘
                               Validates oauth_flow cookie
                               Exchanges code for token ──► Token endpoint
                               Extracts user_id (oid claim)
                               Upserts user in SQLite
                               Sets signed session cookie
                               302 → FRONTEND_URL
    │
    ◄── Browser lands back on frontend with session cookie set
    │
All subsequent API calls send session cookie automatically (browser handles it)
    │
GET /api/v1/jobs ─────────────► AuthMiddleware validates cookie
                                Attaches user_id to request
                                Route handler runs normally
```

---

## What MSUM IT Must Register in Entra

### Step 1 — Create the App Registration

1. Sign in to the [Azure portal](https://portal.azure.com)
2. Navigate to **Microsoft Entra ID → App registrations → New registration**
3. Set:
   - **Name:** `Focus Group Simulator` (or any descriptive name)
   - **Supported account types:** `Accounts in this organizational directory only (MSUM only — Single tenant)`
   - **Redirect URI:** Leave blank for now (set in Step 2)

### Step 2 — Register the Redirect URI

The redirect URI is the exact URL Entra will send the authorization code to after a user authenticates. It must **exactly match** the value of `ENTRA_REDIRECT_URI` in the backend's `.env` file.

Navigate to **App registrations → [your app] → Authentication → Add a platform → Web** and enter:

| Environment | Redirect URI |
|-------------|-------------|
| Production | `https://your-server.mnstate.edu/api/v1/auth/callback` |
| Local development | `http://localhost:8000/api/v1/auth/callback` |

> You can register both URIs and control which one is active via the `ENTRA_REDIRECT_URI` environment variable.

### Step 3 — Generate a Client Secret

Navigate to **App registrations → [your app] → Certificates & secrets → New client secret**.

Copy the secret **value** immediately — it is only shown once. This goes into `ENTRA_CLIENT_SECRET` in `.env`.

### Step 4 — Collect the Required Values

From the app registration overview page, copy:

| Value | Where to find it | .env variable |
|-------|-----------------|---------------|
| Directory (tenant) ID | Overview → Directory (tenant) ID | `ENTRA_TENANT_ID` |
| Application (client) ID | Overview → Application (client) ID | `ENTRA_CLIENT_ID` |
| Client secret value | Certificates & secrets (Step 3) | `ENTRA_CLIENT_SECRET` |

### Step 5 — Verify Token Claims

The backend uses the `oid` (Object ID) claim from the Entra ID token as the stable user identifier. This claim is present by default in all Entra ID tokens for single-tenant apps — no extra configuration is needed.

> **Why `oid` and not `upn` or email?** The `oid` is a permanent, immutable identifier tied to the user's account in the directory. It survives username changes, email changes, and department moves.

---

## Session Cookie Behavior

After a successful login, the backend sets a single `session` cookie:

| Property | Value |
|----------|-------|
| Name | `session` |
| Contents | Cryptographically signed payload containing `user_id` |
| `HttpOnly` | Yes — JavaScript cannot read it |
| `Secure` | Yes — HTTPS only |
| `SameSite` | `Lax` — sent on same-site requests and top-level cross-site navigations (GET) |
| `Path` | `/` — sent on all requests to the backend |
| Max age | `SESSION_MAX_AGE_HOURS` (default 12 hours, 43,200 seconds) |

The cookie is signed using `SESSION_SECRET_KEY`. The session lasts `SESSION_MAX_AGE_HOURS` (default 12 hours). If the signature is invalid or the cookie has expired, the backend returns `401 UNAUTHENTICATED`.

---

## C# Frontend Integration

### Login

Redirect the user's browser to the backend login endpoint. Do not try to construct the Entra URL yourself — the backend handles it.

```
GET /api/v1/auth/login
```

The backend redirects the browser to Entra, and after authentication, Entra redirects back through the backend callback, which sets the session cookie and redirects the browser to `FRONTEND_URL`.

### Detecting Login Failure

If the OAuth dance fails for any reason (invalid state, Entra error, etc.), the backend redirects to `FRONTEND_URL?auth_error=<message>`. Check for the `auth_error` query parameter on the landing page and show the user an appropriate message.

### Making Authenticated Requests

Once the session cookie is set, the browser sends it automatically with every request to the backend origin. No extra headers are required for browser-initiated requests.

For server-side or programmatic requests (e.g., from a C# backend making API calls on behalf of a user), include the session cookie in the `Cookie` header:

```
Cookie: session=<value>
```

### Handling 401 Responses

When any API call returns `401 UNAUTHENTICATED`, the session has expired or is missing. Redirect the user to `GET /api/v1/auth/login` to re-authenticate.

Do **not** attempt to refresh the token client-side — the backend manages the full token lifecycle.

```
HTTP 401 → redirect to /api/v1/auth/login
```

### Logout

```
POST /api/v1/auth/logout
```

The backend revokes every session issued to the user so far (on all devices), deletes the session cookie and responds `303` to `FRONTEND_URL`. No request body is required. Use a `POST` form or `fetch`; `GET` returns `405`. A later login issues a fresh, valid session.

---

## Exempt Endpoints

The following paths do **not** require authentication and will never return `401`:

| Path | Purpose |
|------|---------|
| `/api/v1/auth/*` | Login, callback, logout |
| `/health` | Docker health check |
| `/metrics` | Operational metrics |
| `/docs` | OpenAPI documentation UI (404 unless `API_DOCS_ENABLED=true`) |
| `/redoc` | ReDoc documentation UI (404 unless `API_DOCS_ENABLED=true`) |
| `/openapi.json` | OpenAPI schema (404 unless `API_DOCS_ENABLED=true`) |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| All API requests return `401` immediately after deploy | `SESSION_SECRET_KEY` not set or changed | Verify `SESSION_SECRET_KEY` in `.env` is non-empty and consistent across restarts |
| Login redirects to Entra but callback returns to frontend with `auth_error` | Redirect URI mismatch | Ensure `ENTRA_REDIRECT_URI` in `.env` exactly matches the URI registered in Entra |
| `401` after `SESSION_MAX_AGE_HOURS` (default 12 hours) | Session cookie expired (by design) | This is expected behavior — the user must log in again |
| Login works locally but not in production | HTTP vs HTTPS cookie | The `Secure` flag requires HTTPS in production; the server must be behind TLS |
