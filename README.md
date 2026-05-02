# AI Focus Group Simulator

A research tool developed at Minnesota State University Moorhead that uses large language models (LLMs) to simulate realistic qualitative focus group sessions. The system generates synthetic participants with distinct personas, facilitates multi-turn discussions around structured research questions, and captures both verbal dialogue and non-verbal behavioral reactions.

Developed under the **Dille-Grant Research Initiative** by the Department of Computer Science and Information Systems.

This is an extension of our previous works, which live at these repos:
- [Phase-1](https://github.com/dille-grant-research/phase-1): Initial research code, proof of concept, and early prototyping
- [Phase-2](https://github.com/dille-grant-research/phase-2): Refined simulation engine, anti-sycophancy mechanisms, and multi-tier memory architecture

---

## Research Purpose

Traditional focus group research is time-intensive and logistically complex. This system is designed to augment or prototype qualitative research workflows by generating realistic focus group transcripts — useful for piloting research instruments, exploring discussion dynamics, and producing synthetic data for analysis before or alongside real sessions.

The simulator's core contributions are:

- **Anti-sycophancy enforcement** — Participants maintain independent stances. A real-time stance tracking system (`StanceMatrix`) detects and prevents artificial consensus drift.
- **Three-tier memory architecture** — Each participant maintains episodic identity memory, cross-question stance memory, and a rolling working memory window — producing contextually coherent, persona-consistent responses across long discussions.
- **LLM-agnostic inference** — The simulation engine is decoupled from any specific model provider, supporting local inference via Ollama and cloud providers (OpenAI, Anthropic) through a common interface.
- **Structured research output** — Every session produces a full transcript, per-turn non-verbal reactions, a stance matrix, and a `ResearchAssistant` analysis of themes, consensus levels, and notable interaction dynamics.

---

## System Overview

The production system consists of three components:

| Component | Technology | Role |
|-----------|-----------|------|
| **Simulation Engine** | Python / asyncio | Generates focus group sessions |
| **API Backend** | FastAPI + SQLite | Job queue, authentication, result storage |
| **Authentication** | Microsoft Entra (Azure AD) | MSUM SSO — any valid Star ID |

The frontend UI is developed and maintained separately by **MSUM IT**. The backend exposes a documented REST API; the frontend communicates exclusively through that contract.

For the full system design, see [`docs/architecture.md`](docs/architecture.md).

---

## For MSUM IT — Quick Start

> This section is for the IT team integrating the backend with the university's frontend application.

### What This Repo Contains

- **`backend/`** — The FastAPI backend, simulation engine, and job worker
- **`docs/api/`** — Complete API reference for building against the backend
- **`docs/deployment.md`** — Step-by-step server deployment guide
- **`docker/`** — Single-command Docker deployment
- **`mock-ui/`** — A minimal reference UI demonstrating the full auth and job flow (for reference only — not production)

### Deployment (3 Steps)

**Prerequisites:** Docker and Docker Compose installed on the server. Port `8000` open.

```bash
# 1. Clone the repository
git clone https://github.com/dille-grant-research/focus-group-simulator.git
cd focus-group-simulator

# 2. Configure environment
cp docker/.env.example docker/.env
# Edit docker/.env — fill in Entra credentials and Ollama host
nano docker/.env

# 3. Start the system
docker compose -f docker/docker-compose.yml up -d

# Verify it's running
curl http://<server-address>:8000/health
```

For the full deployment guide including updates, backups, log access, and troubleshooting, see [`docs/deployment.md`](docs/deployment.md).

### Authentication Setup

This backend uses **Microsoft Entra OIDC** (Authorization Code flow). Before deployment, MSUM IT must register one redirect URI per environment in the Entra app registration:

| Environment | Redirect URI |
|-------------|-------------|
| Local dev | `https://localhost:8000/auth/callback` |
| Production | `https://<your-server>/auth/callback` |

For the complete auth flow, token claims, and how the C# frontend should handle login and 401 responses, see [`docs/api/authentication.md`](docs/api/authentication.md).

### API Documentation

| Document | Contents |
|----------|----------|
| [`docs/api/endpoints.md`](docs/api/endpoints.md) | All endpoints with request/response schemas and examples |
| [`docs/api/authentication.md`](docs/api/authentication.md) | Auth flow, session cookies, 401 handling |
| [`docs/api/error_responses.md`](docs/api/error_responses.md) | All error codes and what the frontend should do for each |
| [`docs/configuration.md`](docs/configuration.md) | `SessionConfig` field reference for building the submission form |

### Key Behavioral Guarantees

- A queued or running job will always reach `completed` or `failed` — no zombie states.
- Results are available for **30 days** after completion, then permanently deleted.
- Users can only access their own jobs — enforced at the database level, not just the route level.
- The API is stateless — the frontend stores no simulation state; everything lives server-side.

---

## Local Development

```bash
# Install backend dependencies
cd backend
pip install -e ".[dev,test]"

# Run the dev server (auto-reload)
uvicorn api.main:app --reload --port 8000

# Run tests
pytest tests/ -q --tb=short

# Run tests with coverage
pytest tests/ --cov=src/focus_group --cov=api --cov-report=term-missing

# Lint and type check
ruff check . && black --check . && mypy src/ api/
```

Or use the setup script:

```bash
bash scripts/setup.sh   # One-time environment setup
bash scripts/start.sh   # Start in dev mode (default)
bash scripts/start.sh prod  # Start via Docker
```

---

## Repository Structure

```
focus-group-simulator/
├── backend/
│   ├── src/focus_group/     # Simulation engine (framework-free, native asyncio)
│   ├── api/                 # FastAPI app, routes, middleware, job worker
│   ├── tests/               # Unit, API, and integration tests
│   └── pyproject.toml
├── mock-ui/                 # Reference UI (HTML/JS) — for IT reference only
├── docs/
│   ├── api/                 # Endpoint reference, auth guide, error codes
│   ├── architecture.md      # Full system design document
│   ├── configuration.md     # SessionConfig field reference
│   └── deployment.md        # MSUM IT deployment runbook
├── docker/                  # Dockerfile, docker-compose.yml, .env.example
└── scripts/                 # setup.sh, start.sh
```

---

## License

Copyright © 2026 Minnesota State University Moorhead, Department of Computer Science and Information Systems. All rights reserved. See [`LICENSE`](LICENSE) for terms.