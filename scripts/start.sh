#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
    echo "Usage: $0 [--dev | --dev-ui | --prod]"
    echo ""
    echo "  --dev     Start the API server locally with hot reload (requires .env in backend/)"
    echo "  --dev-ui  Start API + Next.js mock UI (requires .env in backend/, npm deps in frontend/)"
    echo "  --prod    Build and start the Docker container stack"
    exit 1
}

# Kill any stale process holding a port, silently.
clear_port() {
    local port="$1"
    local pids
    pids=$(lsof -ti:"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "Clearing stale process on :${port}..."
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 0.5
    fi
}

case "${1:-}" in
  --dev)
    clear_port 8000
    cd "$REPO_ROOT/backend"
    export PYTHONPATH="$REPO_ROOT/backend"
    exec uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    ;;
  --dev-ui)
    clear_port 8000
    clear_port 3000
    cd "$REPO_ROOT/backend"
    export PYTHONPATH="$REPO_ROOT/backend"
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 &
    UVICORN_PID=$!
    trap 'kill $UVICORN_PID 2>/dev/null; exit' INT TERM
    echo "API server started on :8000 (PID $UVICORN_PID)"
    echo "Starting Next.js dev server on :3000..."
    cd "$REPO_ROOT/frontend"
    npm run dev
    kill $UVICORN_PID 2>/dev/null
    ;;
  --prod)
    exec docker compose -f "$REPO_ROOT/docker/docker-compose.yml" up -d --build
    ;;
  *)
    usage
    ;;
esac
