#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
    echo "Usage: $0 [--dev | --prod]"
    echo ""
    echo "  --dev   Start the API server locally with hot reload (requires .env in backend/)"
    echo "  --prod  Build and start the Docker container stack"
    exit 1
}

case "${1:-}" in
  --dev)
    cd "$REPO_ROOT/backend"
    export PYTHONPATH="$REPO_ROOT/backend"
    exec uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    ;;
  --prod)
    exec docker compose -f "$REPO_ROOT/docker/docker-compose.yml" up -d --build
    ;;
  *)
    usage
    ;;
esac
