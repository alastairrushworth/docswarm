#!/usr/bin/env bash
# Single-command launcher. Usage: bash orchestration/run.sh {local|remote}
set -euo pipefail

MODE="${1:-local}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

case "$MODE" in
  local)
    echo ">>> bringing up local stack (plumbing only)"
    docker compose -f docker-compose.yml -f docker-compose.local.yml up --build -d \
        ollama-translator ollama-judge judge
    docker compose -f docker-compose.yml -f docker-compose.local.yml run --rm \
        -e DOCSWARM_MODE=local \
        developer-agent \
        python scripts/run_validation.py --no-git
    docker compose -f docker-compose.yml -f docker-compose.local.yml down
    ;;
  remote)
    echo ">>> remote run (H100 droplet)"
    python orchestration/provision.py up
    DROPLET_IP="$(cat .droplet_ip 2>/dev/null || true)"
    if [[ -z "${DROPLET_IP}" ]]; then
      echo "no droplet IP recorded; aborting"; exit 1
    fi
    trap 'python orchestration/provision.py down' EXIT INT TERM
    BRANCH="$(git rev-parse --abbrev-ref HEAD)"
    echo ">>> pushing $BRANCH"
    git push origin "$BRANCH"
    echo ">>> ssh + bring up on droplet"
    ssh -t -o StrictHostKeyChecking=accept-new "root@${DROPLET_IP}" \
        "cd /workspace && git pull && \
         docker compose -f docker-compose.yml -f docker-compose.remote.yml up --build -d \
            ollama-translator ollama-judge judge && \
         docker compose -f docker-compose.yml -f docker-compose.remote.yml run --rm \
            developer-agent python scripts/run_validation.py"
    ;;
  *)
    echo "usage: $0 {local|remote}"; exit 2
    ;;
esac
