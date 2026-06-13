#!/usr/bin/env bash
# One-time setup on a fresh RunPod pod.
# Installs Ollama, Node.js + Claude Code, and Python packages natively
# (no Docker), then clones the repo to /workspace (network volume).
# Idempotent: re-running on a partially-set-up pod succeeds.
# Inputs:
#   $REPO_URL, $REPO_BRANCH   env vars
set -euo pipefail

log() { echo ">>> $*"; }

# 1. Ollama (zstd required by the installer on Ubuntu 22.04)
if ! command -v ollama >/dev/null 2>&1; then
    log "installing ollama"
    DEBIAN_FRONTEND=noninteractive apt-get install -y -q zstd
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# 2. Node.js 20 + Claude Code (developer-agent uses claude CLI)
if ! command -v node >/dev/null 2>&1; then
    log "installing node.js 20"
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    DEBIAN_FRONTEND=noninteractive apt-get install -y -q nodejs
fi
if ! command -v claude >/dev/null 2>&1; then
    log "installing claude code"
    npm install -g @anthropic-ai/claude-code
fi

# 3. Python packages (runpod/pytorch already has numpy/scipy/torch/cuda)
log "installing python packages"
pip install --quiet --upgrade \
    "pydantic>=2.6" "pyyaml>=6.0" "pymupdf>=1.24" \
    "Pillow>=10.0" "httpx>=0.27" "numpy>=1.26" "scipy>=1.11" "pytest>=8.0"

# 4. SSH config for GitHub (deploy key already at /root/.ssh/id_ed25519)
mkdir -p /root/.ssh
chmod 700 /root/.ssh
if [[ -f /root/.ssh/id_ed25519 ]]; then
    chmod 600 /root/.ssh/id_ed25519
fi
ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null || true
sort -u /root/.ssh/known_hosts -o /root/.ssh/known_hosts

# 5. Clone repo to /workspace (idempotent)
if [[ -d /workspace/.git ]]; then
    log "/workspace already a git repo; fetching latest"
    git -C /workspace fetch origin
    git -C /workspace checkout "$REPO_BRANCH"
    git -C /workspace pull --ff-only
else
    log "cloning $REPO_URL ($REPO_BRANCH) → /workspace"
    # git clone refuses a non-empty target; init+fetch handles that case too
    git init /workspace
    git -C /workspace remote add origin "$REPO_URL"
    git -C /workspace fetch origin
    git -C /workspace checkout -B "$REPO_BRANCH" "origin/$REPO_BRANCH"
fi

# 6. Stage deploy key for git push inside the agent
log "staging deploy key at /workspace/secrets/deploy_key"
mkdir -p /workspace/secrets
cp /root/.ssh/id_ed25519 /workspace/secrets/deploy_key
chmod 600 /workspace/secrets/deploy_key

log "setup complete"
