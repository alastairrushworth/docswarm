#!/usr/bin/env bash
# Idempotent: re-running on a partially-set-up droplet must succeed.
# Inputs:
#   $REPO_URL, $REPO_BRANCH       env vars
#   "$@"                          Ollama model tags to pre-pull
set -euo pipefail

log() { echo ">>> $*"; }

# 1. Docker (engine + buildx + compose plugin).
# Some DO base images (e.g. gpu-h100x1-base) ship docker-ce alone, so we
# explicitly install the compose plugin whether or not docker is present.
if ! command -v docker >/dev/null 2>&1; then
    log "installing docker"
    curl -fsSL https://get.docker.com | sh
fi
if ! docker compose version >/dev/null 2>&1; then
    log "installing docker compose plugin"
    apt-get update
    apt-get install -y docker-compose-plugin
fi

# 2. NVIDIA Container Toolkit
if ! dpkg -l 2>/dev/null | grep -q nvidia-container-toolkit; then
    log "installing nvidia container toolkit"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update
    apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
fi

# 3. Ollama
if ! command -v ollama >/dev/null 2>&1; then
    log "installing ollama"
    curl -fsSL https://ollama.com/install.sh | sh
fi
systemctl enable --now ollama
# wait for ollama to be ready
for _ in $(seq 1 30); do
    ollama list >/dev/null 2>&1 && break
    sleep 2
done

# 4. Pull models passed as args (dedup; coder/vision/judge often the same tag)
declare -A seen
for model in "$@"; do
    if [[ -z "${seen[$model]:-}" ]]; then
        log "pulling $model"
        ollama pull "$model"
        seen[$model]=1
    fi
done

# 5. Pre-pull base docker images so first `make run` doesn't download them
log "pre-pulling docker base images"
docker pull ollama/ollama:latest >/dev/null
docker pull python:3.11-slim >/dev/null

# 6. SSH config for GitHub (deploy key already at /root/.ssh/id_ed25519)
mkdir -p /root/.ssh
chmod 700 /root/.ssh
if [[ -f /root/.ssh/id_ed25519 ]]; then
    chmod 600 /root/.ssh/id_ed25519
fi
ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null || true
sort -u /root/.ssh/known_hosts -o /root/.ssh/known_hosts

# 7. Clone repo to /workspace (idempotent)
if [[ ! -d /workspace/.git ]]; then
    log "cloning $REPO_URL ($REPO_BRANCH) → /workspace"
    git clone -b "$REPO_BRANCH" "$REPO_URL" /workspace
else
    log "/workspace already a git repo; fetching latest"
    git -C /workspace fetch origin
    git -C /workspace checkout "$REPO_BRANCH"
    git -C /workspace pull --ff-only
fi

log "setup complete"
