FROM python:3.11-slim

# System tools + Node (for Claude Code).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        openssh-client \
        gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Claude Code CLI — points at local Ollama via ANTHROPIC_BASE_URL set by compose.
RUN npm install -g @anthropic-ai/claude-code

WORKDIR /workspace

RUN pip install --no-cache-dir \
    pydantic>=2.6 \
    pyyaml>=6.0 \
    pymupdf>=1.24 \
    Pillow>=10.0 \
    httpx>=0.27 \
    numpy>=1.26 \
    scipy>=1.11 \
    pytest>=8.0

ENV PYTHONPATH=/workspace:/workspace/module

# Default: drop into a shell so the user / Claude Code can drive iteration.
CMD ["/bin/bash"]
