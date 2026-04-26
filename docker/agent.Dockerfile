FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Claude Code CLI (developer agent's brain).
# Installed in the snapshot so the container starts cold-fast on the H100.
RUN curl -fsSL https://claude.ai/install.sh | sh || true

WORKDIR /workspace

COPY module/pyproject.toml /tmp/module-pyproject.toml
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
