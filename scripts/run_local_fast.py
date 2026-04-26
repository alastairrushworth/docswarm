"""Bring up the docker stack with stub Ollama, run one round, tear down.

Verifies wiring end-to-end: container networking, judge daemon, inbox/feedback
file protocol, trend file, report. No real models pulled. Target: <60s.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = [
    "docker", "compose",
    "-f", "docker-compose.yml",
    "-f", "docker-compose.local.yml",
    "-f", "docker-compose.fast.yml",
]


def run(*cmd: str, check: bool = True) -> int:
    print("$", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT, check=check).returncode


def main() -> int:
    # Seed synthetic PDF + truth (host-side; mounted into containers).
    run(sys.executable, str(ROOT / "scripts/seed_synthetic.py"))

    try:
        run(*COMPOSE, "up", "--build", "-d", "ollama-main", "judge")
        run(
            *COMPOSE, "run", "--rm",
            "developer-agent",
            "python", "scripts/run_validation.py", "--no-git", "--single-round",
        )
        run(sys.executable, str(ROOT / "scripts/report.py"))
    finally:
        run(*COMPOSE, "down", check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
