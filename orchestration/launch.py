"""Single-command launcher for a remote H200 iteration run.

Drives `doctl` from your local terminal:
  1. Push the current branch to origin.
  2. `doctl compute droplet create` from the snapshot in config.yaml.
  3. SSH to the droplet, `git pull`, `docker compose up --build`.
  4. Stream stdout/stderr back to the local terminal.
  5. Tear the droplet down on exit (success, error, or Ctrl-C).

Usage:
    python orchestration/launch.py up        # full provision → run → teardown
    python orchestration/launch.py down      # teardown a droplet whose ID we recorded
    python orchestration/launch.py snapshot  # walks you through snapshot creation

Requires: `doctl` authenticated locally (`doctl auth init`), a configured
deploy SSH key (id in `digitalocean.ssh_key_id`), and a snapshot with Docker,
NVIDIA toolkit, Ollama and the model weights pre-pulled.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DROPLET_FILE = ROOT / ".droplet_id"
IP_FILE = ROOT / ".droplet_ip"


def _cfg() -> dict:
    with (ROOT / "config.yaml").open() as f:
        return yaml.safe_load(f)


def _require_doctl() -> None:
    if shutil.which("doctl") is None:
        sys.exit("doctl not found on PATH. Install: https://docs.digitalocean.com/reference/doctl/")


def _doctl(*args: str, capture: bool = False) -> str:
    if capture:
        r = subprocess.run(["doctl", *args], check=True, capture_output=True, text=True)
        return r.stdout.strip()
    subprocess.run(["doctl", *args], check=True)
    return ""


def _git_branch() -> str:
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
                       check=True, capture_output=True, text=True)
    return r.stdout.strip()


def _create_droplet(cfg: dict) -> tuple[str, str]:
    do = cfg["digitalocean"]
    snapshot = do.get("snapshot_id", "")
    if not snapshot:
        sys.exit("config.digitalocean.snapshot_id is empty; run `make build-snapshot` first")
    ssh_key_id = do.get("ssh_key_id", "")
    if not ssh_key_id:
        sys.exit("config.digitalocean.ssh_key_id is empty; `doctl compute ssh-key list`")
    name = do.get("droplet_name", "docswarm-h200")

    print(f">>> creating droplet {name} ({do['size']}, region={do['region']})")
    out = _doctl(
        "compute", "droplet", "create", name,
        "--region", do["region"],
        "--size", do["size"],
        "--image", str(snapshot),
        "--ssh-keys", str(ssh_key_id),
        "--tag-names", "docswarm",
        "--wait",
        "--output", "json",
        capture=True,
    )
    droplets = json.loads(out)
    droplet = droplets[0] if isinstance(droplets, list) else droplets
    droplet_id = str(droplet["id"])
    DROPLET_FILE.write_text(droplet_id)

    ip = ""
    for n in droplet.get("networks", {}).get("v4", []):
        if n.get("type") == "public":
            ip = n.get("ip_address", "")
            break
    if not ip:
        # `--wait` should have populated networks; poll if not.
        ip = _wait_for_ip(droplet_id)
    IP_FILE.write_text(ip)
    print(f">>> droplet {droplet_id} active at {ip}")
    return droplet_id, ip


def _wait_for_ip(droplet_id: str, timeout: float = 300.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        out = _doctl("compute", "droplet", "get", droplet_id,
                     "--output", "json", capture=True)
        d = json.loads(out)
        if isinstance(d, list):
            d = d[0]
        for n in d.get("networks", {}).get("v4", []):
            if n.get("type") == "public":
                return n["ip_address"]
        time.sleep(5)
    sys.exit(f"droplet {droplet_id} never reported a public IP")


def _push_branch(branch: str) -> None:
    print(f">>> pushing {branch} to origin")
    subprocess.run(["git", "push", "origin", branch], cwd=ROOT, check=True)


def _ssh_run(ip: str, cmd: str) -> int:
    full = [
        "ssh", "-t",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ServerAliveInterval=30",
        f"root@{ip}", cmd,
    ]
    return subprocess.run(full, cwd=ROOT).returncode


def _destroy(droplet_id: str | None) -> None:
    if not droplet_id:
        return
    try:
        print(f">>> destroying droplet {droplet_id}")
        subprocess.run(
            ["doctl", "compute", "droplet", "delete", droplet_id, "-f"],
            check=False,
        )
    finally:
        DROPLET_FILE.unlink(missing_ok=True)
        IP_FILE.unlink(missing_ok=True)


def up() -> int:
    _require_doctl()
    cfg = _cfg()
    branch = _git_branch()
    _push_branch(branch)

    droplet_id, ip = _create_droplet(cfg)

    def _on_signal(signum, frame):  # noqa: ARG001
        print(f"\n>>> received signal {signum}; tearing down")
        _destroy(droplet_id)
        sys.exit(130)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        remote_cmd = (
            f"set -e; cd /workspace && "
            f"git fetch origin && git checkout {branch} && git pull --ff-only && "
            f"docker compose up --build --exit-code-from developer-agent"
        )
        rc = _ssh_run(ip, remote_cmd)
        return rc
    finally:
        _destroy(droplet_id)


def down() -> int:
    _require_doctl()
    if not DROPLET_FILE.is_file():
        print(">>> no droplet on record")
        return 0
    droplet_id = DROPLET_FILE.read_text().strip()
    _destroy(droplet_id)
    return 0


def snapshot() -> int:
    print(">>> Snapshot is currently a manual workflow:")
    print("    1. `doctl compute droplet create` an H200 with the base Ubuntu image.")
    print("    2. SSH in. Install: docker, NVIDIA Container Toolkit, ollama.")
    print("    3. Pre-pull every model in config.yaml `models:` (coder, vision, embedding).")
    print("    4. Clone this repo to /workspace.")
    print("    5. `doctl compute droplet-action snapshot --snapshot-name docswarm-base <id>`.")
    print("    6. Paste the snapshot ID into config.yaml `digitalocean.snapshot_id`.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["up", "down", "snapshot"])
    args = p.parse_args()
    if args.action == "up":
        return up()
    if args.action == "down":
        return down()
    return snapshot()


if __name__ == "__main__":
    sys.exit(main())
