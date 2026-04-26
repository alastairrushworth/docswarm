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
        r = subprocess.run(["doctl", *args], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"doctl {' '.join(args)} failed:\n{r.stderr.strip() or r.stdout.strip()}")
        return r.stdout.strip()
    r = subprocess.run(["doctl", *args])
    if r.returncode != 0:
        sys.exit(f"doctl {' '.join(args)} exited {r.returncode}")
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


def _wait_for_ssh(ip: str, timeout: float = 300.0) -> None:
    """Poll until sshd is accepting connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=accept-new",
             "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             f"root@{ip}", "true"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            return
        time.sleep(5)
    sys.exit(f"sshd at {ip} never came up within {timeout:.0f}s")


def _scp(local: Path, ip: str, remote: str) -> None:
    subprocess.run(
        ["scp", "-o", "StrictHostKeyChecking=accept-new",
         str(local), f"root@{ip}:{remote}"],
        check=True,
    )


def _patch_snapshot_id(snapshot_id: str) -> None:
    """In-place edit of config.yaml's digitalocean.snapshot_id, preserving
    formatting and comments."""
    cfg_path = ROOT / "config.yaml"
    text = cfg_path.read_text()
    import re
    new_text, n = re.subn(
        r'(snapshot_id:\s*)"[^"]*"',
        f'\\1"{snapshot_id}"',
        text,
        count=1,
    )
    if n != 1:
        sys.exit("could not locate snapshot_id line in config.yaml")
    cfg_path.write_text(new_text)


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

    print(">>> waiting for sshd")
    _wait_for_ssh(ip)

    try:
        # Standalone compose plugin URL (no apt repo dance, ~10MB binary).
        compose_url = (
            "https://github.com/docker/compose/releases/latest/download/"
            "docker-compose-linux-$(uname -m)"
        )
        remote_cmd = (
            f"set -e; cd /workspace && "
            f"git fetch origin && git checkout {branch} && git pull --ff-only && "
            # Heal snapshots that were built before setup.sh installed the
            # compose plugin. Idempotent (~1s) when already present.
            f"(docker compose version >/dev/null 2>&1 || "
            f"  (install -d /usr/libexec/docker/cli-plugins && "
            f'   curl -fsSL "{compose_url}" -o /usr/libexec/docker/cli-plugins/docker-compose && '
            f"   chmod +x /usr/libexec/docker/cli-plugins/docker-compose)) && "
            # Heal snapshots built before setup.sh staged the deploy key.
            f"(test -f /workspace/secrets/deploy_key || "
            f"  (mkdir -p /workspace/secrets && cp /root/.ssh/id_ed25519 /workspace/secrets/deploy_key && "
            f"   chmod 600 /workspace/secrets/deploy_key)) && "
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
    """Automated one-shot snapshot build:

      1. Spin up a setup droplet (cheap GPU SKU, GPU base image).
      2. SCP deploy key + setup.sh.
      3. Run setup.sh: install Docker / NVIDIA toolkit / Ollama, pull models,
         clone the repo to /workspace.
      4. Shut the droplet down (cleaner snapshots).
      5. doctl droplet-action snapshot --wait.
      6. Patch config.yaml with the new snapshot ID.
      7. Destroy the setup droplet.
    """
    _require_doctl()
    cfg = _cfg()
    do = cfg["digitalocean"]
    repo = cfg["repo"]

    ssh_key_id = do.get("ssh_key_id", "")
    if not ssh_key_id:
        sys.exit("config.digitalocean.ssh_key_id is empty; `doctl compute ssh-key list`")

    deploy_key = ROOT / "secrets/deploy_key"
    if not deploy_key.is_file():
        sys.exit(f"deploy key not found at {deploy_key} — see README")

    setup_script = ROOT / "orchestration/setup.sh"
    if not setup_script.is_file():
        sys.exit(f"setup script missing at {setup_script}")

    size = do.get("snapshot_size") or do["size"]
    image = do.get("snapshot_image", "gpu-h100x1-base")
    name = f"docswarm-snapshot-{int(time.time())}"
    models = [v for v in cfg.get("models", {}).values() if v]

    print(f">>> creating setup droplet {name} ({size}, image={image})")
    out = _doctl(
        "compute", "droplet", "create", name,
        "--region", do["region"],
        "--size", size,
        "--image", image,
        "--ssh-keys", str(ssh_key_id),
        "--tag-names", "docswarm-setup",
        "--wait",
        "--output", "json",
        capture=True,
    )
    droplets = json.loads(out)
    droplet = droplets[0] if isinstance(droplets, list) else droplets
    droplet_id = str(droplet["id"])

    ip = ""
    for n in droplet.get("networks", {}).get("v4", []):
        if n.get("type") == "public":
            ip = n.get("ip_address", "")
            break
    if not ip:
        ip = _wait_for_ip(droplet_id)
    print(f">>> setup droplet {droplet_id} at {ip}; waiting for sshd")
    _wait_for_ssh(ip)

    snapshot_id_out: str | None = None
    try:
        print(">>> copying deploy key + setup.sh")
        _ssh_run(ip, "mkdir -p /root/.ssh && chmod 700 /root/.ssh")
        _scp(deploy_key, ip, "/root/.ssh/id_ed25519")
        _scp(setup_script, ip, "/root/setup.sh")

        env_prefix = (
            f'REPO_URL={subprocess.list2cmdline([repo["url"]])} '
            f'REPO_BRANCH={subprocess.list2cmdline([repo["branch"]])}'
        )
        models_args = " ".join(subprocess.list2cmdline([m]) for m in models)
        cmd = f"chmod +x /root/setup.sh && {env_prefix} /root/setup.sh {models_args}"
        print(">>> running setup.sh on droplet (this takes 30–60 min for first model pulls)")
        rc = _ssh_run(ip, cmd)
        if rc != 0:
            sys.exit(f"setup script failed (rc={rc}); destroying droplet")

        print(">>> shutting down droplet for clean snapshot")
        subprocess.run(
            ["doctl", "compute", "droplet-action", "shutdown", droplet_id, "--wait"],
            check=False,
        )

        snap_name = f"docswarm-base-{int(time.time())}"
        print(f">>> creating snapshot {snap_name} (5–15 min)")
        _doctl("compute", "droplet-action", "snapshot", droplet_id,
               "--snapshot-name", snap_name, "--wait")

        out = _doctl("compute", "snapshot", "list", "--resource", "droplet",
                     "--output", "json", capture=True)
        snapshots = json.loads(out)
        ours = [s for s in snapshots if s.get("name") == snap_name]
        if not ours:
            sys.exit(f"snapshot {snap_name} not found in `doctl compute snapshot list`")
        snapshot_id_out = str(ours[0]["id"])
        _patch_snapshot_id(snapshot_id_out)
        print(f">>> snapshot {snapshot_id_out} written to config.yaml")
    finally:
        print(f">>> destroying setup droplet {droplet_id}")
        subprocess.run(
            ["doctl", "compute", "droplet", "delete", droplet_id, "-f"],
            check=False,
        )

    if snapshot_id_out is None:
        return 1
    print(">>> done. you can now `make run`.")
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
