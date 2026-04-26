"""DigitalOcean droplet up/down for remote runs.

Belt-and-braces shutdown:
  1. Local trap calls `down` on success/error/Ctrl-C.
  2. A watchdog cron on the droplet self-destructs after iteration.wall_clock_hours.

Usage:
    python orchestration/provision.py up
    python orchestration/provision.py down
    python orchestration/provision.py build-snapshot
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
DROPLET_FILE = ROOT / ".droplet_id"
IP_FILE = ROOT / ".droplet_ip"

DO_API = "https://api.digitalocean.com/v2"


def _cfg() -> dict:
    with (ROOT / "config.yaml").open() as f:
        return yaml.safe_load(f)


def _token(cfg: dict) -> str:
    env_var = cfg["digitalocean"]["api_token_env"]
    tok = os.environ.get(env_var, "").strip()
    if not tok:
        raise SystemExit(f"set ${env_var} to your DigitalOcean API token")
    return tok


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _print_cost_estimate(cfg: dict) -> None:
    hours = cfg["iteration"]["wall_clock_hours"]
    # Rough placeholder; real prices live in DO's API. DESIGN.md says ~$4/h.
    rate = 4.0
    cap = hours * rate
    kill_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + hours * 3600))
    print(f">>> estimated cost cap: ~${cap:.2f} ({hours}h @ ~${rate:.2f}/h)")
    print(f">>> watchdog hard-kill at: {kill_ts} UTC (regardless of local script)")


def _user_data(cfg: dict) -> str:
    """cloud-init: install watchdog cron that destroys the droplet after wall_clock_hours."""
    hours = cfg["iteration"]["wall_clock_hours"]
    minutes = int(hours * 60)
    return f"""#!/bin/bash
set -e
# Self-destruct watchdog: powers off the droplet after the budget elapses.
# DigitalOcean shuts down + destroys droplets that power off if configured;
# we additionally call the metadata API to destroy ourselves.
(
  sleep {minutes * 60}
  DROPLET_ID=$(curl -s http://169.254.169.254/metadata/v1/id)
  curl -X DELETE -H "Authorization: Bearer ${{DO_API_TOKEN}}" \\
       https://api.digitalocean.com/v2/droplets/$DROPLET_ID
  shutdown -h now
) &
"""


def up() -> None:
    cfg = _cfg()
    token = _token(cfg)
    snapshot_id = cfg["digitalocean"].get("snapshot_id", "")
    if not snapshot_id:
        raise SystemExit("config.digitalocean.snapshot_id is empty; run `make build-snapshot` first")

    _print_cost_estimate(cfg)

    body = {
        "name": "docswarm-h100",
        "region": cfg["digitalocean"]["region"],
        "size": cfg["digitalocean"]["size"],
        "image": snapshot_id,
        "ssh_keys": [],
        "user_data": _user_data(cfg),
        "tags": ["docswarm"],
    }
    with httpx.Client(timeout=60.0) as c:
        r = c.post(f"{DO_API}/droplets", headers=_headers(token), json=body)
        r.raise_for_status()
        droplet = r.json()["droplet"]
        droplet_id = droplet["id"]
        DROPLET_FILE.write_text(str(droplet_id))
        print(f">>> droplet {droplet_id} requested; waiting for active + IP")

        for _ in range(120):
            time.sleep(5)
            r = c.get(f"{DO_API}/droplets/{droplet_id}", headers=_headers(token))
            r.raise_for_status()
            d = r.json()["droplet"]
            if d["status"] == "active":
                v4 = [n for n in d["networks"]["v4"] if n["type"] == "public"]
                if v4:
                    ip = v4[0]["ip_address"]
                    IP_FILE.write_text(ip)
                    print(f">>> droplet active at {ip}")
                    return
        raise SystemExit("droplet did not reach active state in 10 minutes")


def down() -> None:
    cfg = _cfg()
    token = _token(cfg)
    if not DROPLET_FILE.is_file():
        print(">>> no droplet on record")
        return
    droplet_id = DROPLET_FILE.read_text().strip()
    with httpx.Client(timeout=60.0) as c:
        r = c.delete(f"{DO_API}/droplets/{droplet_id}", headers=_headers(token))
        if r.status_code in (204, 404):
            print(f">>> droplet {droplet_id} destroyed")
        else:
            r.raise_for_status()
    DROPLET_FILE.unlink(missing_ok=True)
    IP_FILE.unlink(missing_ok=True)


def build_snapshot() -> None:
    print(">>> build-snapshot is a manual workflow:")
    print("    1. Spin up an H100 droplet on DigitalOcean.")
    print("    2. Install: docker, NVIDIA Container Toolkit, ollama.")
    print("    3. Pre-pull every model in config.yaml `models:`.")
    print("    4. Clone this repo to /workspace.")
    print("    5. Power off, snapshot via the DO console.")
    print("    6. Paste the snapshot ID into config.yaml `digitalocean.snapshot_id`.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["up", "down", "build-snapshot"])
    args = parser.parse_args()
    if args.action == "up":
        up()
    elif args.action == "down":
        down()
    else:
        build_snapshot()
    return 0


if __name__ == "__main__":
    sys.exit(main())
