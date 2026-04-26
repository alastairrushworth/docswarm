#!/usr/bin/env bash
# Manual teardown — useful if `run.sh remote` was killed before its trap fired.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python orchestration/provision.py down
rm -f .droplet_ip
