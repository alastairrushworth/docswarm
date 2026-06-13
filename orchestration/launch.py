"""Single-command launcher for a remote RunPod GPU run.

Drives RunPod's REST API from your local terminal:
  1. Push the current branch to origin.
  2. POST /v1/pods to create a GPU pod from config.yaml.
  3. Poll until the pod has a public IP and SSH is reachable.
  4. SSH to the pod, `git pull`, `docker compose up --build`.
  5. Stream stdout/stderr back to the local terminal.
  6. Tear the pod down on exit (success, error, or Ctrl-C).

Usage:
    python orchestration/launch.py up      # provision → run → teardown
    python orchestration/launch.py down    # teardown a pod whose ID was recorded
    python orchestration/launch.py setup   # one-time: create network volume + install deps

Requires:
    RUNPOD_API_KEY env var  (https://www.runpod.io/console/user/settings)
    secrets/deploy_key      SSH private key (also used as GitHub deploy key)
    secrets/deploy_key.pub  corresponding public key
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import ssl
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
POD_FILE = ROOT / ".pod_id"
BASE_URL = "https://rest.runpod.io/v1"


def _cfg() -> dict:
    with (ROOT / "config.yaml").open() as f:
        return yaml.safe_load(f)


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        env_file = ROOT / ".env"
        if env_file.is_file():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("RUNPOD_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    if not key:
        sys.exit(
            "RUNPOD_API_KEY not found in environment or .env\n"
            "Get your key at https://www.runpod.io/console/user/settings"
        )
    return key


# Datacenters tried in order when datacenter_id is not set or has no capacity.
# List is all datacenters that support network volumes, US-first.
_DC_FALLBACKS = [
    "US-TX-3", "US-KS-2", "US-GA-2", "US-CA-2", "US-NC-1", "US-NC-2",
    "US-IL-1", "US-MD-1", "US-MO-1", "US-MO-2", "US-NE-1", "US-WA-1",
    "EU-RO-1", "EU-NL-1", "EU-SE-1", "EU-CZ-1", "EU-FR-1",
    "EUR-IS-1", "EUR-IS-3", "EUR-IS-4", "EUR-NO-1", "EUR-NO-2",
    "CA-MTL-3", "CA-MTL-4", "US-GA-2", "AP-JP-1",
]

_NO_CAPACITY_PHRASES = ("no instances currently available", "could not find any pods")


def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except ImportError:
        pass
    return ctx


def _api(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {_api_key()}")
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, context=_ssl_ctx()) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")
        sys.exit(f"RunPod API {method} {path} failed ({e.code}): {msg}")


def _ssh_pubkey() -> str:
    pub = ROOT / "secrets/deploy_key.pub"
    if not pub.is_file():
        sys.exit(
            f"SSH public key not found at {pub}\n"
            "Generate with: ssh-keygen -t ed25519 -f secrets/deploy_key -N ''"
        )
    return pub.read_text().strip()


def _create_pod(cfg: dict, volume_id: str | None = None) -> dict:
    rp = cfg["runpod"]
    body: dict = {
        "name": rp.get("pod_name", "docswarm"),
        "imageName": rp["image"],
        "gpuTypeIds": rp["gpu_type_ids"],
        "gpuCount": 1,
        "containerDiskInGb": rp.get("container_disk_gb", 50),
        "cloudType": "SECURE",
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": _ssh_pubkey()},
    }
    vid = volume_id or rp.get("network_volume_id", "")
    if vid:
        body["networkVolumeId"] = vid
    dc_id = rp.get("datacenter_id", "")
    if dc_id:
        body["dataCenterIds"] = [dc_id]
    print(f">>> creating RunPod pod (GPUs: {rp['gpu_type_ids']})")
    pod = _api("POST", "/pods", body)
    print(f">>> pod {pod['id']} created")
    POD_FILE.write_text(pod["id"])
    return pod


def _get_ssh_port(pod: dict) -> str | None:
    mappings = pod.get("portMappings") or {}
    for key in ("22", "22/tcp"):
        if key in mappings:
            return str(mappings[key])
    return None


def _wait_for_pod(pod_id: str, timeout: float = 300.0) -> dict:
    print(f">>> waiting for pod {pod_id} to become ready")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pod = _api("GET", f"/pods/{pod_id}")
        if pod.get("desiredStatus") == "RUNNING" and pod.get("publicIp") and _get_ssh_port(pod):
            return pod
        time.sleep(5)
    sys.exit(f"pod {pod_id} never became ready within {timeout:.0f}s")


def _delete_pod(pod_id: str | None) -> None:
    if not pod_id:
        return
    try:
        print(f">>> destroying pod {pod_id}")
        _api("DELETE", f"/pods/{pod_id}")
    except SystemExit:
        pass
    finally:
        POD_FILE.unlink(missing_ok=True)


def _ssh_flags(ip: str, port: str) -> list[str]:
    deploy_key = ROOT / "secrets/deploy_key"
    return [
        "-i", str(deploy_key),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ServerAliveInterval=30",
        "-p", port,
    ]


def _wait_for_ssh(ip: str, port: str, timeout: float = 600.0) -> None:
    print(f">>> waiting for SSH at {ip}:{port}")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = subprocess.run(
            ["ssh", *_ssh_flags(ip, port),
             "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             f"root@{ip}", "true"],
            capture_output=True,
        )
        if r.returncode == 0:
            return
        time.sleep(5)
    sys.exit(f"SSH at {ip}:{port} never came up within {timeout:.0f}s")


def _ssh_run(ip: str, port: str, cmd: str) -> int:
    return subprocess.run(
        ["ssh", "-t", *_ssh_flags(ip, port), f"root@{ip}", cmd],
        cwd=ROOT,
    ).returncode


def _scp(local: Path, ip: str, port: str, remote: str) -> None:
    deploy_key = ROOT / "secrets/deploy_key"
    subprocess.run(
        ["scp", "-i", str(deploy_key),
         "-o", "StrictHostKeyChecking=accept-new",
         "-o", "BatchMode=yes",
         "-P", port, str(local), f"root@{ip}:{remote}"],
        check=True,
    )


def _sync_data(ip: str, port: str) -> None:
    """Copy local data/{train,val,test} document folders to /workspace/data/ on the pod.

    Only doc subdirectories (not .gitkeep) are transferred. The network volume
    persists this data across all pod runs, so this only needs to run during
    setup (or when local data changes).
    """
    deploy_key = ROOT / "secrets/deploy_key"
    scp_base = [
        "scp", "-i", str(deploy_key),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "BatchMode=yes",
        "-r", "-P", port,
    ]
    for split in ("train", "val", "test"):
        local = ROOT / "data" / split
        if not local.is_dir():
            continue
        doc_dirs = [d for d in sorted(local.iterdir()) if d.is_dir()]
        if not doc_dirs:
            continue
        print(f">>> syncing data/{split} ({len(doc_dirs)} doc(s))")
        _ssh_run(ip, port, f"mkdir -p /workspace/data/{split}")
        for doc_dir in doc_dirs:
            subprocess.run(
                [*scp_base, str(doc_dir), f"root@{ip}:/workspace/data/{split}/"],
                check=True,
            )


def _git_branch() -> str:
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return r.stdout.strip()


def _push_branch(branch: str) -> None:
    print(f">>> pushing {branch} to origin")
    subprocess.run(["git", "push", "origin", branch], cwd=ROOT, check=True)


def _patch_config(volume_id: str, dc_id: str) -> None:
    cfg_path = ROOT / "config.yaml"
    text = cfg_path.read_text()
    text, n1 = re.subn(
        r'(network_volume_id:\s*)"[^"]*"',
        f'\\1"{volume_id}"',
        text, count=1,
    )
    text, n2 = re.subn(
        r'(datacenter_id:\s*)"[^"]*"',
        f'\\1"{dc_id}"',
        text, count=1,
    )
    if n1 != 1 or n2 != 1:
        sys.exit("could not patch network_volume_id / datacenter_id in config.yaml")
    cfg_path.write_text(text)


def up() -> int:
    cfg = _cfg()
    if not cfg["runpod"].get("network_volume_id"):
        sys.exit("config.runpod.network_volume_id is empty — run `make setup` first")

    branch = _git_branch()
    _push_branch(branch)

    pod = _create_pod(cfg)
    pod_id = pod["id"]

    def _on_signal(signum, frame):  # noqa: ARG001
        print(f"\n>>> received signal {signum}; tearing down")
        _delete_pod(pod_id)
        sys.exit(130)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    pod = _wait_for_pod(pod_id)
    ip = pod["publicIp"]
    port = _get_ssh_port(pod)
    print(f">>> pod ready at {ip}:{port}")

    _wait_for_ssh(ip, port)

    try:
        ollama_cfg = cfg.get("ollama", {})
        models_cfg = cfg.get("models", {})

        # 1. Set up SSH key so git and later agent commits work.
        print(">>> setting up SSH keys")
        rc = _ssh_run(ip, port,
            "mkdir -p /root/.ssh && chmod 700 /root/.ssh && "
            "cp /workspace/secrets/deploy_key /root/.ssh/id_ed25519 && "
            "chmod 600 /root/.ssh/id_ed25519 && "
            "ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null; true"
        )
        if rc != 0:
            return rc

        # 2. Install tools + pull latest code (~5 min on cold pod, fast if cached).
        # Tools (ollama, node, claude, python packages) are not on the network volume —
        # they must be installed fresh each time a new pod starts.
        print(">>> bootstrapping tools and pulling code (~5 min on cold pod)")
        bootstrap = (
            "set -e; "
            "command -v ollama >/dev/null 2>&1 || ("
            "  DEBIAN_FRONTEND=noninteractive apt-get update -q && "
            "  DEBIAN_FRONTEND=noninteractive apt-get install -y -q zstd && "
            "  curl -fsSL https://ollama.ai/install.sh | sh); "
            "command -v node >/dev/null 2>&1 || ("
            "  curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && "
            "  DEBIAN_FRONTEND=noninteractive apt-get install -y -q nodejs); "
            "command -v claude >/dev/null 2>&1 || npm install -g @anthropic-ai/claude-code; "
            "pip install --quiet --upgrade "
            "'pydantic>=2.6' 'pyyaml>=6.0' 'pymupdf>=1.24' 'Pillow>=10.0' "
            "'httpx>=0.27' 'numpy>=1.26' 'scipy>=1.11' 'pytest>=8.0'; "
            f"cd /workspace && git fetch origin && "
            f"git checkout {branch} && git pull --ff-only"
        )
        rc = _ssh_run(ip, port, bootstrap)
        if rc != 0:
            return rc

        _sync_data(ip, port)

        coder_model = models_cfg.get("coder", "qwen3.6:35b")
        embed_model = models_cfg.get("embedding", "nomic-embed-text")
        ollama_env = (
            f"OLLAMA_NUM_PARALLEL={ollama_cfg.get('num_parallel', 4)} "
            f"OLLAMA_MAX_LOADED_MODELS={ollama_cfg.get('max_loaded_models', 2)} "
            f"OLLAMA_KEEP_ALIVE={ollama_cfg.get('keep_alive', '24h')} "
            f"OLLAMA_CONTEXT_LENGTH={ollama_cfg.get('context_length', 65536)}"
        )

        # 3. Start Ollama + judge in background, run developer-agent in foreground.
        # Model weights persist on /workspace/ollama-data between runs.
        run_cmd = (
            "set -e; cd /workspace; "
            f"OLLAMA_MODELS=/workspace/ollama-data {ollama_env} "
            "nohup ollama serve >/var/log/ollama.log 2>&1 & "
            "echo '>>> waiting for ollama'; "
            "for i in $(seq 1 120); do "
            "  curl -sf http://localhost:11434/api/version >/dev/null 2>&1 && break; "
            "  sleep 2; "
            "done; "
            "curl -sf http://localhost:11434/api/version >/dev/null 2>&1 || "
            "  { echo 'ERROR: ollama failed to start'; cat /var/log/ollama.log; exit 1; }; "
            f"echo '>>> pulling models (skipped if already cached)'; "
            f"ollama pull {coder_model}; "
            f"ollama pull {embed_model}; "
            "echo '>>> starting judge'; "
            "DOCSWARM_CONFIG=/workspace/config.yaml "
            "OLLAMA_URL=http://localhost:11434 "
            "PYTHONPATH=/workspace "
            "nohup python -m judge.judge >/var/log/judge.log 2>&1 & "
            "echo '>>> starting developer-agent'; "
            "DOCSWARM_CONFIG=/workspace/config.yaml "
            "OLLAMA_URL=http://localhost:11434 "
            "ANTHROPIC_BASE_URL=http://localhost:11434 "
            "ANTHROPIC_AUTH_TOKEN=ollama "
            "ANTHROPIC_API_KEY= "
            "PYTHONPATH=/workspace:/workspace/module "
            "python scripts/run_validation.py"
        )
        rc = _ssh_run(ip, port, run_cmd)
        return rc
    finally:
        _delete_pod(pod_id)


def down() -> int:
    if not POD_FILE.is_file():
        print(">>> no pod on record")
        return 0
    pod_id = POD_FILE.read_text().strip()
    _delete_pod(pod_id)
    return 0


def setup() -> int:
    """One-time setup:
      1. Create a persistent network volume (stores /workspace across all pod runs).
      2. Spin up a GPU pod with the volume attached.
      3. SSH in, run setup.sh: install Docker + NVIDIA toolkit, clone repo, stage deploy key.
      4. Destroy the pod — volume and its contents persist for all future `up` runs.
      5. Write the volume ID back to config.yaml.

    Ollama models are pulled on the first `up` run and cached on the volume.
    """
    cfg = _cfg()
    rp = cfg["runpod"]
    repo = cfg["repo"]

    deploy_key = ROOT / "secrets/deploy_key"
    if not deploy_key.is_file():
        sys.exit(f"deploy key not found at {deploy_key} — see README")
    setup_script = ROOT / "orchestration/setup.sh"
    if not setup_script.is_file():
        sys.exit(f"setup script missing at {setup_script}")

    # Reuse an existing volume if setup was previously interrupted.
    existing_vol = rp.get("network_volume_id", "")
    if existing_vol:
        existing_vols = _api("GET", "/networkvolumes")
        match = next((v for v in existing_vols if v["id"] == existing_vol), None)
        if match:
            print(f">>> reusing existing volume {existing_vol} in {match['dataCenterId']}")
            _patch_config(existing_vol, match["dataCenterId"])
            cfg = _cfg()
            rp = cfg["runpod"]

    dc_id = rp.get("datacenter_id", "")
    candidates = [dc_id, *_DC_FALLBACKS] if dc_id else _DC_FALLBACKS

    volume_size_gb = rp.get("volume_size_gb", 600)
    volume_id = rp.get("network_volume_id", "")
    chosen_dc = dc_id if volume_id else ""
    pod_id = ""

    if volume_id and chosen_dc:
        # Volume already exists — skip straight to pod creation
        print(f">>> creating pod with existing volume {volume_id}")
        cfg["runpod"]["datacenter_id"] = chosen_dc
        try:
            pod = _create_pod(cfg, volume_id=volume_id)
            pod_id = pod["id"]
        except SystemExit as e:
            sys.exit(f"Pod creation failed with existing volume: {e}")
    else:
        volume_id = ""
        chosen_dc = ""

    for dc in dict.fromkeys(candidates):  # deduplicate, preserve order
        if volume_id:
            break  # already handled above
        print(f">>> trying datacenter {dc}")
        try:
            vol = _api("POST", "/networkvolumes", {
                "name": "docswarm-workspace",
                "size": volume_size_gb,
                "dataCenterId": dc,
            })
        except SystemExit as e:
            print(f"    {dc}: volume creation failed — {e}; skipping")
            continue

        vid = vol["id"]
        print(f"    volume {vid} created; attempting pod")
        cfg["runpod"]["datacenter_id"] = dc
        try:
            pod = _create_pod(cfg, volume_id=vid)
            volume_id = vid
            chosen_dc = dc
            pod_id = pod["id"]
            break
        except SystemExit as e:
            err = str(e)
            if any(p in err.lower() for p in _NO_CAPACITY_PHRASES):
                print(f"    {dc}: no GPU capacity; cleaning up volume and trying next")
                try:
                    _api("DELETE", f"/networkvolumes/{vid}")
                except SystemExit:
                    pass
                continue
            # Non-capacity error — don't silently swallow it
            try:
                _api("DELETE", f"/networkvolumes/{vid}")
            except SystemExit:
                pass
            sys.exit(err)

    if not volume_id:
        sys.exit(
            "No GPU capacity found in any datacenter.\n"
            "Check https://www.runpod.io/gpu-instance/pricing for availability."
        )
    print(f">>> network volume {volume_id} / pod {pod_id} ready in {chosen_dc}")

    try:
        pod = _wait_for_pod(pod_id)
        ip = pod["publicIp"]
        port = _get_ssh_port(pod)
        print(f">>> setup pod ready at {ip}:{port}")
        _wait_for_ssh(ip, port)

        print(">>> copying deploy key and setup.sh")
        _ssh_run(ip, port, "mkdir -p /root/.ssh && chmod 700 /root/.ssh")
        _scp(deploy_key, ip, port, "/root/.ssh/id_ed25519")
        _scp(setup_script, ip, port, "/root/setup.sh")

        env_prefix = (
            f"REPO_URL={subprocess.list2cmdline([repo['url']])} "
            f"REPO_BRANCH={subprocess.list2cmdline([repo['branch']])}"
        )
        cmd = f"chmod +x /root/setup.sh && {env_prefix} /root/setup.sh"
        print(">>> running setup.sh (~15 min)")
        rc = _ssh_run(ip, port, cmd)
        if rc != 0:
            sys.exit(f"setup.sh failed (rc={rc}); volume {volume_id} preserved — rerun `make setup` to retry")

        _sync_data(ip, port)
    finally:
        _delete_pod(pod_id)

    _patch_config(volume_id, chosen_dc)
    print(f">>> volume {volume_id} / datacenter {chosen_dc} written to config.yaml")
    print(">>> setup complete — run `make run` to start a GPU session")
    print("    (first run pulls Ollama models ~30 min; cached on the volume thereafter)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["up", "down", "setup"])
    args = p.parse_args()
    if args.action == "up":
        return up()
    if args.action == "down":
        return down()
    return setup()


if __name__ == "__main__":
    # Ensure print() output appears before subprocess output when stdout is piped.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    sys.exit(main())
