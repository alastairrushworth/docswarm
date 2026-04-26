"""Iteration loop: translate val PDFs, submit broad to judge, commit, push, repeat.

The "developer agent" work — code edits, marking probes — happens *outside* this loop
(via Claude Code in the developer-agent container). This script is the harness.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("loop")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "module"))

from pdf_to_json import pdf_to_json  # noqa: E402


def _load_cfg() -> dict[str, Any]:
    p = Path(os.environ.get("DOCSWARM_CONFIG", str(ROOT / "config.yaml")))
    with p.open() as f:
        return yaml.safe_load(f)


def _val_pdfs(cfg: dict) -> list[Path]:
    d = Path(cfg.get("paths", {}).get("val_pdfs_dir", str(ROOT / "data/val/pdfs")))
    return sorted([p for p in d.glob("*.pdf")])


def _pdf_id(p: Path) -> str:
    return p.stem


def _submit_broad(cfg: dict, pdf_id: str, round_n: int, prediction: dict) -> dict:
    paths = cfg.get("paths", {})
    inbox = Path(paths.get("inbox_dir", str(ROOT / "judge/inbox")))
    feedback = Path(paths.get("feedback_dir", str(ROOT / "judge/feedback")))
    inbox.mkdir(parents=True, exist_ok=True)
    feedback.mkdir(parents=True, exist_ok=True)

    name = f"{pdf_id}__broad__round{round_n}__{uuid.uuid4().hex[:6]}.json"
    req = {
        "mode": "broad",
        "pdf_id": pdf_id,
        "round": round_n,
        "prediction": prediction,
    }
    (inbox / name).write_text(json.dumps(req))

    out = feedback / name
    deadline = time.monotonic() + 600.0
    while time.monotonic() < deadline:
        if out.is_file():
            try:
                return json.loads(out.read_text())
            except json.JSONDecodeError:
                pass
        time.sleep(0.5)
    raise TimeoutError(f"judge did not respond for {name} within 600s")


def _aggregate_per_pdf(feedbacks: list[dict]) -> dict[str, Any]:
    aggs = [f.get("aggregate", 0.0) for f in feedbacks]
    avg = sum(aggs) / len(aggs) if aggs else 0.0
    component_keys = ["schema_validity", "article_count", "metadata", "titles", "text", "order", "pages"]
    components: dict[str, float] = {}
    for k in component_keys:
        vals = [(f.get("components") or {}).get(k, {}).get("score", 0.0) for f in feedbacks]
        components[k] = sum(vals) / len(vals) if vals else 0.0
    return {"aggregate": avg, "components": components}


def _append_trend(cfg: dict, entry: dict) -> Path:
    p = Path(cfg.get("paths", {}).get("trends_file", str(ROOT / "trends/round_history.json")))
    p.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    if p.is_file():
        try:
            history = json.loads(p.read_text())
        except json.JSONDecodeError:
            history = []
    history.append(entry)
    p.write_text(json.dumps(history, indent=2))
    return p


def _git(*args: str, cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True)


def _commit_and_push(cfg: dict, message: str) -> None:
    branch = cfg.get("repo", {}).get("branch", "agent")
    try:
        _git("add", "-A")
        status = _git("status", "--porcelain").stdout.strip()
        if not status:
            logger.info("nothing to commit")
            return
        _git("commit", "-m", message)
        _git("push", "origin", branch, check=False)
    except subprocess.CalledProcessError as e:
        logger.warning("git operation failed: %s\n%s", e, e.stderr)


def _component_summary(components: dict[str, float]) -> str:
    return ", ".join(f"{k} {v:.2f}" for k, v in components.items())


def _model_loadout(cfg: dict) -> str:
    m = cfg.get("models", {})
    parts = []
    for k in ("coder", "translator", "judge", "embedding"):
        if m.get(k):
            parts.append(f"{k}={m[k]}")
    return ", ".join(parts)


def _translate_and_submit(cfg: dict, round_n: int, p: Path) -> dict:
    pid = _pdf_id(p)
    logger.info("round %d: translating %s", round_n, pid)
    prediction = pdf_to_json(str(p))
    logger.info("round %d: submitting broad eval for %s", round_n, pid)
    fb = _submit_broad(cfg, pid, round_n, prediction)
    logger.info("round %d: %s aggregate=%.3f", round_n, pid, fb.get("aggregate", 0.0))
    return fb


def run_round(cfg: dict, round_n: int) -> dict[str, Any]:
    pdfs = _val_pdfs(cfg)
    if not pdfs:
        raise SystemExit(f"no validation PDFs found in {cfg['paths']['val_pdfs_dir']}")

    pdf_concurrency = max(1, int(cfg.get("iteration", {}).get("pdf_concurrency", 1)))

    t0 = time.monotonic()
    feedbacks_by_pdf: dict[str, dict] = {}
    marking_calls = 0

    with ThreadPoolExecutor(max_workers=pdf_concurrency) as ex:
        futs = {ex.submit(_translate_and_submit, cfg, round_n, p): p for p in pdfs}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                feedbacks_by_pdf[_pdf_id(p)] = fut.result()
            except Exception as e:
                logger.warning("round %d: %s failed: %s", round_n, _pdf_id(p), e)
                feedbacks_by_pdf[_pdf_id(p)] = {"aggregate": 0.0, "components": {}}

    feedbacks = [feedbacks_by_pdf[_pdf_id(p)] for p in pdfs]
    agg = _aggregate_per_pdf(feedbacks)
    elapsed = time.monotonic() - t0

    entry = {
        "round": round_n,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "wall_clock_seconds": round(elapsed, 1),
        "aggregate": round(agg["aggregate"], 4),
        "components": {k: round(v, 4) for k, v in agg["components"].items()},
        "marking_calls_in_round": marking_calls,
        "model_loadout": _model_loadout(cfg),
        "per_pdf": [
            {"pdf_id": _pdf_id(p), "aggregate": fb.get("aggregate", 0.0)}
            for p, fb in zip(pdfs, feedbacks)
        ],
    }
    _append_trend(cfg, entry)
    return entry


def _component_short(entry: dict) -> str:
    c = entry["components"]
    return (
        f"titles {c['titles']:.2f}, text {c['text']:.2f}, order {c['order']:.2f}, "
        f"meta {c['metadata']:.2f}, count {c['article_count']:.2f}, "
        f"schema {c['schema_validity']:.2f}, pages {c['pages']:.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-git", action="store_true", help="skip git commit/push (local mode)")
    parser.add_argument("--single-round", action="store_true", help="run one round and exit")
    args = parser.parse_args()

    cfg = _load_cfg()
    iter_cfg = cfg.get("iteration", {})
    wall_clock_s = float(iter_cfg.get("wall_clock_hours", 12)) * 3600
    epsilon = float(iter_cfg.get("epsilon", 0.005))
    min_rounds = int(iter_cfg.get("min_rounds_before_plateau", 3))
    plateau_floor = float(iter_cfg.get("plateau_aggregate_floor", 0.30))
    plateau_window = int(iter_cfg.get("plateau_window", 3))

    deadline = time.monotonic() + wall_clock_s
    best = -math.inf
    rounds_since_best = 0
    round_n = 0
    prev_aggregate = None

    while time.monotonic() < deadline:
        round_n += 1
        entry = run_round(cfg, round_n)
        agg = entry["aggregate"]
        delta = (agg - prev_aggregate) if prev_aggregate is not None else 0.0
        prev_aggregate = agg

        message = (
            f"round {round_n}: aggregate {agg:.3f} (Δ{delta:+.3f}); "
            f"{_component_short(entry)}"
        )
        logger.info(message)
        if not args.no_git:
            _commit_and_push(cfg, message)

        if agg > best + epsilon:
            best = agg
            rounds_since_best = 0
        else:
            rounds_since_best += 1

        plateau_active = round_n >= min_rounds and best >= plateau_floor
        if plateau_active and rounds_since_best >= plateau_window:
            logger.info("plateau: stopping after round %d", round_n)
            break

        if args.single_round:
            break

    logger.info("loop finished after round %d (best aggregate=%.3f)", round_n, best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
