"""Judge daemon: watches inbox/, writes feedback/.

Request files are JSON with at minimum:
  {"mode": "broad"|"marking", "pdf_id": str, "prediction": {...}, ...}

Truth lookup: data/val/truth/{pdf_id}.json (judge-only mount).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from . import broad, marking
from .leakage_filter import collect_truth_strings, filter_hints, overlap_ratio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("judge")


def _load_config() -> dict[str, Any]:
    cfg_path = Path(os.environ.get("DOCSWARM_CONFIG", "/workspace/config.yaml"))
    with cfg_path.open() as f:
        return yaml.safe_load(f)


def _truth_path(cfg: dict, pdf_id: str) -> Path:
    base = Path(cfg.get("paths", {}).get("val_truth_dir", "/workspace/data/val/truth"))
    return base / f"{pdf_id}.json"


def _load_truth(cfg: dict, pdf_id: str) -> dict | None:
    p = _truth_path(cfg, pdf_id)
    if not p.is_file():
        logger.warning("truth not found for pdf_id=%s at %s", pdf_id, p)
        return None
    with p.open() as f:
        return json.load(f)


def handle(req: dict[str, Any], cfg: dict) -> dict[str, Any]:
    mode = req.get("mode")
    pdf_id = req.get("pdf_id", "")
    prediction = req.get("prediction") or {}
    truth = _load_truth(cfg, pdf_id)
    if truth is None:
        return {
            "mode": mode,
            "pdf_id": pdf_id,
            "error": f"truth not found for pdf_id={pdf_id}",
        }

    weights = cfg.get("weights", {})
    leakage = cfg.get("leakage", {})
    threshold = float(leakage.get("hint_overlap_filter_threshold", 0.30))
    allow_structural = bool(leakage.get("allow_structural_hints", True))
    truth_strings = collect_truth_strings(truth)

    if mode == "broad":
        result = broad.evaluate(prediction, truth, weights, allow_structural)
        result["hints"] = filter_hints(result.get("hints", []), truth_strings, threshold)
        return {
            "mode": "broad",
            "pdf_id": pdf_id,
            "round": req.get("round"),
            **result,
        }

    if mode == "marking":
        result = marking.evaluate(prediction, truth, req)
        feedback = result.get("feedback", "")
        if feedback and overlap_ratio(feedback, truth_strings) >= threshold:
            logger.warning(
                "marking feedback redacted for high overlap: %.200s", feedback
            )
            result["feedback"] = "[redacted: overlapped with ground truth]"
            result["verdict"] = "unverifiable"
        return {
            "mode": "marking",
            "pdf_id": pdf_id,
            "question": req.get("question", ""),
            "focus_path": (req.get("focus") or {}).get("path", ""),
            **result,
        }

    return {"error": f"unknown mode: {mode}"}


def _process_one(req_path: Path, cfg: dict, feedback_dir: Path) -> None:
    try:
        with req_path.open() as f:
            req = json.load(f)
    except Exception as e:
        logger.error("malformed request %s: %s", req_path.name, e)
        return

    response = handle(req, cfg)
    out = feedback_dir / req_path.name
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(response, f, indent=2)
    tmp.replace(out)
    logger.info("processed %s -> %s", req_path.name, out.name)


def main() -> int:
    cfg = _load_config()
    paths = cfg.get("paths", {})
    inbox = Path(paths.get("inbox_dir", "/workspace/judge/inbox"))
    feedback = Path(paths.get("feedback_dir", "/workspace/judge/feedback"))
    inbox.mkdir(parents=True, exist_ok=True)
    feedback.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    logger.info("judge watching %s", inbox)
    while True:
        for p in sorted(inbox.glob("*.json")):
            if p.name in seen:
                continue
            if (feedback / p.name).is_file():
                seen.add(p.name)
                continue
            _process_one(p, cfg, feedback)
            seen.add(p.name)
        time.sleep(0.5)


if __name__ == "__main__":
    sys.exit(main() or 0)
