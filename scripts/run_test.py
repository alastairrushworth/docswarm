"""User-only: run the frozen module against the test set.

Writes predictions to data/test/predictions/{pdf_id}.json. Does NOT score — final review
is the user's call.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "module"))

from pdf_to_json import pdf_to_json  # noqa: E402


def main() -> int:
    cfg_path = Path(os.environ.get("DOCSWARM_CONFIG", str(ROOT / "config.yaml")))
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    test_dir = Path(cfg.get("paths", {}).get("test_dir", str(ROOT / "data/test")))
    pdf_dir = test_dir / "pdfs"
    out_dir = test_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.error("no test PDFs in %s", pdf_dir)
        return 1

    for p in pdfs:
        logger.info("translating %s", p.name)
        pred = pdf_to_json(str(p))
        out = out_dir / f"{p.stem}.json"
        out.write_text(json.dumps(pred, indent=2))
        logger.info("wrote %s", out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
