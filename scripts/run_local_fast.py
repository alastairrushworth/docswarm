"""CI-style sanity check: stub out Ollama, run one round on a tiny synthetic PDF.

Completes in under a minute on any machine. Verifies wiring end-to-end:
module → judge → broad-eval → trend file → report.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("local-fast")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "module"))


def _make_synthetic_pdf(out: Path) -> None:
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "The Bicycling World, Vol. 5 No. 18, June 3 1892")
    page.insert_text((72, 144), "Editor: L. J. Berger")
    page.insert_text((72, 200), "Article: That's So!")
    doc.save(out)
    doc.close()


def _stub_pdf_to_json(pdf_path: str) -> dict:
    return {
        "magazine": {
            "editor": "L. J. Berger",
            "issue": {"date": "1892-06-03", "volume": 5, "number": 18},
            "publisher": {"name": "N. H. Van Sicklen", "address": "57 Plymouth Place, Chicago"},
            "cost": {"issue": None, "annual": "$2.00", "semiannual": "$1.00"},
        },
        "articles": [
            {"title": "That's So!", "text": ["stub paragraph"], "pages": [1], "kind": "prose"},
        ],
    }


def _make_truth(out: Path, pdf_id: str) -> None:
    truth = {
        "magazine": {
            "editor": "L. J. Berger",
            "issue": {"date": "1892-06-03", "volume": 5, "number": 18},
            "publisher": {"name": "N. H. Van Sicklen", "address": "57 Plymouth Place, Chicago"},
            "cost": {"issue": None, "annual": "$2.00", "semiannual": "$1.00"},
        },
        "articles": [
            {"title": "That's So!", "text": ["expected paragraph"], "pages": [1], "kind": "prose"},
        ],
    }
    (out / f"{pdf_id}.json").write_text(json.dumps(truth, indent=2))


def main() -> int:
    pdf_dir = ROOT / "data/val/pdfs"
    truth_dir = ROOT / "data/val/truth"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    pdf_id = "synthetic_smoke"
    pdf_path = pdf_dir / f"{pdf_id}.pdf"
    if not pdf_path.is_file():
        _make_synthetic_pdf(pdf_path)
    if not (truth_dir / f"{pdf_id}.json").is_file():
        _make_truth(truth_dir, pdf_id)

    # Start judge in a thread.
    sys.path.insert(0, str(ROOT))
    from judge import judge as judge_mod
    judge_thread = threading.Thread(target=judge_mod.main, daemon=True)
    judge_thread.start()
    time.sleep(0.5)

    # Monkey-patch pdf_to_json to a stub (no Ollama).
    import scripts.run_validation as loop_mod
    loop_mod.pdf_to_json = _stub_pdf_to_json

    cfg_path = Path(os.environ.get("DOCSWARM_CONFIG", str(ROOT / "config.yaml")))
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    entry = loop_mod.run_round(cfg, round_n=1)
    print(json.dumps(entry, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
