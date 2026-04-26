"""Seed a tiny synthetic (PDF, truth) pair for run-local-fast.

Idempotent — does nothing if the files already exist.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDF_ID = "synthetic_smoke"
PDF_DIR = ROOT / "data/val/pdfs"
TRUTH_DIR = ROOT / "data/val/truth"


def _make_pdf(out: Path) -> None:
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "The Bicycling World, Vol. 5 No. 18, June 3 1892")
    page.insert_text((72, 144), "Editor: L. J. Berger")
    page.insert_text((72, 200), "That's So!")
    page.insert_text((72, 240), "It is not always the man who rides the swiftest...")
    doc.save(out)
    doc.close()


TRUTH = {
    "magazine": {
        "editor": "L. J. Berger",
        "issue": {"date": "1892-06-03", "volume": 5, "number": 18},
        "publisher": {"name": "N. H. Van Sicklen", "address": "57 Plymouth Place, Chicago"},
        "cost": {"issue": None, "annual": "$2.00", "semiannual": "$1.00"},
    },
    "articles": [
        {
            "title": "That's So!",
            "text": ["It is not always the man who rides the swiftest..."],
            "pages": [1],
            "kind": "prose",
        }
    ],
}


def main() -> int:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = PDF_DIR / f"{PDF_ID}.pdf"
    truth_path = TRUTH_DIR / f"{PDF_ID}.json"
    if not pdf_path.is_file():
        _make_pdf(pdf_path)
        print(f"seeded {pdf_path}")
    if not truth_path.is_file():
        truth_path.write_text(json.dumps(TRUTH, indent=2))
        print(f"seeded {truth_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
