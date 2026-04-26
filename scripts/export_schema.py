"""Export module/pdf_to_json/schema.py to schema/schema.json (for docs)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "module"))

from pdf_to_json.schema import Document  # noqa: E402


def main() -> int:
    out = ROOT / "schema" / "schema.json"
    out.write_text(json.dumps(Document.model_json_schema(), indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
