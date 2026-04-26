"""Print the round-history table from trends/round_history.json."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cfg_path = Path(os.environ.get("DOCSWARM_CONFIG", str(ROOT / "config.yaml")))
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    p = Path(cfg.get("paths", {}).get("trends_file", str(ROOT / "trends/round_history.json")))
    if not p.is_file():
        print("no trends yet")
        return 0
    history = json.loads(p.read_text())

    cols = ["schema_validity", "article_count", "metadata", "titles", "text", "order", "pages"]
    short = {"schema_validity": "schema", "article_count": "count", "metadata": "meta"}
    headers = ["Round", "Agg"] + [short.get(c, c) for c in cols] + ["Δ", "time"]
    print(" | ".join(f"{h:>6}" for h in headers))
    prev = None
    for e in history:
        agg = e.get("aggregate", 0.0)
        delta = "-" if prev is None else f"{agg - prev:+.2f}"
        prev = agg
        comps = e.get("components", {})
        wc = e.get("wall_clock_seconds", 0)
        time_str = f"{int(wc // 60)}m" if wc >= 60 else f"{int(wc)}s"
        row = [
            f"{e.get('round', 0)}",
            f"{agg:.2f}",
            *[f"{comps.get(c, 0.0):.2f}" for c in cols],
            delta,
            time_str,
        ]
        print(" | ".join(f"{v:>6}" for v in row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
