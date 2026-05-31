"""Developer-agent inner loop: run the translator over the TRAIN pairs and
score it with the *same* broad metric the judge uses on val.

Train ground truth is readable by the agent (unlike val/test), so this gives a
fast, honest signal while iterating — no judge round-trip, no val gate. Reuses
`judge.broad.evaluate` and `config.yaml.weights` so the numbers are directly
comparable to the round trend.

Usage:
    python scripts/score_train.py                 # all train docs
    python scripts/score_train.py <doc-id> ...    # only the named doc folders
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("score_train")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "module"))
sys.path.insert(0, str(ROOT))  # so `judge` package imports resolve in the agent container

from pdf_to_json import pdf_to_json  # noqa: E402

try:
    from judge import broad  # noqa: E402
except ImportError as e:  # pragma: no cover - surfaced to the user
    raise SystemExit(
        "could not import judge.broad — the judge package must be importable "
        f"(PYTHONPATH includes repo root). Original error: {e}"
    )

_COMPONENTS = ["schema_validity", "article_count", "metadata", "titles", "text", "order", "pages"]


def _load_cfg() -> dict:
    p = Path(os.environ.get("DOCSWARM_CONFIG", str(ROOT / "config.yaml")))
    with p.open() as f:
        return yaml.safe_load(f)


def _train_docs(cfg: dict, only: list[str]) -> list[tuple[str, Path, Path]]:
    paths = cfg.get("paths", {})
    train_dir = Path(paths.get("train_dir", str(ROOT / "data/train")))
    pdf_name = paths.get("pdf_filename", "original.pdf")
    truth_name = paths.get("truth_filename", "transcribed.json")
    if not train_dir.is_dir():
        return []
    out: list[tuple[str, Path, Path]] = []
    for sub in sorted(train_dir.iterdir()):
        if not sub.is_dir():
            continue
        if only and sub.name not in only:
            continue
        pdf, truth = sub / pdf_name, sub / truth_name
        if pdf.is_file() and truth.is_file():
            out.append((sub.name, pdf, truth))
    return out


def main() -> int:
    cfg = _load_cfg()
    weights = cfg.get("weights", {})
    jp = broad.params_from_config(cfg)
    only = sys.argv[1:]

    docs = _train_docs(cfg, only)
    if not docs:
        logger.error("no train docs found (looked under paths.train_dir%s)",
                     f" matching {only}" if only else "")
        return 1

    aggs: list[float] = []
    comp_totals: dict[str, float] = {k: 0.0 for k in _COMPONENTS}

    for pdf_id, pdf, truth_path in docs:
        logger.info("translating %s", pdf_id)
        pred = pdf_to_json(str(pdf))
        truth = json.loads(truth_path.read_text())
        result = broad.evaluate(
            pred, truth, weights, allow_structural_hints=True,
            bands=jp["bands"], alignment_floor=jp["alignment_floor"],
        )
        agg = result["aggregate"]
        aggs.append(agg)
        comps = {k: result["components"][k]["score"] for k in _COMPONENTS}
        for k in _COMPONENTS:
            comp_totals[k] += comps[k]
        comp_str = ", ".join(f"{k} {comps[k]:.2f}" for k in _COMPONENTS)
        print(f"{pdf_id:35s} agg={agg:.3f}  [{comp_str}]")
        for hint in result.get("hints", []):
            print(f"    hint: {hint}")

    n = len(aggs)
    mean_agg = sum(aggs) / n
    mean_comps = ", ".join(f"{k} {comp_totals[k] / n:.2f}" for k in _COMPONENTS)
    print("-" * 80)
    print(f"MEAN over {n} train doc(s): agg={mean_agg:.3f}  [{mean_comps}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
