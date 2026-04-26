"""Resolve a marking-mode `focus.path` into the corresponding truth slice.

Predicted-side article indices are mapped to truth indices via Hungarian alignment.
"""
from __future__ import annotations

import re
from typing import Any

from .alignment import align_articles

_ARTICLE_RE = re.compile(r"^articles\[(\d+)\](?:\.(.+))?$")
_META_FIELDS = {
    "magazine.editor", "magazine.publisher.name", "magazine.publisher.address",
    "magazine.issue.date", "magazine.issue.volume", "magazine.issue.number",
    "magazine.cost", "magazine.cost.issue", "magazine.cost.annual", "magazine.cost.semiannual",
}


def _get_path(d: Any, dotted: str) -> Any:
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def resolve(
    path: str,
    prediction: dict[str, Any],
    truth: dict[str, Any],
) -> dict[str, Any]:
    """Return {pred_value, truth_value, kind, notes}.

    kind ∈ {"article", "metadata", "unknown"}; notes describes any ambiguity
    (e.g. predicted article index has no truth match).
    """
    pred_articles = prediction.get("articles") or []
    truth_articles = truth.get("articles") or []

    m = _ARTICLE_RE.match(path)
    if m:
        idx = int(m.group(1))
        sub = m.group(2)
        if idx >= len(pred_articles):
            return {
                "pred_value": None,
                "truth_value": None,
                "kind": "article",
                "notes": f"no predicted article at index {idx}",
            }
        matches = align_articles(pred_articles, truth_articles)
        match = next(((i, j, s) for i, j, s in matches if i == idx), None)
        if match is None:
            return {
                "pred_value": pred_articles[idx] if not sub else _get_path(pred_articles[idx], sub),
                "truth_value": None,
                "kind": "article",
                "notes": f"predicted article #{idx} has no truth alignment (likely extra)",
            }
        _, j, sim = match
        pred = pred_articles[idx] if not sub else _get_path(pred_articles[idx], sub)
        true = truth_articles[j] if not sub else _get_path(truth_articles[j], sub)
        return {
            "pred_value": pred,
            "truth_value": true,
            "kind": "article",
            "notes": f"aligned to truth article #{j} (sim={sim:.2f})",
        }

    if path in _META_FIELDS or path.startswith("magazine."):
        return {
            "pred_value": _get_path(prediction, path),
            "truth_value": _get_path(truth, path),
            "kind": "metadata",
            "notes": "",
        }

    return {
        "pred_value": None,
        "truth_value": None,
        "kind": "unknown",
        "notes": f"unrecognized path: {path}",
    }
