"""Marking mode: targeted scoring of a partial prediction."""
from __future__ import annotations

from typing import Any

from .similarity import jaccard, text_similarity, title_similarity


def _get_meta_field(d: dict[str, Any], dotted: str) -> Any:
    cur: Any = d.get("magazine") or {}
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def evaluate(
    prediction: dict[str, Any],
    truth: dict[str, Any],
    scope: dict[str, Any],
) -> dict[str, Any]:
    """Return scores only for the requested scope plus a categorical hint if low."""
    scores: dict[str, float] = {}
    errors: list[dict[str, Any]] = []
    hints: list[str] = []

    if "metadata_field" in scope:
        field = scope["metadata_field"]
        p = _get_meta_field(prediction, field)
        t = _get_meta_field(truth, field)
        match = 1.0 if (p is not None and p == t) else 0.0
        scores[field] = match
        if match < 1.0:
            errors.append({"category": "wrong_metadata_field", "field": field})
            hints.append(f"Metadata field '{field}' does not match.")
        return {"scores": scores, "categorical_errors": errors, "hints": hints}

    if scope.get("section") == "metadata":
        from .broad import _metadata
        score, matched, total = _metadata(prediction, truth)
        scores["metadata"] = score
        if score < 1.0:
            hints.append(f"Metadata: {matched}/{total} fields match.")
        return {"scores": scores, "categorical_errors": errors, "hints": hints}

    if "article_index" in scope:
        idx = int(scope["article_index"])
        pred_articles = prediction.get("articles") or []
        truth_articles = truth.get("articles") or []
        if idx >= len(pred_articles):
            return {
                "scores": {},
                "categorical_errors": [{"category": "missing_article"}],
                "hints": [f"No predicted article at index {idx}."],
            }
        # Match against the closest-aligned truth article (best title+text).
        p = pred_articles[idx]
        if not truth_articles:
            return {
                "scores": {},
                "categorical_errors": [{"category": "extra_article"}],
                "hints": ["No truth articles available for comparison."],
            }
        best_j = max(
            range(len(truth_articles)),
            key=lambda j: 0.5 * title_similarity(p.get("title", ""), truth_articles[j].get("title", ""))
            + 0.5 * text_similarity(
                " ".join(p.get("text") or []),
                " ".join(truth_articles[j].get("text") or []),
            ),
        )
        t = truth_articles[best_j]

        fields = scope.get("fields", "all")
        if fields == "all":
            fields = ["title", "text", "kind", "pages"]

        for f in fields:
            if f == "title":
                scores["title"] = title_similarity(p.get("title", ""), t.get("title", ""))
            elif f == "text":
                scores["text"] = text_similarity(
                    " ".join(p.get("text") or []),
                    " ".join(t.get("text") or []),
                )
            elif f == "kind":
                scores["kind"] = 1.0 if p.get("kind") == t.get("kind") else 0.0
                if scores["kind"] < 1.0 and t.get("kind") == "verse":
                    errors.append({"category": "verse_misformatted_as_prose"})
                    hints.append(
                        "This article appears to be verse; predicted text uses prose paragraph splits."
                    )
            elif f == "pages":
                sp = set(p.get("pages") or [])
                st = set(t.get("pages") or [])
                if not sp and not st:
                    scores["pages"] = 1.0
                elif not sp or not st:
                    scores["pages"] = 0.0
                else:
                    scores["pages"] = len(sp & st) / len(sp | st)

        return {"scores": scores, "categorical_errors": errors, "hints": hints}

    return {
        "scores": {},
        "categorical_errors": [],
        "hints": ["Unrecognized scope; supported: article_index, metadata_field, section=metadata."],
    }
