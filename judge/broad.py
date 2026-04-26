"""Broad-mode scoring: weighted continuous components, Hungarian-aligned articles."""
from __future__ import annotations

from typing import Any

from .alignment import align_articles
from .similarity import jaccard, text_similarity, title_similarity


def _schema_validity(pred: dict[str, Any]) -> float:
    """Partial credit per top-level / metadata field that parses sensibly."""
    fields = []
    mag = pred.get("magazine") or {}
    fields.append(bool(mag.get("editor")))
    issue = mag.get("issue") or {}
    fields.append(bool(issue.get("date")))
    fields.append(isinstance(issue.get("volume"), int))
    fields.append(isinstance(issue.get("number"), int))
    pub = mag.get("publisher") or {}
    fields.append(bool(pub.get("name")))
    fields.append(bool(pub.get("address")))
    fields.append(isinstance(pred.get("articles"), list))
    if not fields:
        return 0.0
    return sum(1 for f in fields if f) / len(fields)


def _article_count(pred: dict[str, Any], truth: dict[str, Any]) -> tuple[float, int]:
    n_p = len(pred.get("articles") or [])
    n_t = len(truth.get("articles") or [])
    if n_p == 0 and n_t == 0:
        return 1.0, 0
    denom = max(n_p, n_t)
    delta = n_p - n_t
    return 1.0 - abs(delta) / denom, delta


def _metadata(pred: dict[str, Any], truth: dict[str, Any]) -> tuple[float, int, int]:
    p = pred.get("magazine") or {}
    t = truth.get("magazine") or {}
    fields: list[tuple[Any, Any]] = [
        (p.get("editor"), t.get("editor")),
        ((p.get("issue") or {}).get("date"), (t.get("issue") or {}).get("date")),
        ((p.get("issue") or {}).get("volume"), (t.get("issue") or {}).get("volume")),
        ((p.get("issue") or {}).get("number"), (t.get("issue") or {}).get("number")),
        ((p.get("publisher") or {}).get("name"), (t.get("publisher") or {}).get("name")),
        ((p.get("publisher") or {}).get("address"), (t.get("publisher") or {}).get("address")),
    ]
    matched = sum(1 for a, b in fields if a is not None and a == b)
    total = len(fields)
    return (matched / total if total else 0.0), matched, total


def _titles_score(matches, pred_articles, truth_articles) -> float:
    if not matches:
        return 0.0
    return sum(
        title_similarity(pred_articles[i].get("title", ""), truth_articles[j].get("title", ""))
        for i, j, _ in matches
    ) / len(matches)


def _text_score(matches, pred_articles, truth_articles) -> float:
    if not matches:
        return 0.0
    return sum(
        text_similarity(
            " ".join(pred_articles[i].get("text") or []),
            " ".join(truth_articles[j].get("text") or []),
        )
        for i, j, _ in matches
    ) / len(matches)


def _order_score(matches, n_articles: int) -> float:
    if not matches or n_articles == 0:
        return 0.0
    displacements = [abs(i - j) for i, j, _ in matches]
    return max(0.0, 1.0 - sum(displacements) / (len(displacements) * n_articles))


def _pages_score(matches, pred_articles, truth_articles) -> float:
    if not matches:
        return 0.0
    total = 0.0
    for i, j, _ in matches:
        sp = set(pred_articles[i].get("pages") or [])
        st = set(truth_articles[j].get("pages") or [])
        if not sp and not st:
            total += 1.0
        elif not sp or not st:
            total += 0.0
        else:
            total += len(sp & st) / len(sp | st)
    return total / len(matches)


def _categorical_errors(
    pred: dict[str, Any],
    truth: dict[str, Any],
    matches: list[tuple[int, int, float]],
) -> list[dict[str, Any]]:
    pred_articles = pred.get("articles") or []
    truth_articles = truth.get("articles") or []

    matched_pred = {i for i, _, _ in matches}
    matched_truth = {j for _, j, _ in matches}

    errors: list[dict[str, Any]] = []

    missing = len(truth_articles) - len(matched_truth)
    if missing > 0:
        errors.append({"category": "missing_article", "count": missing})
    extra = len(pred_articles) - len(matched_pred)
    if extra > 0:
        errors.append({"category": "extra_article", "count": extra})

    for i, j, _ in matches:
        p = pred_articles[i]
        t = truth_articles[j]
        if p.get("kind") != t.get("kind"):
            cat = (
                "verse_misformatted_as_prose"
                if t.get("kind") == "verse" and p.get("kind") == "prose"
                else "wrong_kind"
            )
            errors.append({"category": cat, "predicted_index": i})
        sp = set(p.get("pages") or [])
        st = set(t.get("pages") or [])
        if sp and st and sp != st:
            errors.append({"category": "page_number_wrong", "predicted_index": i})

    return errors


def _structural_hints(
    errors: list[dict[str, Any]],
    truth: dict[str, Any],
    allow_structural: bool,
) -> list[str]:
    hints: list[str] = []
    counts = {e.get("category"): e.get("count") for e in errors if "count" in e}

    if counts.get("missing_article"):
        if allow_structural:
            truth_articles = truth.get("articles") or []
            pages = sorted({p for a in truth_articles for p in (a.get("pages") or [])})
            page_str = ", ".join(str(p) for p in pages) if pages else "unknown"
            hints.append(
                f"{counts['missing_article']} expected article(s) missing; "
                f"truth articles span pages: {page_str}."
            )
        else:
            hints.append(f"{counts['missing_article']} expected article(s) missing.")
    if counts.get("extra_article"):
        hints.append(f"{counts['extra_article']} predicted article(s) have no truth match.")
    for e in errors:
        if e.get("category") == "verse_misformatted_as_prose":
            hints.append(
                f"Predicted article #{e['predicted_index']} appears to be verse "
                "but was emitted as prose."
            )
        elif e.get("category") == "page_number_wrong":
            hints.append(
                f"Predicted article #{e['predicted_index']} has a wrong page number."
            )
    return hints


def evaluate(
    prediction: dict[str, Any],
    truth: dict[str, Any],
    weights: dict[str, float],
    allow_structural_hints: bool,
) -> dict[str, Any]:
    pred_articles = prediction.get("articles") or []
    truth_articles = truth.get("articles") or []
    matches = align_articles(pred_articles, truth_articles)

    schema = _schema_validity(prediction)
    count_score, delta = _article_count(prediction, truth)
    meta_score, matched, total = _metadata(prediction, truth)
    titles = _titles_score(matches, pred_articles, truth_articles)
    text = _text_score(matches, pred_articles, truth_articles)
    n_articles = max(len(pred_articles), len(truth_articles), 1)
    order = _order_score(matches, n_articles)
    pages = _pages_score(matches, pred_articles, truth_articles)

    components = {
        "schema_validity": {"score": schema},
        "article_count":   {"score": count_score, "delta": delta},
        "metadata":        {"score": meta_score, "matched": matched, "total": total},
        "titles":          {"score": titles},
        "text":            {"score": text},
        "order":           {"score": order},
        "pages":           {"score": pages},
    }

    aggregate = sum(
        weights.get(k, 0.0) * components[k]["score"] for k in components
    )

    errors = _categorical_errors(prediction, truth, matches)
    hints = _structural_hints(errors, truth, allow_structural_hints)

    return {
        "aggregate": round(aggregate, 4),
        "components": components,
        "categorical_errors": errors,
        "hints": hints,
    }
