"""Broad-mode scoring: weighted continuous components, Hungarian-aligned articles.

Scoring grades *information*, not transcription form. Metadata and titles use
form-insensitive similarity with tolerance bands; article alignment uses a
similarity floor so a missing/extra article is not force-matched to an unrelated
one. Every imperfect component emits a non-prescriptive hint that points the
developer agent at *where/what kind* of problem to investigate — never the truth
content nor the exact fix.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any

from .alignment import align_articles
from .normalize import field_similarity
from .similarity import text_similarity, title_similarity

_DEFAULT_BANDS = {"metadata": 0.90, "title": 0.85}


def _as_dict(x: Any) -> dict[str, Any]:
    """Coerce a possibly-malformed nested field to a dict. Ground truth or model
    output can carry a string/None where the schema expects an object (e.g.
    `magazine.publisher` as a bare string); treat those as empty rather than
    crashing the whole eval. `x or {}` is not enough — a truthy non-dict (a
    non-empty string) slips through and then `.get` raises."""
    return x if isinstance(x, dict) else {}


def params_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Pull the essence-over-form knobs from config so the judge and the agent's
    train self-scoring use identical settings."""
    j = cfg.get("judge", {}) or {}
    bands = {**_DEFAULT_BANDS, **(j.get("bands") or {})}
    return {"bands": bands, "alignment_floor": float(j.get("alignment_floor", 0.0))}


# --------------------------------------------------------------------------- #
# Component scores
# --------------------------------------------------------------------------- #

def _schema_validity(pred: dict[str, Any]) -> float:
    """Partial credit per top-level / metadata field that parses sensibly."""
    checks = _schema_checks(pred)
    return sum(1 for ok in checks.values() if ok) / max(len(checks), 1)


def _schema_checks(pred: dict[str, Any]) -> dict[str, bool]:
    mag = _as_dict(pred.get("magazine"))
    issue = _as_dict(mag.get("issue"))
    pub = _as_dict(mag.get("publisher"))
    return {
        "magazine.editor": bool(mag.get("editor")),
        "magazine.issue.date": bool(issue.get("date")),
        "magazine.issue.volume (int)": isinstance(issue.get("volume"), int)
        and not isinstance(issue.get("volume"), bool),
        "magazine.issue.number (int)": isinstance(issue.get("number"), int)
        and not isinstance(issue.get("number"), bool),
        "magazine.publisher.name": bool(pub.get("name")),
        "magazine.publisher.address": bool(pub.get("address")),
        "articles (list)": isinstance(pred.get("articles"), list),
    }


def _article_count(pred: dict[str, Any], truth: dict[str, Any]) -> tuple[float, int]:
    n_p = len(pred.get("articles") or [])
    n_t = len(truth.get("articles") or [])
    if n_p == 0 and n_t == 0:
        return 1.0, 0
    denom = max(n_p, n_t)
    delta = n_p - n_t
    return 1.0 - abs(delta) / denom, delta


def _precision_score(pred_articles: list[Any], matches: list[tuple[int, int, float]]) -> float:
    """Fraction of *predicted* articles that aligned to a real one.

    Low precision means over-segmentation — ads, mastheads, standing departments
    or sub-headed fragments emitted as separate articles. This is the component
    the metric was previously blind to: titles/text/order/pages score the matched
    pairs only, so 49 phantom articles cost nothing there, and article_count
    (a symmetric count-delta) under-weights it. Precision makes those false
    positives cost aggregate score directly, creating the gradient to stop
    over-segmenting."""
    n_pred = len(pred_articles)
    if n_pred == 0:
        return 1.0  # no predictions → no false positives (recall is article_count's job)
    return len({i for i, _, _ in matches}) / n_pred


def _to_int(x: Any) -> int | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        m = re.search(r"-?\d+", x)
        return int(m.group()) if m else None
    return None


def _parse_date(x: Any) -> date | None:
    if isinstance(x, str):
        try:
            return date.fromisoformat(x[:10])
        except ValueError:
            return None
    return None


def _date_score(a: Any, b: Any) -> float:
    da, db = _parse_date(a), _parse_date(b)
    if da and db:
        if da == db:
            return 1.0
        if (da.year, da.month) == (db.year, db.month):
            return 0.7
        if da.year == db.year:
            return 0.5
        return 0.0
    return field_similarity(a, b)  # unparseable on either side → fuzzy string


def _int_score(a: Any, b: Any) -> float:
    ia, ib = _to_int(a), _to_int(b)
    if ia is None or ib is None:
        return 0.0
    return 1.0 if ia == ib else 0.0


def _metadata(
    pred: dict[str, Any], truth: dict[str, Any], band: float
) -> tuple[float, int, int, list[dict[str, Any]]]:
    """Graded per-field metadata. Form-insensitive for text fields; date and
    integer fields parsed before comparison. Returns (score, matched, total,
    per_field) where per_field carries each field's banded score, raw
    similarity, and whether the prediction populated it (for hints)."""
    p = _as_dict(pred.get("magazine"))
    t = _as_dict(truth.get("magazine"))
    pi, ti = _as_dict(p.get("issue")), _as_dict(t.get("issue"))
    pp, tp = _as_dict(p.get("publisher")), _as_dict(t.get("publisher"))

    per_field: list[dict[str, Any]] = []

    def record(name: str, raw: float, present: bool, banded: bool = True) -> float:
        snapped = 1.0 if (banded and raw >= band) else raw
        per_field.append({"name": name, "score": snapped, "raw": raw, "present": present})
        return snapped

    scores = [
        record("editor", field_similarity(p.get("editor"), t.get("editor")),
               bool(p.get("editor"))),
        record("issue.date", _date_score(pi.get("date"), ti.get("date")),
               pi.get("date") not in (None, ""), banded=False),
        record("issue.volume", _int_score(pi.get("volume"), ti.get("volume")),
               pi.get("volume") is not None, banded=False),
        record("issue.number", _int_score(pi.get("number"), ti.get("number")),
               pi.get("number") is not None, banded=False),
        record("publisher.name", field_similarity(pp.get("name"), tp.get("name")),
               bool(pp.get("name"))),
        record("publisher.address", field_similarity(pp.get("address"), tp.get("address")),
               bool(pp.get("address"))),
    ]
    total = len(scores)
    score = sum(scores) / total if total else 0.0
    matched = sum(1 for s in scores if s >= 0.999)
    return score, matched, total, per_field


def _titles_score(matches, pred_articles, truth_articles, band: float) -> float:
    if not matches:
        return 0.0
    total = 0.0
    for i, j, _ in matches:
        s = title_similarity(pred_articles[i].get("title", ""), truth_articles[j].get("title", ""))
        total += 1.0 if s >= band else s
    return total / len(matches)


def _text_pair_scores(matches, pred_articles, truth_articles) -> list[tuple[int, float]]:
    """(predicted_index, text_similarity) per matched pair."""
    out = []
    for i, j, _ in matches:
        s = text_similarity(
            " ".join(pred_articles[i].get("text") or []),
            " ".join(truth_articles[j].get("text") or []),
        )
        out.append((i, s))
    return out


def _text_score(pair_scores: list[tuple[int, float]]) -> float:
    if not pair_scores:
        return 0.0
    return sum(s for _, s in pair_scores) / len(pair_scores)


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


# --------------------------------------------------------------------------- #
# Hints — one per imperfect component. Non-prescriptive, non-leaking:
# reference predicted indices / schema field names / numeric scores only;
# never truth content, never the exact correction.
# --------------------------------------------------------------------------- #

def _page_offset(matches, pred_articles, truth_articles) -> int | None:
    """If every single-page matched pair is shifted by the same nonzero
    constant, return it (a likely page-numbering-convention mismatch)."""
    diffs = set()
    pairs = 0
    for i, j, _ in matches:
        sp = sorted(pred_articles[i].get("pages") or [])
        st = sorted(truth_articles[j].get("pages") or [])
        if len(sp) == 1 and len(st) == 1:
            diffs.add(sp[0] - st[0])
            pairs += 1
    if pairs >= 2 and len(diffs) == 1:
        d = next(iter(diffs))
        return d if d != 0 else None
    return None


def _component_hints(
    pred: dict[str, Any],
    truth: dict[str, Any],
    matches: list[tuple[int, int, float]],
    components: dict[str, dict[str, Any]],
    per_field: list[dict[str, Any]],
    text_pairs: list[tuple[int, float]],
    band: dict[str, float],
    allow_structural: bool,
) -> list[str]:
    pred_articles = pred.get("articles") or []
    truth_articles = truth.get("articles") or []
    hints: list[str] = []

    # schema
    gaps = [name for name, ok in _schema_checks(pred).items() if not ok]
    if gaps:
        hints.append(
            "Schema gaps in: " + ", ".join(gaps)
            + ". Ensure each is present and of the correct type per schema.py."
        )

    # article count
    delta = components["article_count"].get("delta", 0)
    if delta > 0:
        hints.append(
            f"You produced {delta} more article(s) than expected. Investigate "
            "over-segmentation: ads, mastheads, standing departments, or a single "
            "piece split at sub-headings may be counted as separate articles."
        )
    elif delta < 0:
        hints.append(
            f"You produced {abs(delta)} fewer article(s) than expected. Investigate "
            "dropped or merged items: letters/columns folded together, short notices "
            "skipped, or a continuation absorbed into the wrong article."
        )

    # precision — over-segmentation made visible to the aggregate
    prec = components.get("precision", {}).get("score", 1.0)
    if prec < 0.95 and pred_articles:
        n_extra = len(pred_articles) - len({i for i, _, _ in matches})
        hints.append(
            f"{n_extra} of {len(pred_articles)} predicted article(s) matched nothing expected "
            f"(precision {prec:.2f}). These phantom articles now cost aggregate score directly. "
            "Tighten segmentation so ads, mastheads, standing departments and sub-headed "
            "fragments are not emitted as separate articles."
        )

    # metadata, per field
    for f in per_field:
        if f["score"] >= 0.999:
            continue
        name = f["name"]
        if not f["present"]:
            hints.append(f"Metadata '{name}' is not populated — locate and extract it.")
        elif f["raw"] >= 0.6:
            hints.append(
                f"Metadata '{name}' is close but not matching — likely abbreviation, "
                "OCR character errors, or truncation. Re-examine the source region."
            )
        else:
            hints.append(
                f"Metadata '{name}' is materially different from expected. Re-extract it "
                "and verify you are reading the correct masthead/colophon region."
            )

    # titles
    below = sum(
        1 for i, j, _ in matches
        if title_similarity(pred_articles[i].get("title", ""), truth_articles[j].get("title", ""))
        < band.get("title", 0.85)
    )
    if below:
        hints.append(
            f"{below} of {len(matches)} aligned title(s) diverge from expected. Common "
            "causes: dropping/adding a leading article, including or omitting a subtitle, "
            "or capturing a section/department header instead of the article title."
        )

    # text — flag the weakest few articles
    weak = sorted((s, i) for i, s in text_pairs if s < 0.6)[:3]
    for s, i in weak:
        hints.append(
            f"Article #{i}'s body diverges materially from expected (text similarity "
            f"{s:.2f}). Check for truncation, a missed continuation onto a later page, "
            "or inclusion of neighbouring column text."
        )

    # order
    if components["order"]["score"] < 0.95 and matches:
        hints.append(
            f"Reading order diverges (order score {components['order']['score']:.2f}). "
            "Re-check the multi-column reading sequence and where continued articles are "
            "placed relative to items that start later."
        )

    # pages
    if components["pages"]["score"] < 0.999 and matches:
        offset = _page_offset(matches, pred_articles, truth_articles)
        if offset is not None:
            hints.append(
                "Page numbers appear consistently offset by a fixed amount. Verify the "
                "printed-folio vs PDF-page-index convention and whether front matter/cover "
                "is counted."
            )
        else:
            hints.append(
                "Some page assignments differ from expected. Re-check continuation page "
                "handling and the page-numbering convention."
            )

    # missing-article location (structural, gated): where to look, not what is there
    if allow_structural:
        matched_truth = {j for _, j, _ in matches}
        missing_pages = sorted(
            {p for k, a in enumerate(truth_articles) if k not in matched_truth
             for p in (a.get("pages") or [])}
        )
        if missing_pages:
            page_str = ", ".join(str(p) for p in missing_pages)
            hints.append(f"Expected article(s) you did not match appear on page(s): {page_str}.")

    return hints


# --------------------------------------------------------------------------- #

def evaluate(
    prediction: dict[str, Any],
    truth: dict[str, Any],
    weights: dict[str, float],
    allow_structural_hints: bool,
    *,
    bands: dict[str, float] | None = None,
    alignment_floor: float = 0.0,
) -> dict[str, Any]:
    band = {**_DEFAULT_BANDS, **(bands or {})}
    pred_articles = prediction.get("articles") or []
    truth_articles = truth.get("articles") or []
    matches = align_articles(pred_articles, truth_articles, floor=alignment_floor)

    schema = _schema_validity(prediction)
    count_score, delta = _article_count(prediction, truth)
    meta_score, matched, total, per_field = _metadata(prediction, truth, band["metadata"])
    titles = _titles_score(matches, pred_articles, truth_articles, band["title"])
    text_pairs = _text_pair_scores(matches, pred_articles, truth_articles)
    text = _text_score(text_pairs)
    n_articles = max(len(pred_articles), len(truth_articles), 1)
    order = _order_score(matches, n_articles)
    pages = _pages_score(matches, pred_articles, truth_articles)
    precision = _precision_score(pred_articles, matches)

    components = {
        "schema_validity": {"score": schema},
        "article_count":   {"score": count_score, "delta": delta},
        "precision":       {"score": precision},
        "metadata":        {"score": meta_score, "matched": matched, "total": total},
        "titles":          {"score": titles},
        "text":            {"score": text},
        "order":           {"score": order},
        "pages":           {"score": pages},
    }

    aggregate = sum(weights.get(k, 0.0) * components[k]["score"] for k in components)

    errors = _categorical_errors(prediction, truth, matches)
    hints = _component_hints(
        prediction, truth, matches, components, per_field, text_pairs, band, allow_structural_hints
    )

    return {
        "aggregate": round(aggregate, 4),
        "components": components,
        "categorical_errors": errors,
        "hints": hints,
    }
