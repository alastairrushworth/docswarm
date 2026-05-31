"""Tests for essence-over-form broad scoring + non-leaking hints.

text_similarity falls back to token Jaccard when no embedding server is
reachable, so these run deterministically offline.
"""
import copy

from judge import broad
from judge.alignment import align_articles

WEIGHTS = {
    "schema_validity": 0.10, "article_count": 0.10, "metadata": 0.15,
    "titles": 0.15, "text": 0.30, "order": 0.10, "pages": 0.10,
}
BANDS = {"metadata": 0.90, "title": 0.85}
FLOOR = 0.15


def _truth():
    return {
        "magazine": {
            "editor": "L. J. Berger",
            "issue": {"date": "1892-06-03", "volume": 5, "number": 18},
            "publisher": {"name": "N. H. Van Sicklen", "address": "57 Plymouth Place, Chicago"},
            "cost": {},
        },
        "articles": [
            {"title": "That's So!", "text": ["alpha beta gamma delta"], "pages": [1], "kind": "prose"},
            {"title": "Both Were Pleased.", "text": ["epsilon zeta", "eta theta"], "pages": [2], "kind": "verse"},
        ],
    }


def _evaluate(pred, truth):
    return broad.evaluate(pred, truth, WEIGHTS, True, bands=BANDS, alignment_floor=FLOOR)


def test_punctuation_only_metadata_is_full_credit():
    truth = _truth()
    pred = copy.deepcopy(truth)
    pred["magazine"]["editor"] = "L.J. Berger"          # punctuation/spacing only
    pred["magazine"]["issue"]["volume"] = "5"           # string vs int — coerced
    res = _evaluate(pred, truth)
    assert res["components"]["metadata"]["score"] == 1.0
    assert res["components"]["metadata"]["matched"] == 6


def test_truncated_field_gets_partial_credit_and_a_hint():
    truth = _truth()
    pred = copy.deepcopy(truth)
    pred["magazine"]["publisher"]["address"] = "57 Plymouth Place"  # dropped ", Chicago"
    res = _evaluate(pred, truth)
    meta = res["components"]["metadata"]["score"]
    assert 0.85 < meta < 1.0  # not zero (old exact-match), not full
    assert any("publisher.address" in h for h in res["hints"])
    # ...but the hint must not reveal the missing value:
    assert not any("Chicago" in h for h in res["hints"])


def test_wrong_value_still_punished():
    truth = _truth()
    pred = copy.deepcopy(truth)
    pred["magazine"]["issue"]["date"] = "1893-06-03"  # wrong year
    res = _evaluate(pred, truth)
    # 5 fields full + date 0.0 → 5/6
    assert abs(res["components"]["metadata"]["score"] - 5 / 6) < 1e-6


def test_alignment_floor_drops_unrelated_pairs():
    truth = _truth()
    pred_articles = [
        {"title": "Completely Unrelated Heading", "text": ["qux quux corge"], "pages": [9], "kind": "prose"},
    ]
    # With no floor the single prediction is force-matched to a truth article.
    assert len(align_articles(pred_articles, truth["articles"], floor=0.0)) == 1
    # With the floor it is not — so it reads as 1 extra + 2 missing.
    assert align_articles(pred_articles, truth["articles"], floor=FLOOR) == []

    pred = {"magazine": truth["magazine"], "articles": pred_articles}
    res = _evaluate(pred, truth)
    cats = {e["category"]: e.get("count") for e in res["categorical_errors"]}
    assert cats.get("missing_article") == 2
    assert cats.get("extra_article") == 1


def test_title_band_snaps_near_matches():
    truth = _truth()
    pred = copy.deepcopy(truth)
    # exact-but-for-punctuation title should land at full credit via the band
    pred["articles"][0]["title"] = "That's So"   # dropped "!"
    res = _evaluate(pred, truth)
    assert res["components"]["titles"]["score"] == 1.0


def test_hints_never_leak_truth_content():
    truth = _truth()
    pred = {
        "magazine": {
            "editor": "Wrong Person",
            "issue": {"date": "1900-01-01", "volume": 99, "number": 1},
            "publisher": {"name": "Other Co", "address": "Somewhere"},
            "cost": {},
        },
        "articles": [
            {"title": "Mismatched", "text": ["nothing in common here"], "pages": [5], "kind": "prose"},
        ],
    }
    res = _evaluate(pred, truth)
    assert res["hints"]  # something should be flagged
    blob = " ".join(res["hints"])
    for secret in ["Van Sicklen", "Plymouth", "Berger", "Both Were Pleased", "alpha beta"]:
        assert secret not in blob
