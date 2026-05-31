"""Tests for surface-form normalization and fuzzy field similarity."""
from judge.normalize import field_similarity, normalize


def test_normalize_strips_punct_case_whitespace():
    assert normalize("L. J. Berger") == "l j berger"
    assert normalize("  57  Plymouth Place, Chicago ") == "57 plymouth place chicago"
    assert normalize(None) == ""


def test_punctuation_only_difference_is_equivalent():
    # The brittleness the judge must NOT punish: spacing/period variants are equal,
    # and apostrophe/punctuation noise stays well above the metadata band (0.90).
    assert field_similarity("L. J. Berger", "L.J. Berger") == 1.0
    assert field_similarity("57 Plymouth Place", "57 plymouth place") == 1.0
    assert field_similarity("That's So!", "Thats So") >= 0.90


def test_truncation_loses_some_credit_but_not_all():
    s = field_similarity("57 Plymouth Place, Chicago", "57 Plymouth Place")
    assert 0.5 < s < 1.0


def test_materially_different_scores_low():
    assert field_similarity("N. H. Van Sicklen", "Acme Publishing Co.") < 0.4


def test_empty_vs_present():
    assert field_similarity("", "something") == 0.0
    assert field_similarity("", "") == 1.0
