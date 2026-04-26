from judge.leakage_filter import collect_truth_strings, filter_hints, overlap_ratio


TRUTH = {
    "magazine": {
        "editor": "L. J. Berger",
        "publisher": {"name": "N. H. Van Sicklen"},
    },
    "articles": [{"title": "That's So!", "text": ["It is not always the man who rides the swiftest"]}],
}


def test_collect_truth_strings():
    s = collect_truth_strings(TRUTH)
    assert "L. J. Berger" in s
    assert "It is not always the man who rides the swiftest" in s


def test_overlap_high_when_quoting_truth():
    truth_strings = collect_truth_strings(TRUTH)
    leak = "the man who rides the swiftest is missing"
    assert overlap_ratio(leak, truth_strings, n=4) > 0.0


def test_filter_redacts_high_overlap():
    truth_strings = collect_truth_strings(TRUTH)
    out = filter_hints(["one missing article"], truth_strings, 0.3)
    assert out == ["one missing article"]
    out = filter_hints(["the man who rides the swiftest"], truth_strings, 0.3)
    assert "redacted" in out[0]
