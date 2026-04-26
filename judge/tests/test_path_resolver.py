from judge.path_resolver import resolve


PRED = {
    "magazine": {
        "editor": "L. J. Berger",
        "publisher": {"name": "N. H. Van Sicklen", "address": "57 Plymouth Place"},
    },
    "articles": [
        {"title": "That's So!", "text": ["a", "b"], "pages": [1], "kind": "prose"},
        {"title": "Both Were Pleased.", "text": ["x", "y"], "pages": [1], "kind": "verse"},
    ],
}
TRUTH = {
    "magazine": {
        "editor": "L. J. Berger",
        "publisher": {"name": "N. H. Van Sicklen", "address": "57 Plymouth Place, Chicago"},
    },
    "articles": [
        {"title": "That's So!", "text": ["aa", "bb"], "pages": [1], "kind": "prose"},
        {"title": "Both Were Pleased.", "text": ["xx", "yy"], "pages": [1], "kind": "verse"},
    ],
}


def test_resolve_metadata_field():
    r = resolve("magazine.publisher.address", PRED, TRUTH)
    assert r["kind"] == "metadata"
    assert r["pred_value"] == "57 Plymouth Place"
    assert r["truth_value"] == "57 Plymouth Place, Chicago"


def test_resolve_article_full():
    r = resolve("articles[0]", PRED, TRUTH)
    assert r["kind"] == "article"
    assert r["pred_value"]["title"] == "That's So!"
    assert r["truth_value"]["title"] == "That's So!"


def test_resolve_article_subpath():
    r = resolve("articles[1].text", PRED, TRUTH)
    assert r["pred_value"] == ["x", "y"]
    assert r["truth_value"] == ["xx", "yy"]


def test_resolve_unknown_path():
    r = resolve("nonsense", PRED, TRUTH)
    assert r["kind"] == "unknown"


def test_resolve_out_of_range_article():
    r = resolve("articles[99]", PRED, TRUTH)
    assert r["pred_value"] is None
    assert "no predicted article" in r["notes"]
