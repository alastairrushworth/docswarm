"""Marking-mode tests with the LLM client stubbed out."""
from __future__ import annotations

from unittest.mock import patch

from judge import marking


PRED = {
    "magazine": {
        "editor": "L. J. Berger",
        "publisher": {"name": "N. H. Van Sicklen", "address": "57 Plymouth Place"},
    },
    "articles": [
        {"title": "That's So!", "text": ["paragraph"], "pages": [1], "kind": "prose"},
    ],
}
TRUTH = {
    "magazine": {
        "editor": "L. J. Berger",
        "publisher": {"name": "N. H. Van Sicklen", "address": "57 Plymouth Place, Chicago"},
    },
    "articles": [
        {"title": "That's So!", "text": ["paragraph one", "paragraph two"], "pages": [1], "kind": "prose"},
    ],
}


def test_missing_question_short_circuits():
    out = marking.evaluate(PRED, TRUTH, {"focus": {"path": "articles[0].text"}})
    assert out["verdict"] == "unverifiable"
    assert "question" in out["feedback"]


def test_missing_path_short_circuits():
    out = marking.evaluate(PRED, TRUTH, {"question": "is it ok?"})
    assert out["verdict"] == "unverifiable"
    assert "path" in out["feedback"]


def test_llm_response_passes_through():
    fake = {
        "verdict": "incomplete",
        "feedback": "the body is shorter than truth",
        "suggested_focus_path": "articles[0].text",
    }
    with patch("judge.marking.llm_client.chat_json", return_value=fake):
        out = marking.evaluate(
            PRED,
            TRUTH,
            {
                "question": "is the body complete?",
                "focus": {"path": "articles[0].text", "value": ["paragraph"]},
            },
        )
    assert out == fake


def test_llm_unavailable_yields_unverifiable():
    with patch("judge.marking.llm_client.chat_json", side_effect=ConnectionError("nope")):
        out = marking.evaluate(
            PRED,
            TRUTH,
            {
                "question": "is the editor right?",
                "focus": {"path": "magazine.editor", "value": "L. J. Berger"},
            },
        )
    assert out["verdict"] == "unverifiable"
    assert "judge LLM unavailable" in out["feedback"]


def test_invalid_verdict_coerced():
    fake = {"verdict": "great!", "feedback": "looks fine", "suggested_focus_path": None}
    with patch("judge.marking.llm_client.chat_json", return_value=fake):
        out = marking.evaluate(
            PRED,
            TRUTH,
            {
                "question": "?",
                "focus": {"path": "magazine.editor", "value": "L. J. Berger"},
            },
        )
    assert out["verdict"] == "unverifiable"
