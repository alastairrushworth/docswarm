"""Marking mode: freeform LLM-driven Q&A about a focused JSON slice.

The judge sees only the predicted JSON and the truth JSON — never the source PDF.
Feedback is JSON-relative: discrepancies between two JSON documents, never
layout-relative.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import yaml

from . import llm_client
from .path_resolver import resolve

logger = logging.getLogger("judge.marking")

FOCUS_VALUE_BYTE_CAP = 8 * 1024  # 8KB

SYSTEM_PROMPT = """\
You are a feedback grader for an automated PDF-to-JSON translator. You hold the
ground-truth JSON for the document under evaluation. The translator's developer
agent will ask freeform questions about specific JSON slices and you will guide
them.

You see TWO JSON documents only — the predicted slice and the truth slice. You
do NOT have access to the source PDF. Do not reason about page layout, columns,
margins, image regions, or anything visual.

HARD RULES — non-negotiable:
1. Never quote verbatim from the truth. Never paraphrase a sentence from the
   truth. If you do, your output will be redacted.
2. Speak in JSON-relative, qualitative terms: "the body is materially shorter
   than truth", "this article is verse, not prose", "the publisher address
   appears truncated", "the date does not match".
3. Suggest WHERE in the JSON to look (`articles[i].text`, `magazine.editor`),
   not WHAT TO WRITE.
4. If the prediction is correct at this slice, say so plainly with verdict
   "correct".
5. If the focus slice is too small to assess (e.g. a single article body but
   the question is about ordering), set verdict "unverifiable" and ask for a
   different focus.
6. Stay under 80 words in `feedback`.

Respond with a single JSON object, no prose outside it:
{
  "verdict": "correct" | "incomplete" | "wrong" | "unverifiable",
  "feedback": "string, ≤80 words",
  "suggested_focus_path": "articles[i].text" | "magazine.publisher" | null
}
"""


def _load_judge_model() -> str:
    cfg_path = os.environ.get("DOCSWARM_CONFIG", "/workspace/config.yaml")
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("models", {}).get("judge", "qwen2.5:32b")
    except FileNotFoundError:
        return "qwen2.5:32b"


def _shrink(value: Any, cap: int = FOCUS_VALUE_BYTE_CAP) -> tuple[Any, bool]:
    """Truncate a focus value if it exceeds the byte cap."""
    blob = json.dumps(value, ensure_ascii=False)
    if len(blob.encode("utf-8")) <= cap:
        return value, False
    if isinstance(value, str):
        return value.encode("utf-8")[:cap].decode("utf-8", errors="ignore"), True
    if isinstance(value, list):
        out: list = []
        size = 2
        for item in value:
            blob = json.dumps(item, ensure_ascii=False)
            if size + len(blob.encode("utf-8")) > cap:
                break
            out.append(item)
            size += len(blob.encode("utf-8")) + 1
        return out, True
    return value, False


def _build_user_prompt(
    question: str,
    path: str,
    pred_slice: Any,
    truth_slice: Any,
    notes: str,
) -> str:
    return (
        f"PATH: {path}\n"
        f"NOTES: {notes or '(none)'}\n\n"
        f"PREDICTED SLICE:\n{json.dumps(pred_slice, indent=2, ensure_ascii=False)}\n\n"
        f"TRUTH SLICE:\n{json.dumps(truth_slice, indent=2, ensure_ascii=False)}\n\n"
        f"QUESTION FROM THE TRANSLATOR:\n{question}\n\n"
        "Respond with the JSON object specified by the system prompt."
    )


def _coerce_response(raw: dict[str, Any]) -> dict[str, Any]:
    verdict = raw.get("verdict", "unverifiable")
    if verdict not in {"correct", "incomplete", "wrong", "unverifiable"}:
        verdict = "unverifiable"
    feedback = str(raw.get("feedback", "")).strip()
    suggested = raw.get("suggested_focus_path")
    if suggested is not None and not isinstance(suggested, str):
        suggested = None
    return {
        "verdict": verdict,
        "feedback": feedback,
        "suggested_focus_path": suggested,
    }


def evaluate(
    prediction: dict[str, Any],
    truth: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    """Freeform marking via the judge LLM, JSON-only."""
    question = (request.get("question") or "").strip()
    focus = request.get("focus") or {}
    path = focus.get("path", "")

    if not question:
        return {
            "verdict": "unverifiable",
            "feedback": "no question provided",
            "suggested_focus_path": None,
        }
    if not path:
        return {
            "verdict": "unverifiable",
            "feedback": "no focus.path provided",
            "suggested_focus_path": None,
        }

    resolved = resolve(path, prediction, truth)
    pred_slice = focus.get("value", resolved["pred_value"])
    truth_slice = resolved["truth_value"]
    notes = resolved["notes"]

    pred_slice, pred_truncated = _shrink(pred_slice)
    truth_slice, truth_truncated = _shrink(truth_slice)
    if pred_truncated or truth_truncated:
        notes = (notes + "; slice truncated to fit byte cap").strip("; ")

    model = _load_judge_model()
    user = _build_user_prompt(question, path, pred_slice, truth_slice, notes)
    try:
        raw = llm_client.chat_json(
            model=model,
            system=SYSTEM_PROMPT,
            user=user,
            timeout=60.0,
            options={"temperature": 0.2},
        )
    except Exception as e:
        logger.warning("judge LLM call failed: %s", e)
        return {
            "verdict": "unverifiable",
            "feedback": f"judge LLM unavailable: {e}",
            "suggested_focus_path": None,
        }
    return _coerce_response(raw)
