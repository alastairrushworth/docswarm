"""String / text similarity helpers.

Default to a fast token-Jaccard similarity. If an Ollama embedding model is reachable,
text similarity falls back to cosine of `nomic-embed-text` embeddings.
"""
from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any

import httpx
import numpy as np

logger = logging.getLogger("judge.similarity")

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _tokens(s: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(s or "")}


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _judge_ollama_url() -> str:
    return os.environ.get("OLLAMA_JUDGE_URL", "http://ollama-judge:11434").rstrip("/")


def _embed_model() -> str:
    return os.environ.get("JUDGE_EMBED_MODEL", "nomic-embed-text")


@lru_cache(maxsize=512)
def _embed(text: str) -> tuple[float, ...] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        r = httpx.post(
            f"{_judge_ollama_url()}/api/embeddings",
            json={"model": _embed_model(), "prompt": text},
            timeout=15.0,
        )
        r.raise_for_status()
        v = r.json().get("embedding") or []
        if not v:
            return None
        return tuple(float(x) for x in v)
    except Exception as e:
        logger.debug("embed unavailable, falling back to jaccard: %s", e)
        return None


def text_similarity(a: str, b: str) -> float:
    va = _embed(a)
    vb = _embed(b)
    if va is None or vb is None:
        return jaccard(a, b)
    a_arr = np.asarray(va)
    b_arr = np.asarray(vb)
    denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def title_similarity(a: str, b: str) -> float:
    return jaccard(a, b)


def combined_similarity(pred: dict[str, Any], truth: dict[str, Any]) -> float:
    t = title_similarity(str(pred.get("title", "")), str(truth.get("title", "")))
    pred_text = " ".join(pred.get("text") or [])
    truth_text = " ".join(truth.get("text") or [])
    x = text_similarity(pred_text, truth_text)
    return 0.5 * t + 0.5 * x
