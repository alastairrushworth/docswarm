"""String / text similarity helpers.

Default to a fast token-Jaccard similarity. If an Ollama embedding model is
reachable, text similarity falls back to cosine of embedding vectors.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

import httpx
import numpy as np

from .config import get

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


def _ollama_url() -> str:
    env = os.environ.get("OLLAMA_URL", "")
    if env:
        return env.rstrip("/")
    return str(get("ollama.url", "http://localhost:11434")).rstrip("/")


def _embed_model() -> str:
    return str(get("models.embedding", "nomic-embed-text"))


# Embedding circuit breaker. The article-alignment matrix embeds every unique
# article text; a stalled/cold embed model must not turn one broad eval into
# N+M serial timeouts (that overran the harness's 600s judge window). So: cache
# only successes (a failure is never cached, so it can self-heal), and on any
# failure open a short cooldown during which we score with jaccard instead of
# paying the per-call timeout again.
_embed_cache: dict[str, tuple[float, ...]] = {}
_embed_blocked_until = 0.0


def _embed_timeout() -> float:
    return float(get("judge.embed_timeout_seconds", 8.0))


def _embed_cooldown() -> float:
    return float(get("judge.embed_cooldown_seconds", 60.0))


def _embed(text: str) -> tuple[float, ...] | None:
    global _embed_blocked_until
    text = (text or "").strip()
    if not text:
        return None
    hit = _embed_cache.get(text)
    if hit is not None:
        return hit
    if time.monotonic() < _embed_blocked_until:
        return None
    try:
        r = httpx.post(
            f"{_ollama_url()}/api/embeddings",
            json={"model": _embed_model(), "prompt": text},
            timeout=_embed_timeout(),
        )
        r.raise_for_status()
        v = r.json().get("embedding") or []
        if not v:
            return None
        vec = tuple(float(x) for x in v)
        _embed_cache[text] = vec
        return vec
    except Exception as e:
        cooldown = _embed_cooldown()
        _embed_blocked_until = time.monotonic() + cooldown
        logger.warning(
            "embeddings unavailable (%s); scoring text with jaccard for ~%.0fs", e, cooldown
        )
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
