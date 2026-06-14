"""Thin Ollama HTTP client (vision + text + embedding)."""
from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from .config import get

logger = logging.getLogger("pdf_to_json.ollama")

_TRACE_CHARS = 500


def _t(s: str) -> str:
    return s[:_TRACE_CHARS] + "…" if len(s) > _TRACE_CHARS else s


def _url() -> str:
    return get("ollama.url", "http://localhost:11434").rstrip("/")


def _b64_image(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def _log_metrics(model: str, data: dict[str, Any], wall_s: float) -> None:
    """Surface Ollama's own timing/token counts so slow or truncated calls are
    diagnosable straight from the `make run` stream. `done_reason="length"`
    means the output hit num_predict and was truncated (likely invalid JSON)."""
    out_tokens = data.get("eval_count")
    eval_dur_ns = data.get("eval_duration")
    tok_s = out_tokens / (eval_dur_ns / 1e9) if out_tokens and eval_dur_ns else None
    logger.info(
        "generate  model=%s wall=%.1fs in_tokens=%s out_tokens=%s tok/s=%s done=%s",
        model, wall_s, data.get("prompt_eval_count"), out_tokens,
        f"{tok_s:.1f}" if tok_s else "-", data.get("done_reason"),
    )


def generate(
    model: str,
    prompt: str,
    images: list[str | Path] | None = None,
    *,
    timeout: float = 60.0,
    options: dict[str, Any] | None = None,
) -> str:
    img_count = len(images) if images else 0
    opts = options or {}
    logger.info(
        "generate  model=%s images=%d num_ctx=%s num_predict=%s timeout=%.0fs  prompt=%s",
        model, img_count, opts.get("num_ctx", "-"), opts.get("num_predict", "-"),
        timeout, _t(prompt),
    )
    payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
    if images:
        payload["images"] = [_b64_image(p) for p in images]
    if options:
        payload["options"] = options
    t0 = time.monotonic()
    r = httpx.post(f"{_url()}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    result = data.get("response", "")
    _log_metrics(model, data, time.monotonic() - t0)
    logger.info("generate  response=%s", _t(result))
    return result


def embed(model: str, text: str, *, timeout: float = 30.0) -> list[float]:
    r = httpx.post(
        f"{_url()}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json().get("embedding", [])
