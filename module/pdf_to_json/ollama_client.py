"""Thin Ollama HTTP client (vision + text + embedding)."""
from __future__ import annotations

import base64
import logging
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


def generate(
    model: str,
    prompt: str,
    images: list[str | Path] | None = None,
    *,
    timeout: float = 60.0,
    options: dict[str, Any] | None = None,
) -> str:
    img_count = len(images) if images else 0
    logger.info("generate  model=%s images=%d  prompt=%s", model, img_count, _t(prompt))
    payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
    if images:
        payload["images"] = [_b64_image(p) for p in images]
    if options:
        payload["options"] = options
    r = httpx.post(f"{_url()}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    result = r.json().get("response", "")
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
