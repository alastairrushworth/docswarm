"""Thin Ollama HTTP client (vision + text + embedding)."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx

from .config import get


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
    payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
    if images:
        payload["images"] = [_b64_image(p) for p in images]
    if options:
        payload["options"] = options
    r = httpx.post(f"{_url()}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "")


def embed(model: str, text: str, *, timeout: float = 30.0) -> list[float]:
    r = httpx.post(
        f"{_url()}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json().get("embedding", [])
