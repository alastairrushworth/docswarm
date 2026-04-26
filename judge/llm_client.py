"""Judge-side Ollama HTTP client.

Reads the Ollama URL from config.yaml (`ollama.url`) — same instance the
translator and developer agent use. Different model tag, different prompt.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from .config import get

logger = logging.getLogger("judge.llm")


def _ollama_url() -> str:
    return str(get("ollama.url", "http://ollama-main:11434")).rstrip("/")


def chat(
    model: str,
    system: str,
    user: str,
    *,
    timeout: float = 60.0,
    options: dict[str, Any] | None = None,
    response_format_json: bool = True,
) -> str:
    """Single-turn chat. Returns the assistant text."""
    payload: dict[str, Any] = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if options:
        payload["options"] = options
    if response_format_json:
        payload["format"] = "json"
    r = httpx.post(f"{_ollama_url()}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    msg = data.get("message") or {}
    return msg.get("content", "")


def chat_json(
    model: str,
    system: str,
    user: str,
    *,
    timeout: float = 60.0,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = chat(model, system, user, timeout=timeout, options=options, response_format_json=True)
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
        logger.warning("judge LLM returned non-JSON: %.200s", raw)
        return {}
