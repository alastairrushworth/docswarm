"""Judge-side Ollama HTTP client.

Reads the Ollama URL from config.yaml (`ollama.url`) — same instance the
translator and developer agent use. Different model tag, different prompt.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

import os

from .config import get

logger = logging.getLogger("judge.llm")

_TRACE_CHARS = 500


def _t(s: str) -> str:
    return s[:_TRACE_CHARS] + "…" if len(s) > _TRACE_CHARS else s


def _ollama_url() -> str:
    env = os.environ.get("OLLAMA_URL", "")
    if env:
        return env.rstrip("/")
    return str(get("ollama.url", "http://localhost:11434")).rstrip("/")


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
    logger.info("chat  model=%s  system=%s  user=%s", model, _t(system), _t(user))
    r = httpx.post(f"{_ollama_url()}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    msg = data.get("message") or {}
    result = msg.get("content", "")
    logger.info("chat  response=%s", _t(result))
    return result


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
