"""Per-page intermediate caching, keyed by content hash of the PDF."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from .config import get


def pdf_content_hash(pdf_path: str | Path) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_root() -> Path:
    root = Path(get("paths.cache_dir", ".cache/pdf_to_json"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _key(pdf_hash: str, page_index: int, model_tag: str, prompt_version: str) -> str:
    raw = f"{pdf_hash}|{page_index}|{model_tag}|{prompt_version}".encode()
    return hashlib.sha256(raw).hexdigest()


def _path_for(key: str) -> Path:
    return _cache_root() / f"{key}.json"


def load(pdf_hash: str, page_index: int, model_tag: str, prompt_version: str) -> Any | None:
    p = _path_for(_key(pdf_hash, page_index, model_tag, prompt_version))
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def store(pdf_hash: str, page_index: int, model_tag: str, prompt_version: str, value: Any) -> None:
    """Atomic write: serialize → write to .tmp.<uuid> → os.replace.

    Concurrent writers don't trample each other; killed processes can't leave
    a torn-but-parseable file in the cache.
    """
    p = _path_for(_key(pdf_hash, page_index, model_tag, prompt_version))
    tmp = p.with_suffix(p.suffix + f".tmp.{uuid.uuid4().hex}")
    blob = json.dumps(value)
    tmp.write_text(blob)
    os.replace(tmp, p)
