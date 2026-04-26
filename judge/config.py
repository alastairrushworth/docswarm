"""Config loader for the judge package — same lookup contract as the module's loader."""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml


def _config_path() -> Path | None:
    env = os.environ.get("DOCSWARM_CONFIG")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "config.yaml"
        if candidate.is_file():
            return candidate
    workspace = Path("/workspace/config.yaml")
    return workspace if workspace.is_file() else None


@functools.lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    p = _config_path()
    if p is None:
        return {}
    with p.open() as f:
        return yaml.safe_load(f) or {}


def get(path: str, default: Any = None) -> Any:
    cur: Any = load()
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
