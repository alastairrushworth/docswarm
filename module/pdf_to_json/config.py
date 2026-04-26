"""Config loader. Hardcoded constants in code are forbidden — read from here."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _config_path() -> Path:
    env = os.environ.get("DOCSWARM_CONFIG")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "config.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("config.yaml not found; set DOCSWARM_CONFIG to override.")


_cache: dict[str, Any] | None = None


def load_config() -> dict[str, Any]:
    global _cache
    if _cache is None:
        with _config_path().open() as f:
            _cache = yaml.safe_load(f) or {}
    return _cache


def get(path: str, default: Any = None) -> Any:
    """Dotted-path getter, e.g. get('iteration.page_budget_seconds')."""
    cur: Any = load_config()
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
