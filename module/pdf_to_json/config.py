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


def _apply_local_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get("DOCSWARM_MODE", "").lower() != "local":
        return cfg
    overrides = cfg.get("local") or {}
    for section_name, section_overrides in overrides.items():
        if not isinstance(section_overrides, dict):
            continue
        target = cfg.setdefault(section_name, {})
        for key, value in section_overrides.items():
            if value is None:
                target.pop(key, None)
            else:
                target[key] = value
    return cfg


_cache: dict[str, Any] | None = None


def load_config() -> dict[str, Any]:
    global _cache
    if _cache is None:
        with _config_path().open() as f:
            _cache = _apply_local_overrides(yaml.safe_load(f))
    return _cache


def get(path: str, default: Any = None) -> Any:
    """Dotted-path getter, e.g. get('iteration.page_budget_seconds')."""
    cur: Any = load_config()
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
