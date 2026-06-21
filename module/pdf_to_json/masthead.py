"""Dedicated masthead / colophon metadata extraction from the first page of a magazine.

Uses a focused vision prompt that instructs the model to read ONLY the masthead/colophon
(typically at top or bottom edge of page 0) and returns every identifiable field.
The result is post-processed by ``parse_masthead`` before reaching the schema layer.
"""

from __future__ import annotations

import re
from typing import Any

PROMPT_VERSION = "v1"

_MASTHEAD_PROMPT = """\
You are reading the masthead / colophon of a vintage magazine — typically at the
top or bottom edge of the first page (often on a decorative banner).  Return ONLY
a JSON object with these fields.  Omit any field you cannot identify (use ``null``).

{
  "editor":       "Name of the editor, exactly as written",
  "volume":       "Volume number — may be Roman numerals (e.g. CI) or Arabic (5)",
  "number":       "Issue / edition number as a string or integer",
  "date":         "Publication date in any format (e.g. 'June 25, 1941' or '3d.')",
  "publisher_name":     "Name of the publisher / company",
  "publisher_address":  "Publisher street address OR city/state/country line",
  "cost_issue":     "Price per single copy as a string (e.g. '3d', '$0.25')",
  "cost_annual":    "Annual subscription price as a string, if listed",
  "cost_semiannual": "Semi-annual / six-month price as a string, if listed"
}

RULES:
- `volume` should be the raw text from the page (e.g. "CI" or "5").
- If publisher is a single line "Name, Address" do NOT split it yourself; put
  the full line in `publisher_name` and leave `publisher_address` empty.
"""


# --------------------------------------------------------------------------- #
# JSON parsing helpers
# --------------------------------------------------------------------------- #

def _try_json(raw: str) -> dict[str, Any] | None:
    """Try to extract a JSON object from the model's raw text response."""
    import json
    raw = raw.strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    brace = 0
    start = -1
    for i, ch in enumerate(raw):
        if ch == '{':
            if brace == 0:
                start = i
            brace += 1
        elif ch == '}':
            brace -= 1
            if brace == 0 and start >= 0:
                try:
                    return json.loads(raw[start:i + 1])
                except (json.JSONDecodeError, ValueError):
                    pass
    return None


def _str(v: Any) -> str | None:
    """Return stripped string value or None."""
    if v is None:
        return None
    s = str(v).strip()
    # Return empty string as empty string, not None (for consistency)
    return s


# --------------------------------------------------------------------------- #
# Publisher splitting
# --------------------------------------------------------------------------- #

def parse_publisher_from_full_line(full_line: str) -> tuple[str | None, str | None]:
    """Split a comma-separated publisher line into (name, address).

    Examples:
        "Temple Press Ltd., Bowling Green Lane, London, E.C.1"
            → ("Temple Press Ltd.", "Bowling Green Lane, London, E.C.1")
    """
    if not full_line or "," not in full_line:
        return None, None
    parts = full_line.split(",")
    name = parts[0].strip()
    addr = ", ".join(p.strip() for p in parts[1:]) if len(parts) > 1 else ""
    name = re.sub(r"\.\s*$", "", name)
    return name or None, addr or None


# --------------------------------------------------------------------------- #
# Roman / arabic volume parsing
# --------------------------------------------------------------------------- #

_ROMAN_MAP = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}


def roman_to_int(s: str) -> int | None:
    """Convert a Roman numeral string to integer, or None."""
    s = s.upper().strip()
    if not s or not all(c in _ROMAN_MAP for c in s):
        return None
    result = 0
    prev = 0
    for ch in reversed(s):
        val = _ROMAN_MAP[ch]
        if val < prev:
            result -= val
        else:
            result += val
        prev = val
    return result or None


def parse_volume(raw: str) -> int | None:
    """Parse volume from various formats (Arabic, Roman, mixed)."""
    raw = raw.strip()
    try:
        return int(raw)
    except ValueError:
        pass
    result = roman_to_int(raw)
    if result and result > 0:
        return result
    m = re.search(r'(\d+)', raw)
    return int(m.group(1)) if m else None


def parse_number(raw: str) -> int | None:
    """Parse issue/edition number (often a long integer like 2630)."""
    raw = raw.strip()
    try:
        return int(raw)
    except ValueError:
        pass
    m = re.search(r'(\d+)', raw)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def extract_masthead(raw: str) -> dict[str, Any]:
    """Parse the vision model's raw masthead response into a metadata dict.

    Handles both old-style fields (volume/number/date) and new-style
    (volume_raw/number_raw/date_raw). Also supports publisher_full_line for
    cases where name+address were not separated by the vision model.
    """
    data = _try_json(raw)
    if not data:
        return {}

    result: dict[str, Any] = {}

    # Editor
    val = _str(data.get("editor"))
    if val:
        result["editor"] = val

    # Volume — try new-style first, fall back to old
    vol_raw = _str(data.get("volume_raw")) or _str(data.get("volume"))
    if vol_raw:
        result["volume_raw"] = vol_raw

    # Number — try new-style first, fall back to old
    num_raw = _str(data.get("number_raw")) or _str(data.get("number"))
    if num_raw:
        result["number_raw"] = num_raw

    # Date — try new-style first, fall back to old
    date_raw = _str(data.get("date_raw")) or _str(data.get("date"))
    if date_raw:
        result["date_raw"] = date_raw

    # Cost fields
    for key in ("cost_issue", "cost_annual", "cost_semiannual"):
        val = _str(data.get(key))
        if val:
            result[key] = val

    # Publisher fields — two patterns: separate name/address OR full_line.
    raw_name = _str(data.get("publisher_name")) or ""
    raw_addr = _str(data.get("publisher_address")) or ""
    full_line = _str(data.get("publisher_full_line")) or raw_name

    # Convert empty strings to None for proper filtering in pipeline._build_metadata()
    if not raw_name:
        raw_name = None
    if not raw_addr:
        raw_addr = None

    if raw_name:
        result["publisher_name"] = str(raw_name).strip()
    if raw_addr:
        result["publisher_address"] = str(raw_addr).strip()
    elif full_line and "," in full_line:
        name, addr = parse_publisher_from_full_line(full_line)
        if name and not result.get("publisher_name"):
            result["publisher_name"] = name
        if addr and not result.get("publisher_address"):
            result["publisher_address"] = addr

    return result


def _to_date(raw: str) -> Any:  # date | None
    """Try to turn a free-form date string into a ``datetime.date``."""
    from datetime import date as dt_date
    if not raw:
        return None
    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "jun": 6, "jul": 7, "aug": 8,
        "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    raw_lower = raw.lower().strip()
    for name, num in month_map.items():
        if raw_lower.startswith(name):
            rest = raw[len(name):].strip().lstrip("., ")
            m = re.search(r'(\d{1,4})', rest)
            day_m = re.match(r'(\d{1,2})', rest)
            year = int(m.group(1)) if m else None
            day = int(day_m.group(1)) if day_m else 1
            if year and (year > 1800 or year < 2100):
                return dt_date(year, num, day)
    try:
        return dt_date.fromisoformat(raw)
    except (ValueError, TypeError):
        pass
    return None
