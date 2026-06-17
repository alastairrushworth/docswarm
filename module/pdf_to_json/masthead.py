"""Masthead / colophon metadata extraction from the first page of a magazine PDF.

Uses a dedicated vision prompt that explicitly instructs the model to read
the masthead (usually at top or bottom of page 0) and return every identifiable
field — editor, volume, issue number, date, publisher name+address, pricing.

The prompt is designed so that even partial / messy OCR output produces usable
fields; post-processing in `parse_publisher()` cleans up the result before it
reaches the schema layer.
"""

from __future__ import annotations

import re
from typing import Any

PROMPT_VERSION = "v1"

_MASTHEAD_PROMPT = """\
You are reading the masthead / colophon of a vintage magazine — typically at the
top or bottom edge of the first page (often on a decorative banner).  Return ONLY
a JSON object with these fields.  Omit any field you cannot identify.

{
  "editor":       "Name of the editor",
  "volume":       "Volume number — may be Roman numerals (e.g. CI) or Arabic (5)",
  "number":       "Issue / edition number as a string or integer",
  "date":         "Publication date in any format (try to return YYYY-MM-DD, but \
any recognizable date is fine)",
  "publisher_name":     "Name of the publisher / company",
  "publisher_address":  "Publisher street address OR city/state/country line",
  "cost_issue":     "Price per single copy as a string (e.g. '3d', '$0.25')",
  "cost_annual":    "Annual subscription price as a string, if listed",
  "cost_semiannual": "Semi-annual / six-month price as a string, if listed",
  "publisher_full_line": "Full publisher line exactly as it appears (if you found \
a comma-separated line like 'Name, Address')"
}

RULES:
- `volume` should be the raw text from the page (e.g. "CI" or "5").
- `number` should be the raw issue/edition number.
- If publisher is a single line "Name, Address" do NOT split it yourself; put
  the full line in `publisher_full_line` and leave `publisher_name` /
  `publisher_address` empty.
- Read carefully — the masthead is small but legible. Look at top AND bottom of
  the page if needed.
"""


def extract_masthead(raw: str) -> dict[str, Any]:
    """Parse the vision model's raw text response into a metadata dict.

    Heuristics handle common vintage-magazine formatting (Roman numerals for
    volume, comma-separated publisher lines, etc.).
    """
    import json

    # Try direct JSON parse first
    data = _try_json(raw)
    if not data:
        return {}

    result: dict[str, Any] = {}

    # Map field names from the model's response to our internal keys
    for key in ("editor", "volume", "number", "date",
                "cost_issue", "cost_annual", "cost_semiannual"):
        val = data.get(key)
        if val is not None and str(val).strip():
            result[key] = str(val).strip()

    # Publisher fields — two possible patterns:
    # Pattern A: separate `publisher_name` / `publisher_address`
    # Pattern B: single `publisher_full_line` like "Temple Press Ltd., Bowling Green Lane, London"
    raw_name = data.get("publisher_name", "")
    raw_addr = data.get("publisher_address", "")
    full_line = data.get("publisher_full_line", "")

    if isinstance(full_line, str) and full_line.strip():
        result["publisher_full_line"] = full_line.strip()
        # If no separate fields, keep full_line for the pipeline to parse
        if not raw_name or not raw_addr:
            result["_publisher_raw"] = full_line.strip()

    if isinstance(raw_name, str) and raw_name.strip():
        result["publisher_name"] = raw_name.strip()
    if isinstance(raw_addr, str) and raw_addr.strip():
        result["publisher_address"] = raw_addr.strip()

    return result


def _try_json(raw: str) -> dict | None:
    """Try to extract JSON from the model's text response."""
    import json
    raw = raw.strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    # Try to find a JSON object in the text
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
                    return json.loads(raw[start:i+1])
                except (json.JSONDecodeError, ValueError):
                    pass
    return None


def parse_publisher_from_full_line(full_line: str) -> tuple[str | None, str | None]:
    """Split a comma-separated publisher line into name and address.

    Examples:
        "Temple Press Ltd., Bowling Green Lane, London, E.C.1"
            → ("Temple Press Ltd.", "Bowling Green Lane, London, E.C.1")
        "N. H. Van Sicklen, 57 Plymouth Place, Chicago"
            → ("N. H. Van Sicklen", "57 Plymouth Place, Chicago")
    """
    if not full_line or "," not in full_line:
        return None, None

    parts = full_line.split(",")
    name = parts[0].strip()
    addr = ", ".join(p.strip() for p in parts[1:]) if len(parts) > 1 else ""
    # Clean up trailing punctuation
    name = re.sub(r"\.\s*$", "", name)
    return name or None, addr or None


def roman_to_int(s: str) -> int | None:
    """Convert a Roman numeral string to integer, or None if not valid."""
    roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    s = s.upper().strip()
    if not s:
        return None
    # Check all chars are valid Roman numerals
    if not all(c in roman_map for c in s):
        return None
    result = 0
    prev = 0
    for ch in reversed(s):
        val = roman_map[ch]
        if val < prev:
            result -= val
        else:
            result += val
        prev = val
    return result


def parse_volume(raw: str) -> int | None:
    """Parse volume from various formats (Arabic, Roman, mixed)."""
    raw = raw.strip()
    # Try direct integer
    try:
        return int(raw)
    except ValueError:
        pass
    # Try Roman numeral
    result = roman_to_int(raw)
    if result and result > 0:
        return result
    # Try extracting digits from mixed strings like "CI" or "Vol. V"
    m = re.search(r'(\d+)', raw)
    if m:
        return int(m.group(1))
    return None


def parse_number(raw: str) -> int | None:
    """Parse issue/edition number (often a long integer like 2630)."""
    raw = raw.strip()
    try:
        return int(raw)
    except ValueError:
        pass
    m = re.search(r'(\d+)', raw)
    if m:
        return int(m.group(1))
    return None
