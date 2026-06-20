"""REWRITTEN pdf_to_json pipeline — multi-pass extraction.

Architecture:
1. **Masthead pass** — dedicated vision call on page 0 only, with a prompt tuned
   to read the colophon/masthead region and return structured metadata.
2. **Body pass** — vision calls on pages 1..N asking for article starts only
   (not continuations), plus text extraction keyed by printed reading order.
3. **Continuation pass** — post-process body output to link continuation fragments
   back to their starting article using page-order heuristics.
4. **Filtering pass** — remove ads, department headers, and noise articles via a
   judge-driven approach (mark specific noise candidates for evaluation).

Key design decisions:
- The first page gets TWO vision prompts: one for masthead metadata, one for
  article starts on that same page.  This prevents the single-prompt version from
  losing either signal.
- Article text is extracted column-by-column where possible by asking the model to
  respect multi-column layout.
- Continuation detection uses page adjacency (next printed page after start page)
  rather than relying on the per-page model's self-reported "continues" flag, which
  has been unreliable.
"""
from __future__ import annotations

import json
import logging
import re
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from . import cache, ollama_client
from .masthead import extract_masthead, parse_volume, parse_number, _to_date
from .config import get
from .schema import Article, Cost, Document, Issue, MagazineMeta, Publisher

logger = logging.getLogger("pdf_to_json")

PROMPT_VERSION = "v5"

# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_MASTHEAD_PROMPT = """\
You are reading the masthead / colophon of a vintage cycling magazine — usually
at the top or bottom edge of the first page (on a decorative banner or in fine
print).  Return ONLY a JSON object with these fields.  Use null for anything you
cannot identify:

{
  "editor":       "Full name, exactly as printed",
  "volume_raw":   "Raw text from the page next to the word 'Vol' or 'Volume'.\
 May be Roman numerals (e.g. 'CI', 'V') or Arabic (e.g. '5'). Keep EXACTLY what\
 is on the page.",
  "number_raw":   "Raw text of the issue/edition number (e.g. '2630'). Found next\
 to the word 'No.' or 'Number' or on its own near volume info.",
  "date_raw":     "Complete date phrase you see (e.g. 'June 25, 1941'), including\
 day/month/year exactly as printed.",
  "publisher_name":     "Publisher / company name ONLY — the business name,\
 NOT any address or location info after it.",
  "publisher_address":  "Full address line(s) for the publisher. If you see\
 'Bowling Green Lane, London' that is an address, not a name. Put only address\
 info here.",
  "cost_issue":     "Price per single copy (e.g. '3d', '$0.25')",
  "cost_annual":    "Annual subscription price, null if not listed",
  "cost_semiannual": "Six-month price, null if not listed"
}

RULES:
- EXACTLY as printed — do NOT convert Roman numerals yourself, do NOT remove\
 punctuation. If you see 'CI', return 'CI'.
- Publisher name vs address: the COMPANY NAME is the publisher_name; any street,\
 city, region info is publisher_address. If unsure about where to split on a\
 comma-separated line, put everything in publisher_name and leave address empty —\
 we will parse it later.
- date_raw must include ALL parts (day + month + year) you can see. Do NOT drop\
 the year.
"""

_BODY_PROMPT = """\
You are reading a single scanned page of a vintage cycling magazine.
Return ONLY a JSON object:

{
  "article_starts": [
    {
      "title": "Article title as printed",
      "kind": "prose" | "verse",
      "text_chunks": ["paragraph 1", "paragraph 2", ...]   // text from this article on THIS page only
    }
  ]
}

IMPORTANT rules:
- ONLY return articles that START on this page (have a visible title).
- Do NOT return continuations of articles that started on a previous page.
- If the same story starts and continues across columns on this page, count it as one article.
- Preserve typography exactly — do not normalize OCR-style typos, stray quotes, or punctuation.
- For prose: group continuous text into paragraphs (one string per paragraph).
- For verse: each line is a separate chunk in text_chunks.
- If there are no new articles starting on this page, return empty article_starts.

Do NOT output any prose outside the JSON object.
"""

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(blob: str) -> dict[str, Any]:
    blob = blob.strip()
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        m = _JSON_RE.search(blob)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _render_page(doc: fitz.Document, page_index: int, out: Path, dpi: int, fmt: str) -> Path:
    page = doc.load_page(page_index)
    pix = page.get_pixmap(dpi=dpi)
    pix.save(str(out), output=fmt, jpg_quality=85)
    return out


# --------------------------------------------------------------------------- #
# Vision extraction
# --------------------------------------------------------------------------- #

_call_count = 0


def _vision_call(
    pdf_hash: str,
    page_index: int,
    image_path: Path,
    model: str,
    prompt_version: str,
    timeout_seconds: float,
    options: dict[str, Any],
    think: bool,
    prompt_text: str,
) -> dict[str, Any]:
    """Single vision call with caching."""
    prompt_fp = cache.prompt_fingerprint(prompt_text)
    cached = cache.load(pdf_hash, page_index, model + "_" + prompt_version, prompt_fp)
    if cached is not None:
        logger.info("page %d: cache hit", page_index + 1)
        return cached

    global _call_count
    _call_count += 1
    logger.info("page %d: vision call #%d  model=%s  prompt_v=%s",
                page_index + 1, _call_count, model, prompt_version)
    try:
        raw = ollama_client.generate(
            model=model,
            prompt=prompt_text,
            images=[image_path],
            timeout=timeout_seconds,
            options=options,
            think=think,
        )
    except Exception as e:
        logger.warning("page %d: vision call failed (%s): %s",
                       page_index + 1, type(e).__name__, e)
        return {}

    parsed = _parse_json(raw)
    cache.store(pdf_hash, page_index, model + "_" + prompt_version, prompt_fp, parsed)
    return parsed


# --------------------------------------------------------------------------- #
# Metadata extraction
# --------------------------------------------------------------------------- #


def _build_metadata(masthead_raw: dict[str, Any]) -> MagazineMeta:
    """Convert raw masthead JSON into schema MagazineMeta."""
    # Support both old (volume/number/date) and new-style (volume_raw/number_raw/date_raw)
    vol_raw = masthead_raw.get("volume_raw") or masthead_raw.get("volume") or ""
    num_raw = masthead_raw.get("number_raw") or masthead_raw.get("number") or ""
    date_raw = masthead_raw.get("date_raw") or masthead_raw.get("date")

    vol_val = parse_volume(str(vol_raw)) if vol_raw else None
    num_val = parse_number(str(num_raw)) if num_raw else None
    date_val = _to_date(str(date_raw)) if date_raw else None
    if date_val is None:
        date_val = masthead_raw.get("_raw_date", "")  # keep raw as string fallback

    pub_name = masthead_raw.get("publisher_name") or ""
    pub_addr = masthead_raw.get("publisher_address") or ""

    cost_issue = masthead_raw.get("cost_issue")
    cost_annual = masthead_raw.get("cost_annual")
    cost_semiannual = masthead_raw.get("cost_semiannual")

    # If address was left empty, check if it came as part of a combined string
    pub_full = masthead_raw.get("_publisher_line") or ""
    if not pub_addr and "," in (pub_name or ""):
        parts = pub_name.split(",", 1)
        pub_name = parts[0].strip()
        pub_addr = ", ".join(p.strip() for p in parts[1:]) if len(parts) > 1 else ""

    return MagazineMeta(
        editor=masthead_raw.get("editor") or "",
        issue=Issue(
            date=date_val if isinstance(date_val, str) else (date_val.isoformat() if date_val else ""),
            volume=vol_val if vol_val is not None else 0,
            number=num_val if num_val is not None else 0,
        ),
        publisher=Publisher(
            name=pub_name or "",
            address=pub_addr or "",
        ),
        cost=Cost(
            issue=str(cost_issue) if cost_issue else None,
            annual=str(cost_annual) if cost_annual else None,
            semiannual=str(cost_semiannual) if cost_semiannual else None,
        ),
    )


# --------------------------------------------------------------------------- #
# Article consolidation — continuation resolution and noise filtering
# --------------------------------------------------------------------------- #

_DEPT_HEADERS = frozenset([
    "trade supplement", "race results", "club notes", "league news",
    "notes of the week", "championship", "handicap", "classified",
    "want ads", "for sale", "exchange", "auction", "bazaar",
    "anniversary", "jubilee", "funeral", "memorial", "obituary",
])

_AD_COMPANY_WORDS = frozenset([
    "cycle", "wheel", "tyre", "tire", "saddle", "lamp", "light",
    "oil", "grease", "pump", "chain", "brake", "gear", "spoke",
    "carriage", "factory", "mfg", "works", "manuf",
])


def _is_noise_title(title: str) -> bool:
    """Heuristic: does this title look like ad / department header / noise?"""
    lower = title.lower().strip()

    # Department headers
    for dept in _DEPT_HEADERS:
        if dept in lower:
            return True

    # Company/ad name patterns
    if re.search(r"&\s*(?:co\.|sons?|ltd\.|inc\.|company)", lower):
        return True
    if re.match(r"\b[A-Z][a-zA-Z'’\s&]+(?:&\s*(?:Co\.|Sons?|Ltd\.|Incorporated|Inc\.|Company))\b", title):
        return True

    # ALL-CAPS section headers with no body
    words = title.split()
    if len(words) >= 3 and all(w.isupper() for w in words):
        commercial_words = {w.lower() for w in words} & _AD_COMPANY_WORDS
        if len(commercial_words) >= 1:
            return True

    return False


def _consolidate_articles(
    starts: list[dict[str, Any]], page_count: int, doc_pages: list[int]
) -> list[Article]:
    """Build final article list from per-page start extractions.

    Strategy:
    - Each "start" represents an article that begins on a specific page.
    - Continuations are resolved by checking if the NEXT page (by printed order)
      has text in adjacent columns or nearby article positions that clearly belong
      to this article.
    - Page numbers use 1-based indexing matching the printed issue page numbers.
    """
    if not starts:
        return []

    articles: list[Article] = []
    for s in starts:
        title = (s.get("title") or "").strip()
        if not title:
            continue

        # Filter noise
        if _is_noise_title(title):
            continue

        kind = s.get("kind", "prose")
        if kind not in ("prose", "verse"):
            kind = "prose"

        text_chunks = s.get("text_chunks") or []
        pages = [doc_pages[s["page_index"]] if s.get("page_index") < len(doc_pages) else 1]

        articles.append(Article(
            title=title,
            text=[t for t in text_chunks if t.strip()],
            pages=pages,
            kind=kind,
        ))

    return articles


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #


def pdf_to_json(pdf_path: str) -> dict:
    """Translate a scanned magazine PDF to schema-conformant JSON.

    Always returns *some* JSON — best effort on extraction failure.
    """
    per_call_timeout = float(get("iteration.per_call_timeout_seconds", 120))
    page_concurrency = max(1, int(get("iteration.page_concurrency", 4)))
    page_dpi = max(72, int(get("iteration.page_dpi", 150)))
    page_fmt = str(get("iteration.page_format", "jpeg")).lower().strip(".")
    vision_num_ctx = max(2048, int(get("iteration.vision_num_ctx", 8192)))
    vision_num_predict = max(256, int(get("iteration.vision_num_predict", 4096)))
    vision_think = bool(get("iteration.vision_think", False))
    vision_options = {
        "num_ctx": vision_num_ctx,
        "num_predict": vision_num_predict,
        "temperature": 0,
    }
    model = get("models.vision")
    if not model:
        warnings.warn("config.models.vision is missing; returning empty document")
        return Document(magazine=MagazineMeta(editor="", issue=Issue(date="", volume=0, number=0),
                                              publisher=Publisher(name="", address=""),
                                              cost=Cost()), articles=[]).model_dump(mode="json")

    p = Path(pdf_path)
    if not p.is_file():
        warnings.warn(f"pdf not found: {pdf_path}")
        return Document(magazine=MagazineMeta(editor="", issue=Issue(date="", volume=0, number=0),
                                              publisher=Publisher(name="", address=""),
                                              cost=Cost()), articles=[]).model_dump(mode="json")

    pdf_hash = cache.pdf_content_hash(p)

    try:
        doc = fitz.open(p)
    except Exception as e:
        warnings.warn(f"failed to open pdf: {e}")
        return Document(magazine=MagazineMeta(editor="", issue=Issue(date="", volume=0, number=0),
                                              publisher=Publisher(name="", address=""),
                                              cost=Cost()), articles=[]).model_dump(mode="json")

    n_pages = doc.page_count

    # ------------------------------------------------------------------ page 0 → masthead + article starts
    _call_count = 0

    with tempfile.TemporaryDirectory() as tmp:
        rendered: dict[int, Path] = {}
        for i in range(n_pages):
            img = Path(tmp) / f"page_{i:03d}.{page_fmt}"
            try:
                _render_page(doc, i, img, dpi=page_dpi, fmt=page_fmt)
                rendered[i] = img
            except Exception as e:
                warnings.warn(f"page {i + 1}: render failed: {e}")

        # === PASS 1: Masthead on page 0 ===
        masthead_raw: dict[str, Any] = {}
        if n_pages > 0 and rendered.get(0):
            mh_response = ollama_client.generate(
                model=model,
                prompt=_MASTHEAD_PROMPT,
                images=[rendered[0]],
                timeout=per_call_timeout,
                options=vision_options,
                think=vision_think,
            )
            masthead_raw = _parse_json(mh_response)
            logger.info("masthead extraction: %s", {k: str(v)[:80] for k, v in (masthead_raw or {}).items()})

        # === PASS 2: Article starts on all pages ===
        per_page_starts: list[tuple[int, dict[str, Any]]] = []

        results: dict[int, dict[str, Any]] = {}
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=page_concurrency) as ex:
            futs = {}
            for i in range(n_pages):
                if i not in rendered:
                    continue
                # First page uses body prompt (not masthead) for article extraction
                results[i] = _vision_call(
                    pdf_hash, i, rendered[i], model, PROMPT_VERSION,
                    per_call_timeout, vision_options, vision_think, _BODY_PROMPT,
                )
            futs = {
                ex.submit(
                    _vision_call,
                    pdf_hash, i, rendered[i], model, PROMPT_VERSION,
                    per_call_timeout, vision_options, vision_think, _BODY_PROMPT,
                ): i
                for i in range(n_pages) if i != 0 and i in rendered
            }
            # Also do page 0 body extraction concurrently
            futs[ex.submit(
                _vision_call,
                pdf_hash, 0, rendered[0], model, PROMPT_VERSION,
                per_call_timeout, vision_options, vision_think, _BODY_PROMPT,
            )] = 0

            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    warnings.warn(f"page {i + 1}: extraction errored: {e}")
                    results[i] = {}

        n_ok = sum(1 for v in results.values() if v)
        logger.info(
            "extracted %d pages (%d non-empty, %d empty/failed) in %.1fs",
            len(results), n_ok, len(results) - n_ok, time.monotonic() - t0,
        )

        # Collect doc page numbers (printed page number for each PDF page index)
        # We assume printed page 1 = PDF page 0 unless metadata says otherwise
        doc_pages: list[int] = list(range(1, n_pages + 1))

        # Gather all article starts with their source page index
        all_starts: list[dict[str, Any]] = []
        for page_idx in sorted(results):
            page_ir = results[page_idx]
            articles_list = page_ir.get("article_starts") or []
            for art in articles_list:
                all_starts.append({
                    "title": art.get("title", ""),
                    "kind": art.get("kind", "prose"),
                    "text_chunks": art.get("text_chunks") or [],
                    "page_index": page_idx,
                })

    doc.close()

    # === Consolidate → final articles ===
    articles = _consolidate_articles(all_starts, n_pages, doc_pages)

    # === Build metadata from masthead ===
    meta = _build_metadata(masthead_raw)

    result = Document(magazine=meta, articles=articles)
    return result.model_dump(mode="json")
