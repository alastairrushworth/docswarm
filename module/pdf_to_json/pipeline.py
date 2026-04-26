"""Top-level pdf_to_json pipeline.

Internals are the agent's design space (DESIGN.md §7). This is a working baseline:
render each PDF page to PNG, query a vision model with a structured-extraction prompt,
parse JSON, then assemble. Caches per-page IR by content hash.
"""
from __future__ import annotations

import json
import logging
import re
import signal
import tempfile
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from . import cache, ollama_client
from .assemble import assemble, empty_document
from .config import get
from .schema import Document

logger = logging.getLogger("pdf_to_json")

PROMPT_VERSION = "v1"

_VISION_PROMPT = """\
You are extracting structured data from a single page of a late-19th-century cycling
magazine. Respond with a single JSON object. No prose outside the JSON.

Schema:
{
  "is_first_page": bool,
  "magazine": {                // null unless is_first_page
    "editor": str,
    "issue": {"date": "YYYY-MM-DD", "volume": int, "number": int},
    "publisher": {"name": str, "address": str},
    "cost": {"issue": str|null, "annual": str|null, "semiannual": str|null}
  } | null,
  "articles": [
    {
      "title": str,
      "text": [str],            // paragraphs (prose) or lines (verse)
      "kind": "prose" | "verse",
      "starts_on_this_page": bool,
      "continues": bool          // true if it continues to a later page
    }
  ]
}

Preserve typography verbatim — do not normalize stray quotes or OCR-style typos.
Currency values stay as strings (e.g. "$2.00").
"""


class _Timeout(Exception):
    pass


@contextmanager
def _deadline(seconds: int):
    """SIGALRM-based deadline; falls back to a no-op if SIGALRM is unavailable."""
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum, frame):  # noqa: ARG001
        raise _Timeout()

    prev = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def _render_page(doc: fitz.Document, page_index: int, out: Path, dpi: int = 200) -> Path:
    page = doc.load_page(page_index)
    pix = page.get_pixmap(dpi=dpi)
    pix.save(str(out))
    return out


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


def _extract_page(
    pdf_hash: str,
    page_index: int,
    image_path: Path,
    model: str,
    budget_seconds: int,
) -> dict[str, Any]:
    cached = cache.load(pdf_hash, page_index, model, PROMPT_VERSION)
    if cached is not None:
        return cached

    try:
        with _deadline(budget_seconds):
            raw = ollama_client.generate(
                model=model,
                prompt=_VISION_PROMPT,
                images=[image_path],
                timeout=float(budget_seconds),
            )
    except _Timeout:
        warnings.warn(f"page {page_index + 1}: vision call exceeded {budget_seconds}s budget")
        return {}
    except Exception as e:
        warnings.warn(f"page {page_index + 1}: vision call failed: {e}")
        return {}

    parsed = _parse_json(raw)
    if parsed:
        cache.store(pdf_hash, page_index, model, PROMPT_VERSION, parsed)
    return parsed


def _merge_articles(per_page: list[tuple[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Stitch continuations across pages, preserving printed sequence."""
    ordered: list[dict[str, Any]] = []
    open_article: dict[str, Any] | None = None

    for printed_page, page_ir in per_page:
        articles = page_ir.get("articles") or []
        for idx, a in enumerate(articles):
            starts = a.get("starts_on_this_page", True)
            if not starts and open_article is not None and idx == 0:
                open_article["text"].extend(a.get("text") or [])
                if printed_page not in open_article["pages"]:
                    open_article["pages"].append(printed_page)
                if not a.get("continues"):
                    open_article = None
                continue
            new = {
                "title": a.get("title", ""),
                "text": list(a.get("text") or []),
                "kind": a.get("kind", "prose"),
                "pages": [printed_page],
            }
            ordered.append(new)
            open_article = new if a.get("continues") else None
    return ordered


def pdf_to_json(pdf_path: str) -> dict:
    """Translate a scanned magazine PDF to schema-conformant JSON.

    Always returns *some* JSON — best effort on timeout or extraction failure.
    """
    page_budget = int(get("iteration.page_budget_seconds", 30))
    model = get("models.vision_small", "llama3.2-vision:11b")

    p = Path(pdf_path)
    if not p.is_file():
        warnings.warn(f"pdf not found: {pdf_path}")
        return empty_document().model_dump(mode="json")

    pdf_hash = cache.pdf_content_hash(p)

    try:
        doc = fitz.open(p)
    except Exception as e:
        warnings.warn(f"failed to open pdf: {e}")
        return empty_document().model_dump(mode="json")

    per_page: list[tuple[int, dict[str, Any]]] = []
    meta_ir: dict[str, Any] = {}

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(doc.page_count):
            img = Path(tmp) / f"page_{i:03d}.png"
            try:
                _render_page(doc, i, img)
            except Exception as e:
                warnings.warn(f"page {i + 1}: render failed: {e}")
                continue

            t0 = time.monotonic()
            page_ir = _extract_page(pdf_hash, i, img, model, page_budget)
            elapsed = time.monotonic() - t0
            logger.info("page %d/%d: extracted in %.1fs", i + 1, doc.page_count, elapsed)

            if page_ir.get("is_first_page") and not meta_ir:
                m = page_ir.get("magazine")
                if isinstance(m, dict):
                    meta_ir = m

            per_page.append((i + 1, page_ir))

    doc.close()

    articles = _merge_articles(per_page)
    try:
        assembled: Document = assemble(meta_ir, articles)
    except Exception as e:
        warnings.warn(f"assembly failed: {e}")
        assembled = empty_document()

    return assembled.model_dump(mode="json")
