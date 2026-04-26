"""STARTER pdf_to_json pipeline — replace freely.

This file exists so round 1 can produce *some* score. It is NOT the recommended
architecture. The translator is expected to be LLM-heavy and agentic: multi-pass
extraction, specialist sub-modules, model routing, self-checks. Single vision
call per page is unlikely to be sufficient.

The fixed contract is:
- the public function name and signature: `pdf_to_json(pdf_path: str) -> dict`
- the schema source-of-truth in `schema.py`
- the per-page time budget from `config.iteration.page_budget_seconds`
- content-hash cache keying

Everything else here — prompts, model routing, page rendering, stitching — is
yours to delete and replace. See AGENT.md "What is fixed vs what you design".
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
        raw = ollama_client.generate(
            model=model,
            prompt=_VISION_PROMPT,
            images=[image_path],
            timeout=float(budget_seconds),
        )
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
    page_concurrency = max(1, int(get("iteration.page_concurrency", 4)))
    model = get("models.translator", "qwen3-coder:32b")

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
        rendered: dict[int, Path] = {}
        for i in range(doc.page_count):
            img = Path(tmp) / f"page_{i:03d}.png"
            try:
                _render_page(doc, i, img)
                rendered[i] = img
            except Exception as e:
                warnings.warn(f"page {i + 1}: render failed: {e}")

        results: dict[int, dict[str, Any]] = {}
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=page_concurrency) as ex:
            futs = {
                ex.submit(_extract_page, pdf_hash, i, img, model, page_budget): i
                for i, img in rendered.items()
            }
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    warnings.warn(f"page {i + 1}: extraction errored: {e}")
                    results[i] = {}
        logger.info(
            "extracted %d pages in %.1fs (concurrency=%d)",
            len(results), time.monotonic() - t0, page_concurrency,
        )

        for i in sorted(results):
            page_ir = results[i]
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
