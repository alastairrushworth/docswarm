"""LangChain tools for accessing PDF source imagery and citations."""

from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import tool

from docswarm.logger import get_logger

if TYPE_CHECKING:
    from docswarm.config import Config
    from docswarm.storage.database import DatabaseManager

log = get_logger(__name__)


def create_pdf_tools(db: "DatabaseManager") -> list:
    """Create a list of PDF-oriented LangChain tools with the database bound.

    Args:
        db: An initialised :class:`~docswarm.storage.database.DatabaseManager`.

    Returns:
        List of LangChain tool callables.
    """

    @tool
    def get_page_image_info(document_id: str, page_number: int) -> str:
        """Return the filesystem path and metadata for a rendered page image.

        Agents can use this to know exactly where the source image lives on
        disk so they can reference or display it.

        Args:
            document_id: UUID of the document.
            page_number: 1-based page number.

        Returns:
            Formatted string with image path, dimensions, OCR confidence, and
            word count for the requested page.
        """
        pages = db.get_document_pages(document_id)
        if not pages:
            return f"No pages found for document {document_id!r}."

        page = next((p for p in pages if p.get("page_number") == page_number), None)
        if page is None:
            return (
                f"Page {page_number} not found in document {document_id!r}. "
                f"Document has {len(pages)} page(s)."
            )

        image_path = page.get("image_path", "N/A")
        exists = Path(image_path).exists() if image_path != "N/A" else False

        lines = [
            "=== Page image info ===",
            f"Document ID:   {document_id}",
            f"Page number:   {page_number}",
            f"Image path:    {image_path}",
            f"File exists:   {exists}",
            f"Dimensions:    {page.get('width_pts', 'N/A')} × {page.get('height_pts', 'N/A')} pts",
            f"OCR confidence:{page.get('ocr_confidence', 'N/A')}",
            f"Word count:    {page.get('word_count', 'N/A')}",
        ]
        return "\n".join(lines)

    @tool
    def get_source_reference(chunk_id: str) -> str:
        """Return a full formatted citation for a chunk given its ID.

        Looks up the chunk and its parent document to produce a complete
        citation string suitable for inclusion in a wiki article.

        Args:
            chunk_id: UUID string of the chunk.

        Returns:
            A human-readable citation string, e.g.
            ``'Source: "Victorian Furniture" (p.12, chunk 3) — document_id: …'``.
        """
        chunk = db.get_chunk(chunk_id)
        if chunk is None:
            return f"Chunk not found: {chunk_id!r}"

        doc = db.get_document(chunk.get("document_id", ""))
        if doc:
            filename = doc.get("filename", "Unknown")
            title = doc.get("title") or Path(filename).stem
            total_pages = doc.get("total_pages", "N/A")
        else:
            title = "Unknown document"
            total_pages = "N/A"

        reference = chunk.get("reference") or (
            f'"{title}" (p.{chunk.get("page_number")}, ' f'chunk {chunk.get("chunk_index", 0) + 1})'
        )

        lines = [
            "=== Source citation ===",
            f"Reference:   {reference}",
            f"Chunk ID:    {chunk_id}",
            f"Document:    {title}",
            f"Filename:    {doc.get('filename', 'N/A') if doc else 'N/A'}",
            f"Page:        {chunk.get('page_number')} of {total_pages}",
            f"Chunk index: {chunk.get('chunk_index')}",
            f"OCR conf:    {chunk.get('ocr_confidence', 'N/A')}",
            "",
            "Text excerpt:",
            (chunk.get("text") or "")[:400] + ("…" if len(chunk.get("text") or "") > 400 else ""),
        ]
        return "\n".join(lines)

    return [get_page_image_info, get_source_reference]


def create_classification_tools(db: "DatabaseManager", config: "Config") -> list:
    """Create tools for classifying page content using multimodal LLM.

    Args:
        db: An initialised :class:`~docswarm.storage.database.DatabaseManager`.
        config: Application configuration (determines Ollama vs OpenAI).

    Returns:
        List of LangChain tool callables.
    """
    from docswarm.config import Config  # noqa: F811

    @tool
    def classify_page_content(page_id: str) -> str:
        """Classify whether a page is advertising or editorial content.

        Call this BEFORE extracting entities from a page. If the result
        is 'advertisement', skip the page entirely.

        Args:
            page_id: UUID of the page to classify.

        Returns:
            A classification: 'advertisement', 'editorial', or 'mixed',
            with a short explanation.
        """
        page = db.get_page(page_id)
        if page is None:
            return f"Page not found: {page_id!r}"

        image_path = page.get("image_path", "")
        word_count = page.get("word_count", 0) or 0
        raw_text = (page.get("raw_text") or "")[:500]

        # Load and encode the page image
        img_path = Path(image_path)
        if not img_path.exists():
            return (
                f"Page image not found at {image_path}. "
                f"Cannot classify visually. Word count: {word_count}."
            )

        img_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

        prompt = (
            "Look at this scanned magazine/book page and classify it.\n\n"
            "Is this page primarily:\n"
            "- 'advertisement': short promotional content, product ads, classifieds, "
            "price lists, slogans, 'buy now' language, dealer listings\n"
            "- 'editorial': a genuine article, report, review, feature, or informational content "
            "(even if it mentions brands or products at length)\n"
            "- 'mixed': a page with both substantial editorial content and ads\n\n"
            f"The page has {word_count} words. Here is the first 500 characters of OCR text:\n"
            f"{raw_text}\n\n"
            "Reply with EXACTLY one line in this format:\n"
            "CLASSIFICATION: <advertisement|editorial|mixed> — <one sentence reason>"
        )

        try:
            if config.use_ollama:
                answer = _classify_ollama(
                    prompt, img_b64, config.model, config.ollama_base_url
                )
            else:
                answer = _classify_openai(
                    prompt, img_b64, config.openai_api_key, config.openai_model
                )
            log.info("Page %s classified: %s", page_id, answer)
            return answer
        except Exception as e:
            log.warning("Classification failed for page %s: %s", page_id, e)
            return f"Classification failed: {e}. Treat as editorial and proceed."

    return [classify_page_content]


def _classify_ollama(prompt: str, img_b64: str, model: str, base_url: str) -> str:
    """Classify a page image using the Ollama /api/generate endpoint."""
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {"num_predict": 80},
            "think": False,
        }
    ).encode()

    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    return result.get("response", "").strip()


def _classify_openai(prompt: str, img_b64: str, api_key: str, model: str) -> str:
    """Classify a page image using the OpenAI chat completions API."""
    import openai

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=100,
        reasoning_effort="none",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content.strip()
