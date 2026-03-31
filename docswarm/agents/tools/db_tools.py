"""LangChain tools that expose DatabaseManager functionality to agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from docswarm.storage.database import DatabaseManager


def _format_chunk(chunk: dict) -> str:
    """Render a single chunk dict as a human-readable string.

    The reference line is placed first so agents can immediately see the
    provenance of the text.

    Args:
        chunk: Chunk dict from the database.

    Returns:
        Formatted string representation.
    """
    lines = [
        f"REFERENCE: {chunk.get('reference', 'N/A')}",
        f"Chunk ID:  {chunk.get('id', 'N/A')}",
        f"Document:  {chunk.get('document_id', 'N/A')}",
        f"Page:      {chunk.get('page_number', 'N/A')}",
        f"Words:     {chunk.get('word_count', 'N/A')}",
        f"OCR conf:  {chunk.get('ocr_confidence', 'N/A')}",
        "",
        chunk.get("text", ""),
    ]
    return "\n".join(lines)


def create_db_tools(db: "DatabaseManager") -> list:
    """Create a list of LangChain tools with the database instance bound.

    Args:
        db: An initialised :class:`~docswarm.storage.database.DatabaseManager`.

    Returns:
        List of LangChain tool callables ready for use in a LangGraph agent.
    """

    @tool
    def search_chunks(query: str, limit: int = 10) -> str:
        """Search the extracted document text for a given query.

        Use this tool to find relevant passages in the source material.
        Returns formatted results with reference information prominently shown.

        Args:
            query: Keywords or phrase to search for.
            limit: Maximum number of results (default 10).

        Returns:
            Formatted search results including references, chunk IDs, and text.
        """
        results = db.search_chunks(query, limit=limit)
        if not results:
            return f"No chunks found matching query: {query!r}"
        parts = [f"=== Search results for: {query!r} ({len(results)} found) ===\n"]
        for i, chunk in enumerate(results, 1):
            parts.append(f"--- Result {i} ---")
            parts.append(_format_chunk(chunk))
            parts.append("")
        return "\n".join(parts)

    @tool
    def get_chunk(chunk_id: str) -> str:
        """Retrieve the full text and metadata of a specific chunk by its ID.

        Args:
            chunk_id: UUID string of the chunk to retrieve.

        Returns:
            Formatted chunk data including reference, text, and metadata.
        """
        chunk = db.get_chunk(chunk_id)
        if chunk is None:
            return f"Chunk not found: {chunk_id!r}"
        return _format_chunk(chunk)

    @tool
    def get_page_text(document_id: str, page_number: int) -> str:
        """Retrieve all text from a specific page of a document.

        Concatenates all chunks from the given page in order.

        Args:
            document_id: UUID of the document.
            page_number: 1-based page number.

        Returns:
            The concatenated text for the page with a header showing the source.
        """
        chunks = db.get_chunks_by_document(document_id, page_number=page_number)
        if not chunks:
            return f"No text found for document {document_id!r}, page {page_number}."
        doc = db.get_document(document_id)
        title = doc.get("title", document_id) if doc else document_id
        header = f'=== Page {page_number} of "{title}" ===\n'
        text_parts = [chunk.get("text", "") for chunk in chunks]
        return header + "\n\n".join(text_parts)

    @tool
    def list_documents() -> str:
        """List all source documents that have been ingested into the database.

        Returns:
            A formatted list of documents with their IDs, filenames, and page counts.
        """
        docs = db.list_documents()
        if not docs:
            return "No documents found in the database."
        lines = [f"=== {len(docs)} document(s) in database ===\n"]
        for doc in docs:
            lines.append(
                f"ID:       {doc.get('id')}\n"
                f"Filename: {doc.get('filename')}\n"
                f"Title:    {doc.get('title', 'N/A')}\n"
                f"Pages:    {doc.get('total_pages', 'N/A')}\n"
                f"Ingested: {doc.get('ingested_at', 'N/A')}\n"
            )
        return "\n".join(lines)

    @tool
    def get_document_chunks(document_id: str, page_number: int = None) -> str:
        """Get all chunks for a document, optionally filtered to a specific page.

        Args:
            document_id: UUID of the document.
            page_number: If provided, only return chunks from this page.

        Returns:
            Formatted list of chunks with references and text.
        """
        chunks = db.get_chunks_by_document(document_id, page_number=page_number)
        if not chunks:
            filter_info = f", page {page_number}" if page_number is not None else ""
            return f"No chunks found for document {document_id!r}{filter_info}."
        parts = [
            f"=== {len(chunks)} chunk(s) for document {document_id!r}"
            + (f", page {page_number}" if page_number is not None else "")
            + " ===\n"
        ]
        for chunk in chunks:
            parts.append(_format_chunk(chunk))
            parts.append("---")
        return "\n".join(parts)

    @tool
    def get_page_study_status() -> str:
        """Show all pages with their study counts and when they were last studied.

        Use this to understand which pages still need to be worked on and
        which have already been covered by the swarm.

        Returns:
            Formatted table of pages with document, page number, study count,
            and last-studied timestamp.
        """
        rows = db.get_page_study_counts()
        if not rows:
            return "No pages found in the database."
        lines = [f"=== Page study status ({len(rows)} pages) ===\n"]
        for r in rows:
            doc = db.get_document(r["document_id"])
            doc_title = doc.get("title", r["document_id"]) if doc else r["document_id"]
            status = (
                f"studied {r['study_count']}x (last: {r['last_studied']})"
                if r["study_count"]
                else "UNSTUDIED"
            )
            lines.append(f"{doc_title}  p.{r['page_number']}  [{status}]  page_id={r['id']}")
        return "\n".join(lines)

    @tool
    def mark_page_studied(page_id: str, wiki_article_path: str = "") -> str:
        """Record that the swarm has studied a page and produced a wiki article.

        Call this after successfully publishing a wiki article for a page.

        Args:
            page_id: UUID of the page that was studied (from pages.id).
            wiki_article_path: Path of the published wiki article (e.g. 'history/topic').

        Returns:
            Confirmation string.
        """
        page = db.get_page(page_id)
        if page is None:
            return f"Page not found: {page_id!r}"
        db.log_page_study(
            page_id=page_id,
            document_id=page["document_id"],
            wiki_article_path=wiki_article_path,
        )
        return f"Page {page_id} marked as studied (article: {wiki_article_path or 'none'})."

    return [
        search_chunks,
        get_chunk,
        get_page_text,
        list_documents,
        get_document_chunks,
        get_page_study_status,
        mark_page_studied,
    ]
