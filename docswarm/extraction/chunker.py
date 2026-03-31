"""Text chunking for extracted PDF page text."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docswarm.config import Config


class TextChunker:
    """Splits page-level text into overlapping chunks suitable for retrieval.

    The chunker first tries to split on paragraph boundaries (double newlines)
    to preserve natural text boundaries.  Short paragraphs are merged together
    until the target chunk size is reached; long paragraphs are split on
    whitespace boundaries.  A sliding-window overlap is applied so consecutive
    chunks share some context.
    """

    def __init__(self, config: "Config") -> None:
        """Initialise the chunker with the given configuration.

        Args:
            config: A :class:`~docswarm.config.Config` instance that provides
                ``chunk_size`` and ``chunk_overlap`` settings.
        """
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_page(
        self,
        text: str,
        page_id: str,
        document_id: str,
        page_number: int,
        document_title: str | None,
        filename: str,
        ocr_confidence: float,
    ) -> list[dict]:
        """Split a page's text into chunks and return a list of chunk dicts.

        Args:
            text: Raw OCR text for the page.
            page_id: UUID of the parent page record.
            document_id: UUID of the parent document record.
            page_number: 1-based page number.
            document_title: Human-readable document title (used in references).
                If ``None``, *filename* without extension is used.
            filename: Bare filename of the source PDF (fallback for title).
            ocr_confidence: Mean OCR confidence for the page (0–100).

        Returns:
            A list of chunk dicts ready for insertion via
            :meth:`~docswarm.storage.database.DatabaseManager.insert_chunk`.
        """
        if not text or not text.strip():
            return []

        title = document_title or Path(filename).stem

        # Split into raw segments on paragraph boundaries
        segments = self._split_paragraphs(text)

        # Merge/split segments to approach target chunk_size
        merged = self._merge_segments(segments)

        # Apply sliding-window overlap and build final chunk strings
        chunk_strings = self._apply_overlap(merged)

        chunks: list[dict] = []
        char_cursor = 0

        for idx, chunk_text in enumerate(chunk_strings):
            chunk_text_stripped = chunk_text.strip()
            if not chunk_text_stripped:
                continue

            # Find approximate char positions in the original text
            char_start = text.find(chunk_text_stripped[:50], char_cursor)
            if char_start == -1:
                char_start = char_cursor
            char_end = char_start + len(chunk_text_stripped)
            char_cursor = max(char_cursor, char_start)

            reference = f'"{title}" (p.{page_number}, chunk {idx + 1})'

            chunk: dict = {
                "id": str(uuid.uuid4()),
                "document_id": document_id,
                "page_id": page_id,
                "page_number": page_number,
                "chunk_index": idx,
                "text": chunk_text_stripped,
                "char_start": char_start,
                "char_end": char_end,
                "chunk_type": "text",
                "ocr_confidence": ocr_confidence,
                "reference": reference,
                "word_count": len(chunk_text_stripped.split()),
            }
            chunks.append(chunk)

        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split *text* on double-newline paragraph boundaries.

        Single newlines inside paragraphs are preserved. Empty segments are
        discarded.

        Args:
            text: Full page text.

        Returns:
            List of non-empty paragraph strings.
        """
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_on_whitespace(self, text: str, target: int) -> list[str]:
        """Hard-split a long string into pieces of at most *target* characters.

        Splits are made at whitespace boundaries to avoid breaking words.

        Args:
            text: String to split.
            target: Maximum character length of each piece.

        Returns:
            List of string pieces.
        """
        words = text.split()
        pieces: list[str] = []
        current: list[str] = []
        current_len = 0

        for word in words:
            word_len = len(word) + (1 if current else 0)
            if current_len + word_len > target and current:
                pieces.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += word_len

        if current:
            pieces.append(" ".join(current))

        return pieces

    def _merge_segments(self, segments: list[str]) -> list[str]:
        """Merge short paragraphs together and split overlong ones.

        Args:
            segments: Paragraph-level text segments.

        Returns:
            List of text segments that are closer to ``chunk_size`` in length.
        """
        target = self.config.chunk_size
        merged: list[str] = []
        buffer = ""

        for seg in segments:
            # Overlong segment: split it immediately
            if len(seg) > target:
                if buffer:
                    merged.append(buffer)
                    buffer = ""
                merged.extend(self._split_on_whitespace(seg, target))
                continue

            # Would the buffer exceed target if we add this segment?
            candidate = (buffer + "\n\n" + seg).strip() if buffer else seg
            if len(candidate) <= target:
                buffer = candidate
            else:
                if buffer:
                    merged.append(buffer)
                buffer = seg

        if buffer:
            merged.append(buffer)

        return merged

    def _apply_overlap(self, segments: list[str]) -> list[str]:
        """Add sliding-window overlap between consecutive segments.

        The tail of each segment is prepended to the next segment up to
        ``chunk_overlap`` characters (aligned to a word boundary).

        Args:
            segments: List of merged text segments.

        Returns:
            List of chunk strings with overlap applied.
        """
        if not segments:
            return []
        if len(segments) == 1:
            return segments

        overlap = self.config.chunk_overlap
        result: list[str] = [segments[0]]

        for i in range(1, len(segments)):
            prev = segments[i - 1]
            curr = segments[i]

            # Take the last `overlap` chars of the previous segment
            if len(prev) > overlap:
                tail = prev[-overlap:]
                # Align to word boundary
                space_idx = tail.find(" ")
                if space_idx != -1:
                    tail = tail[space_idx + 1 :]
            else:
                tail = prev

            combined = (tail + " " + curr).strip()
            result.append(combined)

        return result
