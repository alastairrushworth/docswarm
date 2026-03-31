"""Tests for docswarm.extraction.chunker.TextChunker."""

from __future__ import annotations

import pytest

from docswarm.config import Config
from docswarm.extraction.chunker import TextChunker


@pytest.fixture
def chunker():
    """TextChunker with a small chunk_size and overlap for easy testing."""
    cfg = Config(chunk_size=200, chunk_overlap=40)
    return TextChunker(cfg)


@pytest.fixture
def chunker_default():
    """TextChunker with default Config (chunk_size=800, overlap=100)."""
    return TextChunker(Config())


SAMPLE_PAGE_ID = "page-uuid-1234"
SAMPLE_DOC_ID = "doc-uuid-5678"


# ---------------------------------------------------------------------------
# chunk_page – basic behaviour
# ---------------------------------------------------------------------------

class TestChunkPage:
    def test_empty_string_returns_empty_list(self, chunker):
        result = chunker.chunk_page(
            text="",
            page_id=SAMPLE_PAGE_ID,
            document_id=SAMPLE_DOC_ID,
            page_number=1,
            document_title="Test Doc",
            filename="test_doc.pdf",
            ocr_confidence=90.0,
        )
        assert result == []

    def test_whitespace_only_returns_empty_list(self, chunker):
        result = chunker.chunk_page(
            text="   \n\n\t  ",
            page_id=SAMPLE_PAGE_ID,
            document_id=SAMPLE_DOC_ID,
            page_number=1,
            document_title="Test Doc",
            filename="test_doc.pdf",
            ocr_confidence=90.0,
        )
        assert result == []

    def test_single_paragraph_produces_at_least_one_chunk(self, chunker):
        result = chunker.chunk_page(
            text="A short paragraph of text that fits in one chunk.",
            page_id=SAMPLE_PAGE_ID,
            document_id=SAMPLE_DOC_ID,
            page_number=1,
            document_title="Test Doc",
            filename="test_doc.pdf",
            ocr_confidence=90.0,
        )
        assert len(result) >= 1

    def test_chunks_respect_chunk_size(self, chunker):
        """Each chunk's text should not exceed chunk_size by much."""
        long_text = " ".join(["word"] * 500)
        result = chunker.chunk_page(
            text=long_text,
            page_id=SAMPLE_PAGE_ID,
            document_id=SAMPLE_DOC_ID,
            page_number=1,
            document_title="Test Doc",
            filename="test_doc.pdf",
            ocr_confidence=90.0,
        )
        assert len(result) > 1
        for chunk in result:
            # Allow a little slack for overlap text
            assert len(chunk["text"]) <= chunker.config.chunk_size + chunker.config.chunk_overlap + 50

    def test_chunk_fields_present(self, chunker):
        result = chunker.chunk_page(
            text="Some text here.",
            page_id=SAMPLE_PAGE_ID,
            document_id=SAMPLE_DOC_ID,
            page_number=5,
            document_title="My Document",
            filename="my_document.pdf",
            ocr_confidence=87.3,
        )
        assert len(result) == 1
        chunk = result[0]
        required_fields = {
            "id", "document_id", "page_id", "page_number", "chunk_index",
            "text", "char_start", "char_end", "chunk_type", "ocr_confidence",
            "reference", "word_count",
        }
        assert required_fields.issubset(chunk.keys())

    def test_chunk_sets_correct_document_id(self, chunker):
        result = chunker.chunk_page(
            text="Text here.",
            page_id=SAMPLE_PAGE_ID,
            document_id=SAMPLE_DOC_ID,
            page_number=1,
            document_title="Test",
            filename="test.pdf",
            ocr_confidence=90.0,
        )
        assert all(c["document_id"] == SAMPLE_DOC_ID for c in result)

    def test_chunk_sets_correct_page_id(self, chunker):
        result = chunker.chunk_page(
            text="Text here.",
            page_id=SAMPLE_PAGE_ID,
            document_id=SAMPLE_DOC_ID,
            page_number=1,
            document_title="Test",
            filename="test.pdf",
            ocr_confidence=90.0,
        )
        assert all(c["page_id"] == SAMPLE_PAGE_ID for c in result)

    def test_chunk_sets_correct_page_number(self, chunker):
        result = chunker.chunk_page(
            text="Text here.",
            page_id=SAMPLE_PAGE_ID,
            document_id=SAMPLE_DOC_ID,
            page_number=7,
            document_title="Test",
            filename="test.pdf",
            ocr_confidence=90.0,
        )
        assert all(c["page_number"] == 7 for c in result)

    def test_chunk_uses_document_title_in_reference(self, chunker):
        result = chunker.chunk_page(
            text="Text here.",
            page_id=SAMPLE_PAGE_ID,
            document_id=SAMPLE_DOC_ID,
            page_number=3,
            document_title="Victorian Furniture",
            filename="irrelevant.pdf",
            ocr_confidence=90.0,
        )
        assert len(result) >= 1
        assert '"Victorian Furniture"' in result[0]["reference"]

    def test_chunk_falls_back_to_filename_stem_when_no_title(self, chunker):
        result = chunker.chunk_page(
            text="Text here.",
            page_id=SAMPLE_PAGE_ID,
            document_id=SAMPLE_DOC_ID,
            page_number=1,
            document_title=None,
            filename="my_test_file.pdf",
            ocr_confidence=90.0,
        )
        assert len(result) >= 1
        assert '"my_test_file"' in result[0]["reference"]

    def test_reference_format(self, chunker):
        """Reference format must be '"Title" (p.N, chunk M)'."""
        result = chunker.chunk_page(
            text="Some page content.",
            page_id=SAMPLE_PAGE_ID,
            document_id=SAMPLE_DOC_ID,
            page_number=4,
            document_title="The Title",
            filename="the_title.pdf",
            ocr_confidence=90.0,
        )
        assert len(result) >= 1
        ref = result[0]["reference"]
        assert ref.startswith('"The Title"')
        assert "(p.4, chunk 1)" in ref

    def test_chunk_index_sequential(self, chunker):
        """chunk_index values must be 0-based and consecutive."""
        long_text = "\n\n".join([f"Paragraph {i} with some content here." for i in range(10)])
        result = chunker.chunk_page(
            text=long_text,
            page_id=SAMPLE_PAGE_ID,
            document_id=SAMPLE_DOC_ID,
            page_number=1,
            document_title="Test",
            filename="test.pdf",
            ocr_confidence=90.0,
        )
        indices = [c["chunk_index"] for c in result]
        assert indices == list(range(len(result)))

    def test_chunk_type_is_text(self, chunker):
        result = chunker.chunk_page(
            text="Some text.",
            page_id=SAMPLE_PAGE_ID,
            document_id=SAMPLE_DOC_ID,
            page_number=1,
            document_title="Test",
            filename="test.pdf",
            ocr_confidence=90.0,
        )
        assert all(c["chunk_type"] == "text" for c in result)


# ---------------------------------------------------------------------------
# Overlap behaviour
# ---------------------------------------------------------------------------

class TestOverlap:
    def test_consecutive_chunks_share_text(self, chunker):
        """The start of chunk N+1 should contain text from the tail of chunk N."""
        # Build text that will split into multiple chunks
        paragraphs = [
            "Alpha beta gamma delta epsilon zeta eta theta iota kappa. " * 3,
            "Lambda mu nu xi omicron pi rho sigma tau upsilon. " * 3,
            "Phi chi psi omega alpha beta gamma delta epsilon zeta. " * 3,
        ]
        text = "\n\n".join(paragraphs)
        result = chunker.chunk_page(
            text=text,
            page_id=SAMPLE_PAGE_ID,
            document_id=SAMPLE_DOC_ID,
            page_number=1,
            document_title="Test",
            filename="test.pdf",
            ocr_confidence=90.0,
        )
        if len(result) >= 2:
            # Take the last 20 chars of chunk 0
            tail = result[0]["text"][-20:].split()[-1]
            # That last word should appear somewhere in chunk 1
            assert tail in result[1]["text"]

    def test_no_overlap_for_single_chunk(self, chunker_default):
        """A text short enough to fit in one chunk should produce only one chunk."""
        result = chunker_default.chunk_page(
            text="Just a short text.",
            page_id=SAMPLE_PAGE_ID,
            document_id=SAMPLE_DOC_ID,
            page_number=1,
            document_title="Test",
            filename="test.pdf",
            ocr_confidence=90.0,
        )
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _split_paragraphs
# ---------------------------------------------------------------------------

class TestSplitParagraphs:
    def test_splits_on_double_newline(self, chunker):
        result = chunker._split_paragraphs("First paragraph.\n\nSecond paragraph.")
        assert len(result) == 2
        assert result[0] == "First paragraph."
        assert result[1] == "Second paragraph."

    def test_discards_empty_segments(self, chunker):
        result = chunker._split_paragraphs("Para one.\n\n\n\nPara two.")
        assert all(p.strip() for p in result)

    def test_single_newlines_preserved_within_paragraph(self, chunker):
        result = chunker._split_paragraphs("Line one.\nLine two.\n\nSecond para.")
        assert len(result) == 2
        assert "Line one." in result[0]
        assert "Line two." in result[0]

    def test_single_paragraph_returns_one_item(self, chunker):
        result = chunker._split_paragraphs("Just one paragraph here.")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _merge_segments
# ---------------------------------------------------------------------------

class TestMergeSegments:
    def test_merges_short_paragraphs(self, chunker):
        segments = ["Short.", "Also short.", "Another tiny one."]
        result = chunker._merge_segments(segments)
        # All should be merged into one since they are well under 200 chars
        assert len(result) == 1

    def test_splits_overlong_segment(self, chunker):
        long_seg = "word " * 100  # 500 chars >> chunk_size 200
        result = chunker._merge_segments([long_seg])
        assert len(result) > 1
        for seg in result:
            assert len(seg) <= chunker.config.chunk_size

    def test_does_not_exceed_chunk_size_when_merging(self, chunker):
        # Two segments that together exceed chunk_size should stay separate
        seg_a = "a " * 100   # 200 chars
        seg_b = "b " * 100   # 200 chars
        result = chunker._merge_segments([seg_a.strip(), seg_b.strip()])
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# _split_on_whitespace
# ---------------------------------------------------------------------------

class TestSplitOnWhitespace:
    def test_respects_word_boundaries(self, chunker):
        text = "the quick brown fox jumps over the lazy dog " * 20
        pieces = chunker._split_on_whitespace(text, 50)
        for piece in pieces:
            # No piece should be longer than 50 chars
            assert len(piece) <= 50 + 20  # tolerance for long words at boundary
            # Each piece should consist of complete words (no mid-word splits)
            assert piece == piece.strip()

    def test_short_text_produces_one_piece(self, chunker):
        text = "Hello world"
        pieces = chunker._split_on_whitespace(text, 100)
        assert len(pieces) == 1
        assert pieces[0] == "Hello world"

    def test_empty_text_returns_empty_list(self, chunker):
        pieces = chunker._split_on_whitespace("", 100)
        assert pieces == []
