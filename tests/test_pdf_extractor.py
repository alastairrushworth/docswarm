"""Tests for docswarm.extraction.pdf_extractor.PDFExtractor.

All fitz (PyMuPDF) and pytesseract calls are mocked so no real PDF or
Tesseract installation is needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from PIL import Image

from docswarm.config import Config
from docswarm.extraction.pdf_extractor import PDFExtractor

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def extractor(tmp_path):
    cfg = Config(pages_dir=str(tmp_path / "pages/"))
    return PDFExtractor(cfg)


def _make_rgb_image(width=100, height=100) -> Image.Image:
    """Create a minimal RGB PIL image for testing."""
    return Image.new("RGB", (width, height), color=(200, 200, 200))


def _mock_fitz_doc(page_count=2, page_width=595.0, page_height=842.0):
    """Return a mock fitz.Document with ``page_count`` pages."""
    mock_doc = MagicMock()
    mock_doc.page_count = page_count

    mock_page = MagicMock()
    mock_page.rect.width = page_width
    mock_page.rect.height = page_height

    mock_doc.__getitem__ = MagicMock(return_value=mock_page)
    mock_doc.close = MagicMock()
    return mock_doc, mock_page


# ---------------------------------------------------------------------------
# extract_document_data
# ---------------------------------------------------------------------------


class TestExtractDocumentData:
    def test_returns_document_and_page_records_tuple(self, extractor, tmp_path):
        fake_pdf = tmp_path / "sample.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        mock_doc, _ = _mock_fitz_doc(page_count=3)

        with (
            patch("fitz.open", return_value=mock_doc),
            patch.object(extractor, "_render_fitz_page", return_value=_make_rgb_image()),
            patch.object(extractor, "ocr_page", return_value=("Sample OCR text here.", 91.5)),
        ):
            doc_record, page_records = extractor.extract_document_data(str(fake_pdf))

        assert isinstance(doc_record, dict)
        assert isinstance(page_records, list)

    def test_document_record_has_required_fields(self, extractor, tmp_path):
        fake_pdf = tmp_path / "antique_furniture.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        mock_doc, _ = _mock_fitz_doc(page_count=1)

        with (
            patch("fitz.open", return_value=mock_doc),
            patch.object(extractor, "_render_fitz_page", return_value=_make_rgb_image()),
            patch.object(extractor, "ocr_page", return_value=("text", 90.0)),
        ):
            doc_record, _ = extractor.extract_document_data(str(fake_pdf))

        required = {"id", "filename", "filepath", "title", "total_pages", "file_size_bytes"}
        assert required.issubset(doc_record.keys())

    def test_document_record_filename_is_basename(self, extractor, tmp_path):
        fake_pdf = tmp_path / "my_report.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        mock_doc, _ = _mock_fitz_doc(page_count=1)

        with (
            patch("fitz.open", return_value=mock_doc),
            patch.object(extractor, "_render_fitz_page", return_value=_make_rgb_image()),
            patch.object(extractor, "ocr_page", return_value=("text", 90.0)),
        ):
            doc_record, _ = extractor.extract_document_data(str(fake_pdf))

        assert doc_record["filename"] == "my_report.pdf"

    def test_document_record_total_pages_matches_fitz(self, extractor, tmp_path):
        fake_pdf = tmp_path / "multipage.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        mock_doc, _ = _mock_fitz_doc(page_count=5)

        with (
            patch("fitz.open", return_value=mock_doc),
            patch.object(extractor, "_render_fitz_page", return_value=_make_rgb_image()),
            patch.object(extractor, "ocr_page", return_value=("text", 90.0)),
        ):
            doc_record, _ = extractor.extract_document_data(str(fake_pdf))

        assert doc_record["total_pages"] == 5

    def test_page_records_count_matches_page_count(self, extractor, tmp_path):
        fake_pdf = tmp_path / "three_pages.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        mock_doc, _ = _mock_fitz_doc(page_count=3)

        with (
            patch("fitz.open", return_value=mock_doc),
            patch.object(extractor, "_render_fitz_page", return_value=_make_rgb_image()),
            patch.object(extractor, "ocr_page", return_value=("text", 88.0)),
        ):
            _, page_records = extractor.extract_document_data(str(fake_pdf))

        assert len(page_records) == 3

    def test_page_records_have_sequential_page_numbers(self, extractor, tmp_path):
        fake_pdf = tmp_path / "pages.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        mock_doc, _ = _mock_fitz_doc(page_count=4)

        with (
            patch("fitz.open", return_value=mock_doc),
            patch.object(extractor, "_render_fitz_page", return_value=_make_rgb_image()),
            patch.object(extractor, "ocr_page", return_value=("text", 90.0)),
        ):
            _, page_records = extractor.extract_document_data(str(fake_pdf))

        page_numbers = [p["page_number"] for p in page_records]
        assert page_numbers == [1, 2, 3, 4]

    def test_title_derived_from_filename_title_cased(self, extractor, tmp_path):
        """Title should be derived from filename stem with underscores/hyphens replaced by spaces."""
        fake_pdf = tmp_path / "victorian_furniture_catalogue.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        mock_doc, _ = _mock_fitz_doc(page_count=1)

        with (
            patch("fitz.open", return_value=mock_doc),
            patch.object(extractor, "_render_fitz_page", return_value=_make_rgb_image()),
            patch.object(extractor, "ocr_page", return_value=("text", 90.0)),
        ):
            doc_record, _ = extractor.extract_document_data(str(fake_pdf))

        assert doc_record["title"] == "Victorian Furniture Catalogue"

    def test_title_replaces_hyphens_with_spaces(self, extractor, tmp_path):
        fake_pdf = tmp_path / "antique-book-collection.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        mock_doc, _ = _mock_fitz_doc(page_count=1)

        with (
            patch("fitz.open", return_value=mock_doc),
            patch.object(extractor, "_render_fitz_page", return_value=_make_rgb_image()),
            patch.object(extractor, "ocr_page", return_value=("text", 90.0)),
        ):
            doc_record, _ = extractor.extract_document_data(str(fake_pdf))

        assert doc_record["title"] == "Antique Book Collection"

    def test_document_record_has_id(self, extractor, tmp_path):
        fake_pdf = tmp_path / "check_id.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        mock_doc, _ = _mock_fitz_doc(page_count=1)

        with (
            patch("fitz.open", return_value=mock_doc),
            patch.object(extractor, "_render_fitz_page", return_value=_make_rgb_image()),
            patch.object(extractor, "ocr_page", return_value=("text", 90.0)),
        ):
            doc_record, _ = extractor.extract_document_data(str(fake_pdf))

        assert doc_record["id"]
        assert len(doc_record["id"]) == 36  # UUID format

    def test_document_record_file_size_bytes_nonzero(self, extractor, tmp_path):
        fake_pdf = tmp_path / "size_check.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 some content here so it has nonzero size")

        mock_doc, _ = _mock_fitz_doc(page_count=1)

        with (
            patch("fitz.open", return_value=mock_doc),
            patch.object(extractor, "_render_fitz_page", return_value=_make_rgb_image()),
            patch.object(extractor, "ocr_page", return_value=("text", 90.0)),
        ):
            doc_record, _ = extractor.extract_document_data(str(fake_pdf))

        assert doc_record["file_size_bytes"] > 0

    def test_page_records_include_raw_text(self, extractor, tmp_path):
        fake_pdf = tmp_path / "text_check.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        mock_doc, _ = _mock_fitz_doc(page_count=1)

        with (
            patch("fitz.open", return_value=mock_doc),
            patch.object(extractor, "_render_fitz_page", return_value=_make_rgb_image()),
            patch.object(extractor, "ocr_page", return_value=("Hello OCR world", 95.0)),
        ):
            _, page_records = extractor.extract_document_data(str(fake_pdf))

        assert page_records[0]["raw_text"] == "Hello OCR world"

    def test_calling_twice_does_not_raise(self, extractor, tmp_path):
        """PDFExtractor itself has no idempotency guard — calling twice is fine."""
        fake_pdf = tmp_path / "idempotent.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        mock_doc, _ = _mock_fitz_doc(page_count=1)

        with (
            patch("fitz.open", return_value=mock_doc),
            patch.object(extractor, "_render_fitz_page", return_value=_make_rgb_image()),
            patch.object(extractor, "ocr_page", return_value=("text", 90.0)),
        ):
            doc1, _ = extractor.extract_document_data(str(fake_pdf))

        mock_doc2, _ = _mock_fitz_doc(page_count=1)

        with (
            patch("fitz.open", return_value=mock_doc2),
            patch.object(extractor, "_render_fitz_page", return_value=_make_rgb_image()),
            patch.object(extractor, "ocr_page", return_value=("text", 90.0)),
        ):
            doc2, _ = extractor.extract_document_data(str(fake_pdf))

        # Both calls return distinct document IDs (two separate extractions)
        assert doc1["id"] != doc2["id"]


# ---------------------------------------------------------------------------
# ocr_page
# ---------------------------------------------------------------------------


class TestOcrPage:
    def test_returns_text_and_confidence_tuple(self, extractor):
        mock_data = {
            "text": ["Hello", "world", ""],
            "conf": [95, 87, -1],
        }

        with patch("pytesseract.image_to_data", return_value=mock_data):
            text, confidence = extractor.ocr_page(_make_rgb_image())

        assert isinstance(text, str)
        assert isinstance(confidence, float)

    def test_returned_text_joins_words(self, extractor):
        mock_data = {
            "text": ["Thomas", "Chippendale", ""],
            "conf": [92, 89, -1],
        }

        with patch("pytesseract.image_to_data", return_value=mock_data):
            text, _ = extractor.ocr_page(_make_rgb_image())

        assert "Thomas" in text
        assert "Chippendale" in text

    def test_excludes_conf_minus_one_from_mean(self, extractor):
        mock_data = {
            "text": ["word", ""],
            "conf": [80, -1],
        }

        with patch("pytesseract.image_to_data", return_value=mock_data):
            _, confidence = extractor.ocr_page(_make_rgb_image())

        assert confidence == pytest.approx(80.0)

    def test_returns_zero_confidence_for_empty_output(self, extractor):
        mock_data = {
            "text": [""],
            "conf": [-1],
        }

        with patch("pytesseract.image_to_data", return_value=mock_data):
            text, confidence = extractor.ocr_page(_make_rgb_image())

        assert text == ""
        assert confidence == 0.0

    def test_confidence_is_mean_of_valid_words(self, extractor):
        mock_data = {
            "text": ["a", "b", "c"],
            "conf": [60, 80, 100],
        }

        with patch("pytesseract.image_to_data", return_value=mock_data):
            _, confidence = extractor.ocr_page(_make_rgb_image())

        assert confidence == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# _preprocess_image
# ---------------------------------------------------------------------------


class TestPreprocessImage:
    def test_returns_grayscale_image(self, extractor):
        rgb_img = _make_rgb_image(80, 60)
        result = extractor._preprocess_image(rgb_img)
        assert result.mode == "L"

    def test_output_has_same_dimensions(self, extractor):
        rgb_img = _make_rgb_image(120, 90)
        result = extractor._preprocess_image(rgb_img)
        assert result.size == (120, 90)

    def test_grayscale_input_works(self, extractor):
        gray_img = Image.new("L", (50, 50), color=128)
        result = extractor._preprocess_image(gray_img)
        assert result.mode == "L"
