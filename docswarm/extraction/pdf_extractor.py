"""PDF extraction using PyMuPDF and Tesseract OCR."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance
from pytesseract import Output

from docswarm.logger import get_logger

if TYPE_CHECKING:
    from docswarm.config import Config

log = get_logger(__name__)


class PDFExtractor:
    """Extracts text and images from scanned PDF files using PyMuPDF and Tesseract.

    Each page is rendered to a PNG image at the requested DPI, preprocessed
    (grayscale + contrast enhancement), and then passed through Tesseract OCR.
    The resulting text, confidence scores, and image paths are stored in the
    database via :class:`~docswarm.storage.database.DatabaseManager`.
    """

    def __init__(self, config: "Config") -> None:
        """Initialise the extractor with the given configuration.

        Args:
            config: A :class:`~docswarm.config.Config` instance controlling
                DPI, OCR language, output directories, etc.
        """
        self.config = config
        Path(config.pages_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_document_data(self, pdf_path: str) -> tuple[dict, list[dict]]:
        """Extract a PDF's pages and run OCR without touching the database.

        This is the CPU-bound part of extraction and is designed to run in a
        worker process.  Image files are saved to ``config.pages_dir``; all
        other results are returned as plain dicts for the caller to insert.

        Args:
            pdf_path: Absolute or relative path to the PDF file.

        Returns:
            A ``(document_record, page_records)`` tuple.  ``document_record``
            contains a generated UUID in its ``id`` field.  Each page record
            includes ``raw_text``, ``image_path``, and OCR metadata but no
            ``created_at`` timestamp (added by the database layer).
        """
        pdf_path = str(Path(pdf_path).resolve())
        filename = Path(pdf_path).name

        log.info("Extracting document: %s", filename)
        stat = Path(pdf_path).stat()
        doc = fitz.open(pdf_path)
        try:
            title = Path(filename).stem.replace("_", " ").replace("-", " ").title()
            document_id = str(uuid.uuid4())

            document_record: dict = {
                "id": document_id,
                "filename": filename,
                "filepath": pdf_path,
                "title": title,
                "total_pages": doc.page_count,
                "file_size_bytes": stat.st_size,
            }

            page_records: list[dict] = []
            log.info("Document has %d page(s): %s", doc.page_count, filename)

            for page_num in range(doc.page_count):
                log.debug("OCR page %d/%d: %s", page_num + 1, doc.page_count, filename)
                page = doc[page_num]

                image = self._render_fitz_page(page, dpi=300)

                image_filename = f"{document_id}_page_{page_num + 1:04d}.png"
                image_path = str(Path(self.config.pages_dir) / image_filename)
                image.save(image_path, format="PNG")

                text, confidence = self.ocr_page(image)
                word_count = len(text.split()) if text else 0

                log.debug(
                    "Page %d OCR confidence: %.1f%%, words: %d",
                    page_num + 1, confidence, word_count,
                )

                page_records.append({
                    "document_id": document_id,
                    "page_number": page_num + 1,
                    "width_pts": page.rect.width,
                    "height_pts": page.rect.height,
                    "image_path": image_path,
                    "raw_text": text,
                    "ocr_confidence": confidence,
                    "word_count": word_count,
                })
        finally:
            doc.close()

        log.info("Finished extracting: %s", filename)
        return document_record, page_records

    def _render_fitz_page(self, page: fitz.Page, dpi: int = 300) -> Image.Image:
        """Render an already-open fitz page to a PIL Image.

        Args:
            page: An open :class:`fitz.Page` object.
            dpi: Output resolution in dots per inch.

        Returns:
            A :class:`PIL.Image.Image` in RGB mode.
        """
        zoom = dpi / 72.0  # PyMuPDF uses 72 DPI as its base
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        return img

    def ocr_page(self, image: Image.Image) -> tuple[str, float]:
        """Run Tesseract OCR on a page image.

        The image is preprocessed (grayscale + contrast enhancement) before
        being passed to Tesseract. The confidence score is the mean of all
        per-word confidence values returned by Tesseract (words with a
        confidence of ``-1`` are excluded from the mean).

        Args:
            image: A :class:`PIL.Image.Image` of the page.

        Returns:
            A ``(text, confidence)`` tuple where *text* is the recognised
            string and *confidence* is a float in the range ``[0, 100]``.
            Returns ``("", 0.0)`` if OCR produces no output.
        """
        preprocessed = self._preprocess_image(image)

        data = pytesseract.image_to_data(
            preprocessed,
            lang=self.config.ocr_language,
            output_type=Output.DICT,
        )

        # Gather words and confidences
        words = []
        confidences = []
        for i, word in enumerate(data["text"]):
            conf = data["conf"][i]
            if conf == -1:
                continue
            if word.strip():
                words.append(word)
                confidences.append(float(conf))

        text = " ".join(words)
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return text, mean_confidence

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """Convert image to grayscale and enhance contrast for better OCR.

        Args:
            img: Input :class:`PIL.Image.Image`.

        Returns:
            Preprocessed grayscale image with contrast factor 1.5 applied.
        """
        grayscale = img.convert("L")
        enhanced = ImageEnhance.Contrast(grayscale).enhance(1.5)
        return enhanced
