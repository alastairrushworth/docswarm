"""Shared pytest fixtures for the docswarm test suite."""

from __future__ import annotations

import pytest

from docswarm.config import Config
from docswarm.storage.database import DatabaseManager


@pytest.fixture
def tmp_config(tmp_path):
    """Config pointing entirely at tmp_path so tests never touch real data."""
    catalog = str(tmp_path / "test_catalog.db")
    data = str(tmp_path / "test_data/")
    pages = str(tmp_path / "pages/")
    wiki_out = str(tmp_path / "wiki/")
    return Config(
        catalog_path=catalog,
        data_path=data,
        pdf_dir=str(tmp_path / "pdfs/"),
        pages_dir=pages,
        wiki_url="",
        wiki_api_key="",
        wiki_output_dir=wiki_out,
        ollama_base_url="http://localhost:11434",
        ocr_language="eng",
        chunk_size=800,
        chunk_overlap=100,
        model="gemma3:4b",
    )


@pytest.fixture
def db(tmp_config):
    """Initialised DatabaseManager backed by a temporary directory.

    Yielded so the connection is closed cleanly after each test.
    """
    manager = DatabaseManager(tmp_config.catalog_path, tmp_config.data_path)
    manager.initialize()
    yield manager
    manager.close()


@pytest.fixture
def sample_document(db):
    """Insert a single test document and return the full record dict."""
    doc = {
        "filename": "test_document.pdf",
        "filepath": "/tmp/test_document.pdf",
        "title": "Test Document",
        "total_pages": 3,
        "file_size_bytes": 102400,
    }
    doc_id = db.insert_document(doc)
    doc["id"] = doc_id
    return doc


@pytest.fixture
def sample_page(db, sample_document):
    """Insert a test page (with raw_text) for sample_document and return its dict."""
    page = {
        "document_id": sample_document["id"],
        "page_number": 1,
        "width_pts": 595.0,
        "height_pts": 842.0,
        "image_path": "/tmp/page_0001.png",
        "raw_text": (
            "This is the first paragraph of the test page.\n\n"
            "It contains some interesting information about antique furniture.\n\n"
            "Thomas Chippendale was a famous English cabinet-maker."
        ),
        "ocr_confidence": 92.5,
        "word_count": 30,
    }
    page_id = db.insert_page(page)
    page["id"] = page_id
    return page


@pytest.fixture
def sample_chunks(db, sample_page, sample_document):
    """Insert 3 test chunks for sample_page and return the list of chunk dicts."""
    chunks = []
    texts = [
        "This is the first paragraph of the test page.",
        "It contains some interesting information about antique furniture.",
        "Thomas Chippendale was a famous English cabinet-maker.",
    ]
    for idx, text in enumerate(texts):
        chunk = {
            "document_id": sample_document["id"],
            "page_id": sample_page["id"],
            "page_number": 1,
            "chunk_index": idx,
            "text": text,
            "char_start": idx * 60,
            "char_end": idx * 60 + len(text),
            "chunk_type": "text",
            "ocr_confidence": 92.5,
            "reference": f'"Test Document" (p.1, chunk {idx + 1})',
            "word_count": len(text.split()),
        }
        chunk_id = db.insert_chunk(chunk)
        chunk["id"] = chunk_id
        chunks.append(chunk)
    return chunks
