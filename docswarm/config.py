"""Configuration management for docswarm."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Central configuration for the docswarm pipeline.

    Holds all settings for PDF extraction, database storage,
    OCR, chunking, and Wiki.js integration.
    """

    catalog_path: str = "docswarm_catalog.db"
    data_path: str = "docswarm_data/"
    pdf_dir: str = "."
    pages_dir: str = "pages/"
    wiki_url: str = ""
    wiki_api_key: str = ""
    wiki_output_dir: str = "wiki/"
    ollama_base_url: str = "http://localhost:11434"
    ocr_language: str = "eng"
    chunk_size: int = 800
    chunk_overlap: int = 100
    model: str = "gemma3:4b"

    @classmethod
    def from_env(cls) -> "Config":
        """Create a Config instance populated from environment variables.

        Environment variables:
            DOCSWARM_CATALOG_PATH: Path to the DuckLake catalog (DuckDB) file.
            DOCSWARM_DATA_PATH: Directory where DuckLake stores Parquet data files.
            DOCSWARM_PDF_DIR: Directory containing source PDFs.
            DOCSWARM_PAGES_DIR: Directory to store rendered page images.
            DOCSWARM_WIKI_URL: Wiki.js base URL (for sync_wiki.py).
            DOCSWARM_WIKI_API_KEY: Wiki.js API key (for sync_wiki.py).
            DOCSWARM_WIKI_OUTPUT_DIR: Local directory for generated markdown articles.
            DOCSWARM_OLLAMA_BASE_URL: Ollama server URL.
            DOCSWARM_OCR_LANGUAGE: Tesseract language code.
            DOCSWARM_CHUNK_SIZE: Target chunk size in characters.
            DOCSWARM_CHUNK_OVERLAP: Overlap between chunks in characters.
            DOCSWARM_MODEL: Ollama model name (e.g. gemma3:4b).

        Returns:
            Config: Populated configuration instance.
        """
        return cls(
            catalog_path=os.environ.get("DOCSWARM_CATALOG_PATH", "docswarm_catalog.db"),
            data_path=os.environ.get("DOCSWARM_DATA_PATH", "docswarm_data/"),
            pdf_dir=os.environ.get("DOCSWARM_PDF_DIR", "."),
            pages_dir=os.environ.get("DOCSWARM_PAGES_DIR", "pages/"),
            wiki_url=os.environ.get("DOCSWARM_WIKI_URL", ""),
            wiki_api_key=os.environ.get("DOCSWARM_WIKI_API_KEY", ""),
            wiki_output_dir=os.environ.get("DOCSWARM_WIKI_OUTPUT_DIR", "wiki/"),
            ollama_base_url=os.environ.get("DOCSWARM_OLLAMA_BASE_URL", "http://localhost:11434"),
            ocr_language=os.environ.get("DOCSWARM_OCR_LANGUAGE", "eng"),
            chunk_size=int(os.environ.get("DOCSWARM_CHUNK_SIZE", "800")),
            chunk_overlap=int(os.environ.get("DOCSWARM_CHUNK_OVERLAP", "100")),
            model=os.environ.get("DOCSWARM_MODEL", "gemma3:4b"),
        )
