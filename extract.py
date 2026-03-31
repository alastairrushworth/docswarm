"""Extract PDFs into the DuckLake lakehouse.

Configure via environment variables or a .env file:
    DOCSWARM_PDF_DIR        Directory containing source PDFs (required)
    DOCSWARM_CATALOG_PATH   DuckLake catalog file (default: docswarm_catalog.db)
    DOCSWARM_DATA_PATH      Parquet data directory (default: docswarm_data/)
    DOCSWARM_PAGES_DIR      Rendered page images directory (default: pages/)
    DOCSWARM_OCR_LANGUAGE   Tesseract language code (default: eng)
    DOCSWARM_CHUNK_SIZE     Target chunk size in characters (default: 800)
    DOCSWARM_CHUNK_OVERLAP  Overlap between chunks in characters (default: 100)
"""

from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    from docswarm.config import Config
    from docswarm.extraction.pipeline import ExtractionPipeline

    config = Config.from_env()
    pipeline = ExtractionPipeline(config)
    pipeline.run(pdf_dir=config.pdf_dir)
