"""End-to-end PDF extraction pipeline."""

from __future__ import annotations

import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from docswarm.logger import get_logger

if TYPE_CHECKING:
    from docswarm.config import Config

console = Console()
log = get_logger(__name__)

_NUM_WORKERS = 5


def _extract_worker(pdf_path: str, config_dict: dict) -> dict:
    """Worker function: OCR + chunk a single PDF, return data (no DB access).

    Runs in a subprocess.  Accepts a plain dict for config so it is
    picklable across process boundaries.

    Returns a dict with keys:
        document_record  – document metadata dict
        page_records     – list of page metadata dicts (includes raw_text)
        chunks_by_page   – {page_index: [chunk_dicts]} keyed by 0-based index
    """
    from docswarm.config import Config
    from docswarm.extraction.chunker import TextChunker
    from docswarm.extraction.pdf_extractor import PDFExtractor

    config = Config(**config_dict)
    extractor = PDFExtractor(config)
    chunker = TextChunker(config)

    document_record, page_records = extractor.extract_document_data(pdf_path)

    chunks_by_page: dict[int, list[dict]] = {}
    for i, page in enumerate(page_records):
        page_id = str(uuid.uuid4())
        page["id"] = page_id
        chunks = chunker.chunk_page(
            text=page.get("raw_text") or "",
            page_id=page_id,
            document_id=document_record["id"],
            page_number=page["page_number"],
            document_title=document_record.get("title"),
            filename=document_record["filename"],
            ocr_confidence=page.get("ocr_confidence") or 0.0,
        )
        chunks_by_page[i] = chunks

    return {
        "document_record": document_record,
        "page_records": page_records,
        "chunks_by_page": chunks_by_page,
    }


class ExtractionPipeline:
    """Orchestrates parallel PDF extraction and chunking.

    OCR and chunking run in up to ``_NUM_WORKERS`` parallel worker processes.
    Database writes happen sequentially in the main process to avoid concurrent
    write conflicts with DuckLake.
    """

    def __init__(self, config: "Config") -> None:
        from docswarm.storage.database import DatabaseManager

        self.config = config
        self.db = DatabaseManager(config.catalog_path, config.data_path)
        self.db.initialize()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, pdf_dir: str | None = None) -> None:
        target_dir = Path(pdf_dir or self.config.pdf_dir)
        if not target_dir.exists():
            log.error("PDF directory not found: %s", target_dir)
            console.print(f"[red]PDF directory not found:[/red] {target_dir}")
            return

        pdf_files = sorted(target_dir.glob("**/*.pdf"))
        if not pdf_files:
            log.warning("No PDF files found in %s", target_dir)
            console.print(f"[yellow]No PDF files found in[/yellow] {target_dir}")
            return

        log.info("Found %d PDF(s) in %s", len(pdf_files), target_dir)
        console.print(
            f"[bold green]docswarm[/bold green] — found [bold]{len(pdf_files)}[/bold] "
            f"PDF(s) in [cyan]{target_dir}[/cyan]"
        )

        # Idempotency: filter out already-ingested PDFs in the main process
        to_process: list[Path] = []
        skipped = 0
        for pdf_path in pdf_files:
            if self.db.get_document_by_filename(pdf_path.name):
                log.info("Skipping (already ingested): %s", pdf_path.name)
                skipped += 1
            else:
                to_process.append(pdf_path)

        stats = {"processed": 0, "skipped": skipped, "chunks": 0, "pages": 0}

        if not to_process:
            log.info("All PDFs already ingested")
            self._print_summary(stats, len(pdf_files))
            return

        log.info(
            "Submitting %d PDF(s) to %d worker process(es)", len(to_process), _NUM_WORKERS
        )

        config_dict = {
            "catalog_path": self.config.catalog_path,
            "data_path": self.config.data_path,
            "pdf_dir": self.config.pdf_dir,
            "pages_dir": self.config.pages_dir,
            "wiki_url": self.config.wiki_url,
            "wiki_api_key": self.config.wiki_api_key,
            "ollama_base_url": self.config.ollama_base_url,
            "ocr_language": self.config.ocr_language,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "model": self.config.model,
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting PDFs…", total=len(to_process))

            with ProcessPoolExecutor(max_workers=_NUM_WORKERS) as pool:
                futures = {
                    pool.submit(_extract_worker, str(p), config_dict): p
                    for p in to_process
                }

                for future in as_completed(futures):
                    pdf_path = futures[future]
                    progress.update(task, description=f"[cyan]{pdf_path.name}[/cyan]")

                    try:
                        result = future.result()
                    except Exception as exc:
                        log.error("Failed to extract %s: %s", pdf_path.name, exc)
                        console.print(f"[red]Error:[/red] {pdf_path.name} — {exc}")
                        progress.advance(task)
                        continue

                    page_count, chunk_count = self._insert_result(result)
                    stats["processed"] += 1
                    stats["pages"] += page_count
                    stats["chunks"] += chunk_count
                    log.info(
                        "Inserted: %s — %d page(s), %d chunk(s)",
                        pdf_path.name, page_count, chunk_count,
                    )
                    progress.advance(task)

        log.info(
            "Extraction complete — processed: %d, skipped: %d, pages: %d, chunks: %d",
            stats["processed"], stats["skipped"], stats["pages"], stats["chunks"],
        )
        self._print_summary(stats, len(pdf_files))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _insert_result(self, result: dict) -> tuple[int, int]:
        """Insert one worker's extraction result into the database.

        Returns:
            ``(page_count, chunk_count)``
        """
        doc_record = result["document_record"]
        page_records = result["page_records"]
        chunks_by_page = result["chunks_by_page"]

        self.db.insert_document(doc_record)

        total_chunks = 0
        for i, page in enumerate(page_records):
            self.db.insert_page(page)
            for chunk in chunks_by_page.get(i, []):
                self.db.insert_chunk(chunk)
            total_chunks += len(chunks_by_page.get(i, []))

        return len(page_records), total_chunks

    def _print_summary(self, stats: dict, total_pdfs: int) -> None:
        table = Table(title="Extraction Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="bold")

        table.add_row("Total PDFs found", str(total_pdfs))
        table.add_row("Newly processed", str(stats["processed"]))
        table.add_row("Skipped (already ingested)", str(stats["skipped"]))
        table.add_row("Pages extracted", str(stats["pages"]))
        table.add_row("Chunks created", str(stats["chunks"]))

        console.print(table)
