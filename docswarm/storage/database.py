"""DuckLake database manager for docswarm."""

from __future__ import annotations

import threading
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from docswarm.logger import get_logger

log = get_logger(__name__)

_LAKE = "docswarm"  # DuckLake attachment name used throughout


def _now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class DatabaseManager:
    """Manages a DuckLake lakehouse storing documents, pages, chunks, and wiki articles.

    All public methods are thread-safe.  LangGraph's ``ToolNode`` dispatches
    tool calls to a ``ThreadPoolExecutor``, so multiple agent tools may hit
    the database concurrently.  DuckDB connections are **not** thread-safe —
    concurrent access from different threads causes SIGSEGV on macOS ARM.
    A reentrant lock serialises every database operation.
    """

    def __init__(self, catalog_path: str, data_path: str = "docswarm_data/") -> None:
        self.catalog_path = str(Path(catalog_path).resolve())
        self.data_path = str(Path(data_path).resolve()) + "/"

        Path(data_path).mkdir(parents=True, exist_ok=True)

        log.info("Opening DuckLake catalog: %s", self.catalog_path)
        self._lock = threading.RLock()
        self.conn = duckdb.connect()
        try:
            self._load_ducklake()
            self._attach()
        except Exception:
            self.conn.close()
            raise

    # ------------------------------------------------------------------
    # DuckLake bootstrap
    # ------------------------------------------------------------------

    def _load_ducklake(self) -> None:
        try:
            self.conn.execute("LOAD ducklake")
            log.debug("DuckLake extension loaded")
        except Exception:
            log.info("Installing DuckLake extension…")
            self.conn.execute("INSTALL ducklake")
            self.conn.execute("LOAD ducklake")

    def _attach(self) -> None:
        self.conn.execute(
            f"ATTACH IF NOT EXISTS 'ducklake:{self.catalog_path}' AS {_LAKE} "
            f"(DATA_PATH '{self.data_path}')"
        )
        log.debug("Attached DuckLake catalog as '%s'", _LAKE)

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create all DuckLake tables if they do not already exist."""
        log.info("Initialising database schema")

        # DEFAULT NOW() and DEFAULT 'text' are not supported by DuckLake —
        # timestamps and defaults are supplied explicitly in insert methods.
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {_LAKE}.documents (
                id VARCHAR,
                filename VARCHAR,
                filepath VARCHAR,
                title VARCHAR,
                total_pages INTEGER,
                file_size_bytes BIGINT,
                ingested_at TIMESTAMP
            )
        """)

        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {_LAKE}.pages (
                id VARCHAR,
                document_id VARCHAR,
                page_number INTEGER,
                width_pts FLOAT,
                height_pts FLOAT,
                image_path VARCHAR,
                raw_text TEXT,
                ocr_confidence FLOAT,
                word_count INTEGER,
                created_at TIMESTAMP
            )
        """)

        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {_LAKE}.chunks (
                id VARCHAR,
                document_id VARCHAR,
                page_id VARCHAR,
                page_number INTEGER,
                chunk_index INTEGER,
                text TEXT,
                char_start INTEGER,
                char_end INTEGER,
                bbox_x0 FLOAT,
                bbox_y0 FLOAT,
                bbox_x1 FLOAT,
                bbox_y1 FLOAT,
                chunk_type VARCHAR,
                ocr_confidence FLOAT,
                reference VARCHAR,
                word_count INTEGER,
                created_at TIMESTAMP
            )
        """)

        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {_LAKE}.wiki_articles (
                id VARCHAR,
                wiki_page_id INTEGER,
                title VARCHAR,
                path VARCHAR,
                content TEXT,
                source_chunk_ids VARCHAR[],
                source_document_ids VARCHAR[],
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)

        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {_LAKE}.page_studies (
                id VARCHAR,
                page_id VARCHAR,
                document_id VARCHAR,
                studied_at TIMESTAMP,
                wiki_article_path VARCHAR
            )
        """)

        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {_LAKE}.entities (
                id VARCHAR,
                name VARCHAR,
                entity_type VARCHAR,
                created_at TIMESTAMP
            )
        """)

        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {_LAKE}.entity_mentions (
                id VARCHAR,
                entity_id VARCHAR,
                page_id VARCHAR,
                document_id VARCHAR,
                context_text TEXT,
                created_at TIMESTAMP
            )
        """)

        log.info("Schema ready")

    # ------------------------------------------------------------------
    # Thread-safe query helpers
    # ------------------------------------------------------------------

    def _query_one(self, sql: str, params: list | None = None) -> dict[str, Any] | None:
        """Execute *sql* and return the first row as a dict, or ``None``."""
        with self._lock:
            cur = self.conn.execute(sql, params) if params else self.conn.execute(sql)
            row = cur.fetchone()
            if row is None:
                return None
            cols = [desc[0] for desc in cur.description]
            return dict(zip(cols, row))

    def _query_all(self, sql: str, params: list | None = None) -> list[dict[str, Any]]:
        """Execute *sql* and return all rows as a list of dicts."""
        with self._lock:
            cur = self.conn.execute(sql, params) if params else self.conn.execute(sql)
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def _exec(self, sql: str, params: list | None = None) -> None:
        """Execute a write statement (INSERT / UPDATE / CREATE) under the lock."""
        with self._lock:
            if params:
                self.conn.execute(sql, params)
            else:
                self.conn.execute(sql)

    @staticmethod
    def _new_id() -> str:
        return str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def insert_document(self, doc: dict) -> str:
        doc_id = doc.get("id") or self._new_id()
        self._exec(
            f"""
            INSERT INTO {_LAKE}.documents
                (id, filename, filepath, title, total_pages, file_size_bytes, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                doc_id,
                doc.get("filename"),
                doc.get("filepath"),
                doc.get("title"),
                doc.get("total_pages"),
                doc.get("file_size_bytes"),
                _now(),
            ],
        )
        log.debug("Inserted document: %s (%s)", doc.get("filename"), doc_id)
        return doc_id

    def get_document(self, doc_id: str) -> dict | None:
        return self._query_one(f"SELECT * FROM {_LAKE}.documents WHERE id = ?", [doc_id])

    def get_document_by_filename(self, filename: str) -> dict | None:
        return self._query_one(f"SELECT * FROM {_LAKE}.documents WHERE filename = ?", [filename])

    def list_documents(self) -> list[dict]:
        return self._query_all(f"SELECT * FROM {_LAKE}.documents ORDER BY ingested_at DESC")

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    def insert_page(self, page: dict) -> str:
        page_id = page.get("id") or self._new_id()
        self._exec(
            f"""
            INSERT INTO {_LAKE}.pages (
                id, document_id, page_number, width_pts, height_pts,
                image_path, raw_text, ocr_confidence, word_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                page_id,
                page.get("document_id"),
                page.get("page_number"),
                page.get("width_pts"),
                page.get("height_pts"),
                page.get("image_path"),
                page.get("raw_text"),
                page.get("ocr_confidence"),
                page.get("word_count"),
                _now(),
            ],
        )
        log.debug("Inserted page %s (doc %s)", page.get("page_number"), page.get("document_id"))
        return page_id

    def get_page(self, page_id: str) -> dict | None:
        return self._query_one(f"SELECT * FROM {_LAKE}.pages WHERE id = ?", [page_id])

    def get_document_pages(self, document_id: str) -> list[dict]:
        return self._query_all(
            f"SELECT * FROM {_LAKE}.pages WHERE document_id = ? ORDER BY page_number",
            [document_id],
        )

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    def insert_chunk(self, chunk: dict) -> str:
        chunk_id = chunk.get("id") or self._new_id()
        self._exec(
            f"""
            INSERT INTO {_LAKE}.chunks (
                id, document_id, page_id, page_number, chunk_index,
                text, char_start, char_end,
                bbox_x0, bbox_y0, bbox_x1, bbox_y1,
                chunk_type, ocr_confidence, reference, word_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                chunk_id,
                chunk.get("document_id"),
                chunk.get("page_id"),
                chunk.get("page_number"),
                chunk.get("chunk_index"),
                chunk.get("text"),
                chunk.get("char_start"),
                chunk.get("char_end"),
                chunk.get("bbox_x0"),
                chunk.get("bbox_y0"),
                chunk.get("bbox_x1"),
                chunk.get("bbox_y1"),
                chunk.get("chunk_type", "text"),
                chunk.get("ocr_confidence"),
                chunk.get("reference"),
                chunk.get("word_count"),
                _now(),
            ],
        )
        log.debug(
            "Inserted chunk %s/%s (doc %s)",
            chunk.get("page_number"),
            chunk.get("chunk_index"),
            chunk.get("document_id"),
        )
        return chunk_id

    def get_chunk(self, chunk_id: str) -> dict | None:
        return self._query_one(f"SELECT * FROM {_LAKE}.chunks WHERE id = ?", [chunk_id])

    def get_page_chunks(self, page_id: str) -> list[dict]:
        return self._query_all(
            f"SELECT * FROM {_LAKE}.chunks WHERE page_id = ? ORDER BY chunk_index",
            [page_id],
        )

    def get_chunks_by_document(
        self, document_id: str, page_number: int | None = None
    ) -> list[dict]:
        if page_number is not None:
            return self._query_all(
                f"""
                SELECT * FROM {_LAKE}.chunks
                WHERE document_id = ? AND page_number = ?
                ORDER BY page_number, chunk_index
                """,
                [document_id, page_number],
            )
        return self._query_all(
            f"""
            SELECT * FROM {_LAKE}.chunks
            WHERE document_id = ?
            ORDER BY page_number, chunk_index
            """,
            [document_id],
        )

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        terms = [t for t in query.split() if t]
        if not terms:
            return []
        conditions = " AND ".join(["text ILIKE ?" for _ in terms])
        params: list[Any] = [f"%{t}%" for t in terms] + [limit]
        results = self._query_all(
            f"""
            SELECT * FROM {_LAKE}.chunks
            WHERE {conditions}
            ORDER BY page_number, chunk_index
            LIMIT ?
            """,
            params,
        )
        log.debug("search_chunks(%r) → %d results", query, len(results))
        return results

    # ------------------------------------------------------------------
    # Wiki articles
    # ------------------------------------------------------------------

    def upsert_wiki_article(self, article: dict) -> str:
        path = article.get("path", "")
        existing = self._query_one(f"SELECT id FROM {_LAKE}.wiki_articles WHERE path = ?", [path])

        source_chunk_ids = article.get("source_chunk_ids") or []
        source_document_ids = article.get("source_document_ids") or []

        if existing:
            article_id = existing["id"]
            self._exec(
                f"""
                UPDATE {_LAKE}.wiki_articles
                SET wiki_page_id = ?,
                    title = ?,
                    content = ?,
                    source_chunk_ids = ?,
                    source_document_ids = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                [
                    article.get("wiki_page_id"),
                    article.get("title"),
                    article.get("content"),
                    source_chunk_ids,
                    source_document_ids,
                    _now(),
                    article_id,
                ],
            )
            log.info("Updated wiki article: %s", path)
        else:
            article_id = article.get("id") or self._new_id()
            now = _now()
            self._exec(
                f"""
                INSERT INTO {_LAKE}.wiki_articles (
                    id, wiki_page_id, title, path, content,
                    source_chunk_ids, source_document_ids, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    article_id,
                    article.get("wiki_page_id"),
                    article.get("title"),
                    path,
                    article.get("content"),
                    source_chunk_ids,
                    source_document_ids,
                    now,
                    now,
                ],
            )
            log.info("Inserted wiki article: %s", path)
        return article_id

    def get_wiki_articles(self) -> list[dict]:
        return self._query_all(f"SELECT * FROM {_LAKE}.wiki_articles ORDER BY created_at DESC")

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    def upsert_entity(self, name: str, entity_type: str) -> str:
        """Insert a new entity or return the existing ID if name already exists.

        Args:
            name: Canonical entity name (case-sensitive).
            entity_type: One of 'person', 'place', 'event', 'object',
                'organisation', or similar.

        Returns:
            The UUID of the entity record.
        """
        with self._lock:
            cur = self.conn.execute(f"SELECT id FROM {_LAKE}.entities WHERE name = ?", [name])
            row = cur.fetchone()
            if row is not None:
                return row[0]

            entity_id = self._new_id()
            self.conn.execute(
                f"INSERT INTO {_LAKE}.entities (id, name, entity_type, created_at) VALUES (?, ?, ?, ?)",
                [entity_id, name, entity_type, _now()],
            )
        log.debug("New entity: %s (%s) → %s", name, entity_type, entity_id)
        return entity_id

    def add_entity_mention(
        self, entity_id: str, page_id: str, document_id: str, context_text: str = ""
    ) -> str:
        """Record that an entity appears on a specific page.

        Args:
            entity_id: UUID of the entity.
            page_id: UUID of the page (references pages.id).
            document_id: UUID of the parent document.
            context_text: Short excerpt of surrounding text.

        Returns:
            The UUID of the new mention record.
        """
        mention_id = self._new_id()
        self._exec(
            f"""
            INSERT INTO {_LAKE}.entity_mentions
                (id, entity_id, page_id, document_id, context_text, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [mention_id, entity_id, page_id, document_id, context_text, _now()],
        )
        log.debug("Entity mention: entity=%s page=%s", entity_id, page_id)
        return mention_id

    def search_entities(self, query: str, limit: int = 10) -> list[dict]:
        """Search entities by name using case-insensitive LIKE matching.

        Args:
            query: Search string.
            limit: Maximum results.

        Returns:
            List of matching entity dicts.
        """
        results = self._query_all(
            f"SELECT * FROM {_LAKE}.entities WHERE name ILIKE ? LIMIT ?",
            [f"%{query}%", limit],
        )
        log.debug("search_entities(%r) → %d results", query, len(results))
        return results

    def get_entity_by_name(self, name: str) -> dict | None:
        """Retrieve an entity by exact name."""
        return self._query_one(f"SELECT * FROM {_LAKE}.entities WHERE name = ?", [name])

    def get_entity_mentions(self, entity_id: str) -> list[dict]:
        """Return all page mentions for an entity, ordered by document and page."""
        mentions = self._query_all(
            f"SELECT * FROM {_LAKE}.entity_mentions WHERE entity_id = ?",
            [entity_id],
        )
        # Enrich with page_number via separate per-row lookups (avoids JOIN crash)
        for m in mentions:
            page = self.get_page(m.get("page_id", ""))
            if page:
                m["page_number"] = page.get("page_number")
        return sorted(mentions, key=lambda m: (m.get("document_id", ""), m.get("page_number", 0)))

    def get_entities_for_page(self, page_id: str) -> list[dict]:
        """Return all entities recorded for a specific page."""
        mentions = self._query_all(
            f"SELECT * FROM {_LAKE}.entity_mentions WHERE page_id = ?",
            [page_id],
        )
        # Enrich each mention with the entity record (avoids JOIN crash)
        results = []
        for m in mentions:
            entity = self._query_one(
                f"SELECT * FROM {_LAKE}.entities WHERE id = ?", [m.get("entity_id", "")]
            )
            if entity:
                entity["context_text"] = m.get("context_text", "")
                entity["mentioned_at"] = m.get("created_at")
                results.append(entity)
        return sorted(results, key=lambda e: e.get("name", ""))

    # ------------------------------------------------------------------
    # Page studies
    # ------------------------------------------------------------------

    def log_page_study(self, page_id: str, document_id: str, wiki_article_path: str = "") -> str:
        """Record that a page has been studied by the swarm.

        Args:
            page_id: UUID of the page (references pages.id).
            document_id: UUID of the parent document.
            wiki_article_path: Path of the wiki article produced, if any.

        Returns:
            The UUID of the new study record.
        """
        study_id = self._new_id()
        self._exec(
            f"""
            INSERT INTO {_LAKE}.page_studies
                (id, page_id, document_id, studied_at, wiki_article_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            [study_id, page_id, document_id, _now(), wiki_article_path],
        )
        log.info("Logged study: page_id=%s → %s", page_id, wiki_article_path or "(no path)")
        return study_id

    def get_next_unstudied_page(self) -> dict | None:
        """Return the next page that has never been studied.

        Pages are ordered by document then page number.
        Uses Python-level set difference to avoid JOIN (DuckLake JOIN crash on ARM).
        """
        with self._lock:
            studied_cur = self.conn.execute(f"SELECT DISTINCT page_id FROM {_LAKE}.page_studies")
            studied_ids = {row[0] for row in studied_cur.fetchall()}

            pages_cur = self.conn.execute(
                f"SELECT * FROM {_LAKE}.pages ORDER BY document_id, page_number"
            )
            all_rows = pages_cur.fetchall()
            cols = [desc[0] for desc in pages_cur.description]

        for row in all_rows:
            page = dict(zip(cols, row))
            if page["id"] not in studied_ids:
                return page
        return None

    def get_page_study_counts(self) -> list[dict]:
        """Return all pages with their study counts and last-studied timestamp.

        Uses Python-level aggregation to avoid JOIN (DuckLake JOIN crash on ARM).
        """
        with self._lock:
            studies_cur = self.conn.execute(f"SELECT page_id, studied_at FROM {_LAKE}.page_studies")
            study_rows = studies_cur.fetchall()

        counts: dict[str, int] = defaultdict(int)
        last_studied: dict[str, str] = {}
        for page_id, studied_at in study_rows:
            counts[page_id] += 1
            if page_id not in last_studied or studied_at > last_studied[page_id]:
                last_studied[page_id] = studied_at

        pages = self._query_all(f"SELECT * FROM {_LAKE}.pages ORDER BY document_id, page_number")
        results = []
        for p in pages:
            pid = p["id"]
            results.append(
                {
                    **p,
                    "study_count": counts.get(pid, 0),
                    "last_studied": last_studied.get(pid),
                }
            )
        return sorted(
            results,
            key=lambda r: (r["study_count"], r.get("document_id", ""), r.get("page_number", 0)),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.conn.close()
        log.debug("Database connection closed")
