"""Tests for DatabaseManager thread safety.

LangGraph's ToolNode dispatches tool calls to a ThreadPoolExecutor,
so multiple agent tools hit the database concurrently.  DuckDB
connections are not thread-safe — without locking this causes SIGSEGV
on macOS ARM.  These tests verify the lock prevents crashes.
"""

from __future__ import annotations

import concurrent.futures
import threading


class TestConcurrentReads:
    def test_parallel_search_chunks_does_not_crash(self, db, sample_chunks):
        """Simulate LangGraph dispatching multiple search_chunks calls in parallel."""
        queries = ["paragraph", "furniture", "Chippendale", "test", "information"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(db.search_chunks, q) for q in queries]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 5
        assert all(isinstance(r, list) for r in results)

    def test_parallel_search_entities_does_not_crash(self, db, sample_chunks):
        """Simulate multiple search_entities calls in parallel."""
        queries = ["Thomas", "Chippendale", "England", "furniture", "nothing"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(db.search_entities, q) for q in queries]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 5

    def test_mixed_read_operations_in_parallel(self, db, sample_chunks, sample_document):
        """Mix of different read operations running concurrently."""

        def read_ops():
            db.search_chunks("test")
            db.list_documents()
            db.search_entities("nothing")
            db.get_page_study_counts()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(read_ops) for _ in range(4)]
            for f in concurrent.futures.as_completed(futures):
                f.result()  # raises if any thread crashed


class TestConcurrentWrites:
    def test_parallel_entity_inserts_do_not_crash(self, db, sample_page):
        """Multiple threads inserting entities simultaneously."""

        def insert_entity(name):
            eid = db.upsert_entity(name=name, entity_type="person")
            db.add_entity_mention(
                entity_id=eid,
                page_id=sample_page["id"],
                document_id=sample_page["document_id"],
                context_text=f"Mention of {name}",
            )
            return eid

        names = [f"Person_{i}" for i in range(10)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(insert_entity, n) for n in names]
            ids = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(ids) == 10
        assert all(isinstance(eid, str) for eid in ids)


class TestConcurrentMixedReadWrite:
    def test_reads_and_writes_interleaved(self, db, sample_page, sample_chunks):
        """Concurrent reads and writes — the exact pattern that caused the segfault."""
        barrier = threading.Barrier(3)

        def writer():
            barrier.wait()
            for i in range(5):
                db.upsert_entity(name=f"Writer_{i}", entity_type="concept")

        def reader_chunks():
            barrier.wait()
            for _ in range(5):
                db.search_chunks("test")

        def reader_entities():
            barrier.wait()
            for _ in range(5):
                db.search_entities("Writer")

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(writer),
                pool.submit(reader_chunks),
                pool.submit(reader_entities),
            ]
            for f in concurrent.futures.as_completed(futures):
                f.result()
