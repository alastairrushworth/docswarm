"""Tests for docswarm.storage.database.DatabaseManager."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------


class TestInitialize:
    def test_creates_all_six_tables(self, db):
        """initialize() must create all expected tables in the DuckLake catalog."""
        expected_tables = {
            "documents",
            "pages",
            "chunks",
            "wiki_articles",
            "entities",
            "entity_mentions",
            "page_studies",
        }
        result = db.conn.execute(
            "SELECT table_name FROM duckdb_tables() WHERE database_name = 'docswarm'"
        ).fetchall()
        found = {row[0] for row in result}
        assert expected_tables.issubset(found)

    def test_initialize_is_idempotent(self, db):
        """Calling initialize() twice must not raise an error."""
        db.initialize()  # second call
        # No exception means the CREATE TABLE IF NOT EXISTS guards worked


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class TestDocuments:
    def test_insert_and_get_document(self, db):
        doc_id = db.insert_document(
            {
                "filename": "sample.pdf",
                "filepath": "/tmp/sample.pdf",
                "title": "Sample",
                "total_pages": 5,
                "file_size_bytes": 50000,
            }
        )
        assert doc_id
        retrieved = db.get_document(doc_id)
        assert retrieved is not None
        assert retrieved["filename"] == "sample.pdf"
        assert retrieved["title"] == "Sample"
        assert retrieved["total_pages"] == 5
        assert retrieved["file_size_bytes"] == 50000

    def test_get_document_returns_none_for_missing_id(self, db):
        result = db.get_document("00000000-0000-0000-0000-000000000000")
        assert result is None

    def test_insert_preserves_provided_id(self, db):
        custom_id = "aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb"
        db.insert_document(
            {
                "id": custom_id,
                "filename": "custom.pdf",
                "filepath": "/tmp/custom.pdf",
                "title": "Custom",
                "total_pages": 1,
                "file_size_bytes": 1000,
            }
        )
        retrieved = db.get_document(custom_id)
        assert retrieved is not None
        assert retrieved["id"] == custom_id

    def test_get_document_by_filename(self, db):
        db.insert_document(
            {
                "filename": "unique_name.pdf",
                "filepath": "/tmp/unique_name.pdf",
                "title": "Unique",
                "total_pages": 2,
                "file_size_bytes": 2000,
            }
        )
        result = db.get_document_by_filename("unique_name.pdf")
        assert result is not None
        assert result["filename"] == "unique_name.pdf"

    def test_get_document_by_filename_returns_none_if_missing(self, db):
        result = db.get_document_by_filename("does_not_exist.pdf")
        assert result is None

    def test_list_documents_empty(self, db):
        docs = db.list_documents()
        assert docs == []

    def test_list_documents_returns_all(self, db):
        for i in range(3):
            db.insert_document(
                {
                    "filename": f"doc{i}.pdf",
                    "filepath": f"/tmp/doc{i}.pdf",
                    "title": f"Doc {i}",
                    "total_pages": i + 1,
                    "file_size_bytes": 1000 * (i + 1),
                }
            )
        docs = db.list_documents()
        assert len(docs) == 3

    def test_list_documents_ordered_by_ingested_at_desc(self, db):
        for i in range(3):
            db.insert_document(
                {
                    "filename": f"order{i}.pdf",
                    "filepath": f"/tmp/order{i}.pdf",
                    "title": f"Order {i}",
                    "total_pages": 1,
                    "file_size_bytes": 100,
                }
            )
        docs = db.list_documents()
        # Most recently ingested should be first
        timestamps = [str(d["ingested_at"]) for d in docs]
        assert timestamps == sorted(timestamps, reverse=True)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


class TestPages:
    def test_insert_and_get_page(self, db, sample_document):
        page = {
            "document_id": sample_document["id"],
            "page_number": 2,
            "width_pts": 595.0,
            "height_pts": 842.0,
            "image_path": "/tmp/page_0002.png",
            "raw_text": "Hello world page two",
            "ocr_confidence": 88.0,
            "word_count": 4,
        }
        page_id = db.insert_page(page)
        assert page_id
        retrieved = db.get_page(page_id)
        assert retrieved is not None
        assert retrieved["page_number"] == 2
        assert retrieved["raw_text"] == "Hello world page two"
        assert retrieved["document_id"] == sample_document["id"]

    def test_get_page_returns_none_for_missing(self, db):
        result = db.get_page("00000000-0000-0000-0000-000000000000")
        assert result is None

    def test_get_document_pages_returns_ordered(self, db, sample_document):
        for num in [3, 1, 2]:
            db.insert_page(
                {
                    "document_id": sample_document["id"],
                    "page_number": num,
                    "width_pts": 595.0,
                    "height_pts": 842.0,
                    "image_path": f"/tmp/pg{num}.png",
                    "raw_text": f"Page {num} text",
                    "ocr_confidence": 90.0,
                    "word_count": 3,
                }
            )
        pages = db.get_document_pages(sample_document["id"])
        assert len(pages) == 3
        page_numbers = [p["page_number"] for p in pages]
        assert page_numbers == sorted(page_numbers)

    def test_get_document_pages_empty_for_unknown_doc(self, db):
        pages = db.get_document_pages("nonexistent-doc-id")
        assert pages == []


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------


class TestChunks:
    def test_insert_and_get_chunk(self, db, sample_page, sample_document):
        chunk = {
            "document_id": sample_document["id"],
            "page_id": sample_page["id"],
            "page_number": 1,
            "chunk_index": 0,
            "text": "A test chunk of text.",
            "char_start": 0,
            "char_end": 21,
            "chunk_type": "text",
            "ocr_confidence": 90.0,
            "reference": '"Test Document" (p.1, chunk 1)',
            "word_count": 5,
        }
        chunk_id = db.insert_chunk(chunk)
        assert chunk_id
        retrieved = db.get_chunk(chunk_id)
        assert retrieved is not None
        assert retrieved["text"] == "A test chunk of text."
        assert retrieved["page_number"] == 1

    def test_get_chunk_returns_none_for_missing(self, db):
        result = db.get_chunk("00000000-0000-0000-0000-000000000000")
        assert result is None

    def test_get_page_chunks_ordered_by_chunk_index(self, db, sample_page, sample_document):
        for idx in [2, 0, 1]:
            db.insert_chunk(
                {
                    "document_id": sample_document["id"],
                    "page_id": sample_page["id"],
                    "page_number": 1,
                    "chunk_index": idx,
                    "text": f"chunk idx {idx}",
                    "char_start": idx * 20,
                    "char_end": idx * 20 + 10,
                    "chunk_type": "text",
                    "ocr_confidence": 90.0,
                    "reference": f'"Doc" (p.1, chunk {idx + 1})',
                    "word_count": 3,
                }
            )
        chunks = db.get_page_chunks(sample_page["id"])
        assert len(chunks) == 3
        indices = [c["chunk_index"] for c in chunks]
        assert indices == sorted(indices)

    def test_get_chunks_by_document_all(self, db, sample_document):
        """get_chunks_by_document without page_number returns all document chunks."""
        page_id_a = db.insert_page(
            {
                "document_id": sample_document["id"],
                "page_number": 1,
                "width_pts": 595.0,
                "height_pts": 842.0,
                "image_path": "/tmp/pa.png",
                "raw_text": "Page A",
                "ocr_confidence": 90.0,
                "word_count": 2,
            }
        )
        page_id_b = db.insert_page(
            {
                "document_id": sample_document["id"],
                "page_number": 2,
                "width_pts": 595.0,
                "height_pts": 842.0,
                "image_path": "/tmp/pb.png",
                "raw_text": "Page B",
                "ocr_confidence": 90.0,
                "word_count": 2,
            }
        )
        for pid, pnum in [(page_id_a, 1), (page_id_b, 2)]:
            db.insert_chunk(
                {
                    "document_id": sample_document["id"],
                    "page_id": pid,
                    "page_number": pnum,
                    "chunk_index": 0,
                    "text": f"page {pnum} chunk",
                    "char_start": 0,
                    "char_end": 10,
                    "chunk_type": "text",
                    "ocr_confidence": 90.0,
                    "reference": f'"Doc" (p.{pnum}, chunk 1)',
                    "word_count": 3,
                }
            )
        all_chunks = db.get_chunks_by_document(sample_document["id"])
        assert len(all_chunks) == 2

    def test_get_chunks_by_document_filtered_by_page_number(self, db, sample_document):
        page_id = db.insert_page(
            {
                "document_id": sample_document["id"],
                "page_number": 7,
                "width_pts": 595.0,
                "height_pts": 842.0,
                "image_path": "/tmp/p7.png",
                "raw_text": "Page 7",
                "ocr_confidence": 90.0,
                "word_count": 2,
            }
        )
        db.insert_chunk(
            {
                "document_id": sample_document["id"],
                "page_id": page_id,
                "page_number": 7,
                "chunk_index": 0,
                "text": "page seven text",
                "char_start": 0,
                "char_end": 15,
                "chunk_type": "text",
                "ocr_confidence": 90.0,
                "reference": '"Doc" (p.7, chunk 1)',
                "word_count": 3,
            }
        )
        chunks = db.get_chunks_by_document(sample_document["id"], page_number=7)
        assert len(chunks) == 1
        assert chunks[0]["page_number"] == 7

        # page 99 should return nothing
        none_chunks = db.get_chunks_by_document(sample_document["id"], page_number=99)
        assert none_chunks == []

    def test_search_chunks_finds_matching_text(self, db, sample_chunks):
        results = db.search_chunks("Chippendale")
        assert len(results) >= 1
        assert any("Chippendale" in c["text"] for c in results)

    def test_search_chunks_returns_empty_for_no_match(self, db, sample_chunks):
        results = db.search_chunks("xyznotfoundinanytext12345")
        assert results == []

    def test_search_chunks_returns_empty_for_blank_query(self, db, sample_chunks):
        results = db.search_chunks("   ")
        assert results == []

    def test_search_chunks_multi_term_and_logic(self, db, sample_chunks):
        """All terms must be present in matching chunk (AND logic)."""
        results = db.search_chunks("Chippendale cabinet-maker")
        assert len(results) >= 1
        for r in results:
            text_lower = r["text"].lower()
            assert "chippendale" in text_lower

    def test_search_chunks_respects_limit(self, db, sample_document, sample_page):
        for i in range(15):
            db.insert_chunk(
                {
                    "document_id": sample_document["id"],
                    "page_id": sample_page["id"],
                    "page_number": 1,
                    "chunk_index": i + 10,
                    "text": f"repeated keyword searchterm chunk number {i}",
                    "char_start": i * 50,
                    "char_end": i * 50 + 40,
                    "chunk_type": "text",
                    "ocr_confidence": 90.0,
                    "reference": f'"Doc" (p.1, chunk {i + 11})',
                    "word_count": 6,
                }
            )
        results = db.search_chunks("searchterm", limit=5)
        assert len(results) <= 5


# ---------------------------------------------------------------------------
# Wiki articles
# ---------------------------------------------------------------------------


class TestWikiArticles:
    def test_upsert_wiki_article_insert(self, db):
        article = {
            "wiki_page_id": 42,
            "title": "Thomas Chippendale",
            "path": "people/thomas-chippendale",
            "content": "# Thomas Chippendale\n\nEnglish cabinet-maker.",
            "source_chunk_ids": [],
            "source_document_ids": [],
        }
        article_id = db.upsert_wiki_article(article)
        assert article_id

        articles = db.get_wiki_articles()
        assert len(articles) == 1
        assert articles[0]["title"] == "Thomas Chippendale"
        assert articles[0]["path"] == "people/thomas-chippendale"

    def test_upsert_wiki_article_update(self, db):
        """Calling upsert with the same path must update, not insert a duplicate."""
        article = {
            "title": "Initial Title",
            "path": "test/update-path",
            "content": "Original content.",
        }
        first_id = db.upsert_wiki_article(article)

        updated = {
            "title": "Updated Title",
            "path": "test/update-path",
            "content": "Updated content with more detail.",
        }
        second_id = db.upsert_wiki_article(updated)

        assert first_id == second_id

        articles = db.get_wiki_articles()
        assert len(articles) == 1
        assert articles[0]["title"] == "Updated Title"
        assert articles[0]["content"] == "Updated content with more detail."

    def test_upsert_wiki_article_different_paths_create_new(self, db):
        db.upsert_wiki_article({"title": "A", "path": "path/a", "content": "A"})
        db.upsert_wiki_article({"title": "B", "path": "path/b", "content": "B"})
        articles = db.get_wiki_articles()
        assert len(articles) == 2


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


class TestEntities:
    def test_upsert_entity_creates_new(self, db):
        entity_id = db.upsert_entity("Thomas Chippendale", "person")
        assert entity_id

        entity = db.get_entity_by_name("Thomas Chippendale")
        assert entity is not None
        assert entity["name"] == "Thomas Chippendale"
        assert entity["entity_type"] == "person"

    def test_upsert_entity_returns_existing_id(self, db):
        id1 = db.upsert_entity("London", "place")
        id2 = db.upsert_entity("London", "place")
        assert id1 == id2

    def test_get_entity_by_name_returns_none_for_missing(self, db):
        result = db.get_entity_by_name("Nobody Noperson")
        assert result is None

    def test_add_entity_mention(self, db, sample_page, sample_document):
        entity_id = db.upsert_entity("Windsor Castle", "place")
        mention_id = db.add_entity_mention(
            entity_id=entity_id,
            page_id=sample_page["id"],
            document_id=sample_document["id"],
            context_text="Windsor Castle is mentioned in this passage.",
        )
        assert mention_id

    def test_search_entities_by_partial_name(self, db):
        db.upsert_entity("Thomas Chippendale", "person")
        db.upsert_entity("Thomas Sheraton", "person")
        db.upsert_entity("Windsor Castle", "place")

        results = db.search_entities("Thomas")
        names = [r["name"] for r in results]
        assert "Thomas Chippendale" in names
        assert "Thomas Sheraton" in names
        assert "Windsor Castle" not in names

    def test_search_entities_case_insensitive(self, db):
        db.upsert_entity("Victoria and Albert Museum", "place")
        results = db.search_entities("victoria")
        assert len(results) >= 1

    def test_search_entities_returns_empty_for_no_match(self, db):
        db.upsert_entity("Thomas Chippendale", "person")
        results = db.search_entities("xyznotanentity99")
        assert results == []

    def test_get_entity_mentions_returns_all(self, db, sample_page, sample_document):
        # Insert a second page so we can add two mentions
        page2_id = db.insert_page(
            {
                "document_id": sample_document["id"],
                "page_number": 2,
                "width_pts": 595.0,
                "height_pts": 842.0,
                "image_path": "/tmp/p2.png",
                "raw_text": "Another page about furniture.",
                "ocr_confidence": 90.0,
                "word_count": 5,
            }
        )
        entity_id = db.upsert_entity("Chippendale Style", "concept")
        db.add_entity_mention(entity_id, sample_page["id"], sample_document["id"], "First mention")
        db.add_entity_mention(entity_id, page2_id, sample_document["id"], "Second mention")

        mentions = db.get_entity_mentions(entity_id)
        assert len(mentions) == 2
        contexts = [m["context_text"] for m in mentions]
        assert "First mention" in contexts
        assert "Second mention" in contexts

    def test_get_entities_for_page_returns_correct_entities(self, db, sample_page, sample_document):
        id1 = db.upsert_entity("Mahogany", "object")
        id2 = db.upsert_entity("Georgian Period", "event")
        db.add_entity_mention(
            id1, sample_page["id"], sample_document["id"], "Mahogany is a hardwood"
        )
        db.add_entity_mention(
            id2, sample_page["id"], sample_document["id"], "Georgian period 1714-1830"
        )

        entities = db.get_entities_for_page(sample_page["id"])
        names = [e["name"] for e in entities]
        assert "Mahogany" in names
        assert "Georgian Period" in names

    def test_get_entities_for_page_empty_when_none(self, db, sample_page):
        entities = db.get_entities_for_page(sample_page["id"])
        assert entities == []

    def test_get_entities_for_page_sorted_by_name(self, db, sample_page, sample_document):
        for name in ["Zebra Ware", "Apple Design", "Mahogany Chair"]:
            eid = db.upsert_entity(name, "object")
            db.add_entity_mention(
                eid, sample_page["id"], sample_document["id"], f"context for {name}"
            )
        entities = db.get_entities_for_page(sample_page["id"])
        names = [e["name"] for e in entities]
        assert names == sorted(names)

    def test_get_entity_mentions_sorted_by_document_and_page(self, db, sample_document):
        page1_id = db.insert_page(
            {
                "document_id": sample_document["id"],
                "page_number": 1,
                "width_pts": 595.0,
                "height_pts": 842.0,
                "image_path": "/tmp/p1.png",
                "raw_text": "p1",
                "ocr_confidence": 90.0,
                "word_count": 1,
            }
        )
        page3_id = db.insert_page(
            {
                "document_id": sample_document["id"],
                "page_number": 3,
                "width_pts": 595.0,
                "height_pts": 842.0,
                "image_path": "/tmp/p3.png",
                "raw_text": "p3",
                "ocr_confidence": 90.0,
                "word_count": 1,
            }
        )
        entity_id = db.upsert_entity("Roving Entity", "concept")
        db.add_entity_mention(entity_id, page3_id, sample_document["id"], "on page 3")
        db.add_entity_mention(entity_id, page1_id, sample_document["id"], "on page 1")

        mentions = db.get_entity_mentions(entity_id)
        assert len(mentions) == 2
        page_numbers = [m["page_number"] for m in mentions]
        assert page_numbers == sorted(page_numbers)


# ---------------------------------------------------------------------------
# Page studies
# ---------------------------------------------------------------------------


class TestPageStudies:
    def test_log_page_study_creates_record(self, db, sample_page, sample_document):
        study_id = db.log_page_study(
            page_id=sample_page["id"],
            document_id=sample_document["id"],
            wiki_article_path="people/thomas-chippendale",
        )
        assert study_id

    def test_get_next_unstudied_page_returns_first_page(self, db, sample_page):
        result = db.get_next_unstudied_page()
        assert result is not None
        assert result["id"] == sample_page["id"]

    def test_get_next_unstudied_page_returns_none_when_all_studied(
        self, db, sample_page, sample_document
    ):
        db.log_page_study(
            page_id=sample_page["id"],
            document_id=sample_document["id"],
        )
        result = db.get_next_unstudied_page()
        assert result is None

    def test_get_next_unstudied_page_skips_studied_pages(self, db, sample_document):
        """When page 1 is studied, get_next_unstudied_page should return page 2."""
        page1_id = db.insert_page(
            {
                "document_id": sample_document["id"],
                "page_number": 1,
                "width_pts": 595.0,
                "height_pts": 842.0,
                "image_path": "/tmp/p1.png",
                "raw_text": "page 1",
                "ocr_confidence": 90.0,
                "word_count": 2,
            }
        )
        page2_id = db.insert_page(
            {
                "document_id": sample_document["id"],
                "page_number": 2,
                "width_pts": 595.0,
                "height_pts": 842.0,
                "image_path": "/tmp/p2.png",
                "raw_text": "page 2",
                "ocr_confidence": 90.0,
                "word_count": 2,
            }
        )
        db.log_page_study(page_id=page1_id, document_id=sample_document["id"])
        result = db.get_next_unstudied_page()
        assert result is not None
        assert result["id"] == page2_id

    def test_get_next_unstudied_page_respects_document_page_number_order(self, db, sample_document):
        """Pages are returned in (document_id, page_number) order."""
        # Insert pages out of order
        for num in [3, 1, 2]:
            db.insert_page(
                {
                    "document_id": sample_document["id"],
                    "page_number": num,
                    "width_pts": 595.0,
                    "height_pts": 842.0,
                    "image_path": f"/tmp/p{num}.png",
                    "raw_text": f"page {num}",
                    "ocr_confidence": 90.0,
                    "word_count": 2,
                }
            )
        first = db.get_next_unstudied_page()
        assert first is not None
        assert first["page_number"] == 1

    def test_get_page_study_counts_includes_all_pages(self, db, sample_document):
        page1_id = db.insert_page(
            {
                "document_id": sample_document["id"],
                "page_number": 1,
                "width_pts": 595.0,
                "height_pts": 842.0,
                "image_path": "/tmp/p1.png",
                "raw_text": "p1",
                "ocr_confidence": 90.0,
                "word_count": 1,
            }
        )
        page2_id = db.insert_page(
            {
                "document_id": sample_document["id"],
                "page_number": 2,
                "width_pts": 595.0,
                "height_pts": 842.0,
                "image_path": "/tmp/p2.png",
                "raw_text": "p2",
                "ocr_confidence": 90.0,
                "word_count": 1,
            }
        )
        db.log_page_study(page1_id, sample_document["id"])

        counts = db.get_page_study_counts()
        assert len(counts) == 2

        by_id = {r["id"]: r for r in counts}
        assert by_id[page1_id]["study_count"] == 1
        assert by_id[page2_id]["study_count"] == 0

    def test_get_page_study_counts_ordered_by_study_count_asc(self, db, sample_document):
        """Least-studied pages appear first."""
        page1_id = db.insert_page(
            {
                "document_id": sample_document["id"],
                "page_number": 1,
                "width_pts": 595.0,
                "height_pts": 842.0,
                "image_path": "/tmp/p1.png",
                "raw_text": "p1",
                "ocr_confidence": 90.0,
                "word_count": 1,
            }
        )
        db.insert_page(
            {
                "document_id": sample_document["id"],
                "page_number": 2,
                "width_pts": 595.0,
                "height_pts": 842.0,
                "image_path": "/tmp/p2.png",
                "raw_text": "p2",
                "ocr_confidence": 90.0,
                "word_count": 1,
            }
        )
        db.log_page_study(page1_id, sample_document["id"])
        db.log_page_study(page1_id, sample_document["id"])

        counts = db.get_page_study_counts()
        study_counts = [r["study_count"] for r in counts]
        assert study_counts == sorted(study_counts)
