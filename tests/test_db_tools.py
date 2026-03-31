"""Tests for docswarm.agents.tools.db_tools.

Uses the real db fixture backed by a temporary DuckLake database.
"""

from __future__ import annotations

import pytest

from docswarm.agents.tools.db_tools import create_db_tools


@pytest.fixture
def db_tools(db):
    return create_db_tools(db)


@pytest.fixture
def tools_by_name(db_tools):
    return {t.name: t for t in db_tools}


# ---------------------------------------------------------------------------
# search_chunks
# ---------------------------------------------------------------------------


class TestSearchChunksTool:
    def test_returns_formatted_results(self, tools_by_name, sample_chunks):
        result = tools_by_name["search_chunks"].invoke({"query": "Chippendale"})
        assert "Chippendale" in result
        assert "REFERENCE:" in result

    def test_returns_no_results_message_when_nothing_matches(self, tools_by_name, sample_chunks):
        result = tools_by_name["search_chunks"].invoke({"query": "xyzquerynotfound99"})
        assert "No chunks found matching" in result

    def test_includes_result_count_in_header(self, tools_by_name, sample_chunks):
        result = tools_by_name["search_chunks"].invoke({"query": "antique"})
        assert "found)" in result

    def test_includes_chunk_id_in_output(self, tools_by_name, sample_chunks):
        result = tools_by_name["search_chunks"].invoke({"query": "antique"})
        assert "Chunk ID:" in result


# ---------------------------------------------------------------------------
# get_chunk
# ---------------------------------------------------------------------------


class TestGetChunkTool:
    def test_returns_formatted_chunk(self, tools_by_name, sample_chunks):
        chunk_id = sample_chunks[0]["id"]
        result = tools_by_name["get_chunk"].invoke({"chunk_id": chunk_id})
        assert "REFERENCE:" in result
        assert sample_chunks[0]["text"] in result

    def test_returns_not_found_message_for_missing_id(self, tools_by_name):
        result = tools_by_name["get_chunk"].invoke(
            {"chunk_id": "00000000-ffff-0000-0000-000000000000"}
        )
        assert "Chunk not found" in result

    def test_output_contains_page_number(self, tools_by_name, sample_chunks):
        chunk_id = sample_chunks[0]["id"]
        result = tools_by_name["get_chunk"].invoke({"chunk_id": chunk_id})
        assert "Page:" in result


# ---------------------------------------------------------------------------
# get_page_text
# ---------------------------------------------------------------------------


class TestGetPageTextTool:
    def test_concatenates_chunks_in_order(self, tools_by_name, sample_chunks, sample_document):
        result = tools_by_name["get_page_text"].invoke(
            {
                "document_id": sample_document["id"],
                "page_number": 1,
            }
        )
        # All three chunk texts should appear in the output
        for chunk in sample_chunks:
            assert chunk["text"] in result

    def test_includes_header_with_title(self, tools_by_name, sample_chunks, sample_document):
        result = tools_by_name["get_page_text"].invoke(
            {
                "document_id": sample_document["id"],
                "page_number": 1,
            }
        )
        assert "Page 1" in result
        assert "Test Document" in result

    def test_returns_no_text_message_for_unknown_page(self, tools_by_name, sample_document):
        result = tools_by_name["get_page_text"].invoke(
            {
                "document_id": sample_document["id"],
                "page_number": 999,
            }
        )
        assert "No text found" in result


# ---------------------------------------------------------------------------
# list_documents
# ---------------------------------------------------------------------------


class TestListDocumentsTool:
    def test_lists_all_documents(self, tools_by_name, sample_document):
        result = tools_by_name["list_documents"].invoke({})
        assert "test_document.pdf" in result
        assert "1 document(s)" in result

    def test_returns_no_documents_message_when_empty(self, db):
        tools = create_db_tools(db)
        tools_map = {t.name: t for t in tools}
        result = tools_map["list_documents"].invoke({})
        assert "No documents found" in result

    def test_output_includes_document_id(self, tools_by_name, sample_document):
        result = tools_by_name["list_documents"].invoke({})
        assert sample_document["id"] in result


# ---------------------------------------------------------------------------
# get_document_chunks
# ---------------------------------------------------------------------------


class TestGetDocumentChunksTool:
    def test_returns_all_chunks_for_document(self, tools_by_name, sample_chunks, sample_document):
        result = tools_by_name["get_document_chunks"].invoke(
            {
                "document_id": sample_document["id"],
            }
        )
        assert "3 chunk(s)" in result

    def test_filters_by_page_number(self, tools_by_name, sample_chunks, sample_document):
        result = tools_by_name["get_document_chunks"].invoke(
            {
                "document_id": sample_document["id"],
                "page_number": 1,
            }
        )
        assert "chunk(s)" in result
        # Should include page number in header
        assert "page 1" in result.lower()

    def test_no_chunks_returns_not_found_message(self, tools_by_name, sample_document):
        result = tools_by_name["get_document_chunks"].invoke(
            {
                "document_id": sample_document["id"],
                "page_number": 99,
            }
        )
        assert "No chunks found" in result


# ---------------------------------------------------------------------------
# get_page_study_status
# ---------------------------------------------------------------------------


class TestGetPageStudyStatusTool:
    def test_shows_unstudied_when_no_studies(self, tools_by_name, sample_page):
        result = tools_by_name["get_page_study_status"].invoke({})
        assert "UNSTUDIED" in result

    def test_shows_studied_after_study_logged(
        self, tools_by_name, sample_page, sample_document, db
    ):
        db.log_page_study(
            page_id=sample_page["id"],
            document_id=sample_document["id"],
            wiki_article_path="test/path",
        )
        result = tools_by_name["get_page_study_status"].invoke({})
        assert "studied" in result.lower()

    def test_returns_no_pages_message_when_db_empty(self, db):
        tools = create_db_tools(db)
        tools_map = {t.name: t for t in tools}
        result = tools_map["get_page_study_status"].invoke({})
        assert "No pages found" in result


# ---------------------------------------------------------------------------
# mark_page_studied
# ---------------------------------------------------------------------------


class TestMarkPageStudiedTool:
    def test_marks_page_as_studied(self, tools_by_name, sample_page, db):
        result = tools_by_name["mark_page_studied"].invoke(
            {
                "page_id": sample_page["id"],
                "wiki_article_path": "people/test-entity",
            }
        )
        assert "marked as studied" in result
        # Verify it was actually logged
        next_unstudied = db.get_next_unstudied_page()
        assert next_unstudied is None

    def test_returns_confirmation_with_article_path(self, tools_by_name, sample_page):
        result = tools_by_name["mark_page_studied"].invoke(
            {
                "page_id": sample_page["id"],
                "wiki_article_path": "people/someone",
            }
        )
        assert "people/someone" in result

    def test_returns_error_for_unknown_page_id(self, tools_by_name):
        result = tools_by_name["mark_page_studied"].invoke(
            {
                "page_id": "00000000-0000-0000-0000-deadbeefcafe",
                "wiki_article_path": "",
            }
        )
        assert "Page not found" in result

    def test_works_without_wiki_article_path(self, tools_by_name, sample_page):
        result = tools_by_name["mark_page_studied"].invoke(
            {
                "page_id": sample_page["id"],
            }
        )
        assert "marked as studied" in result


# ---------------------------------------------------------------------------
# Tool list completeness
# ---------------------------------------------------------------------------


class TestDbToolsCreation:
    def test_creates_seven_tools(self, db):
        tools = create_db_tools(db)
        assert len(tools) == 7

    def test_tool_names(self, db):
        tools = create_db_tools(db)
        names = {t.name for t in tools}
        expected = {
            "search_chunks",
            "get_chunk",
            "get_page_text",
            "list_documents",
            "get_document_chunks",
            "get_page_study_status",
            "mark_page_studied",
        }
        assert names == expected
