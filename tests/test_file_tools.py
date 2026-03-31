"""Tests for docswarm.agents.tools.file_tools.

Uses real tmp_path – no mocking needed for filesystem operations.
"""

from __future__ import annotations

import pytest

from docswarm.agents.tools.file_tools import (
    _build_front_matter,
    _parse_front_matter,
    create_file_read_tools,
    create_file_tools,
)


@pytest.fixture
def wiki_dir(tmp_path):
    d = tmp_path / "wiki"
    d.mkdir()
    return d


@pytest.fixture
def file_tools(wiki_dir):
    return create_file_tools(str(wiki_dir))


@pytest.fixture
def write_tool(file_tools):
    return file_tools[0]  # write_article_file


@pytest.fixture
def read_tool(file_tools):
    return file_tools[1]  # read_article_file


@pytest.fixture
def list_tool(file_tools):
    return file_tools[2]  # list_article_files


@pytest.fixture
def search_tool(file_tools):
    return file_tools[3]  # search_article_files


# ---------------------------------------------------------------------------
# write_article_file
# ---------------------------------------------------------------------------


class TestWriteArticleFile:
    def test_creates_file_at_correct_path(self, write_tool, wiki_dir):
        result = write_tool.invoke(
            {
                "path": "people/thomas-chippendale",
                "title": "Thomas Chippendale",
                "content": "# Thomas Chippendale\n\nA famous cabinet-maker.",
            }
        )
        expected = wiki_dir / "people" / "thomas-chippendale.md"
        assert expected.exists()
        assert str(expected) in result

    def test_creates_parent_directories(self, write_tool, wiki_dir):
        write_tool.invoke(
            {
                "path": "events/great-exhibition/1851",
                "title": "Great Exhibition 1851",
                "content": "Article content here.",
            }
        )
        expected = wiki_dir / "events" / "great-exhibition" / "1851.md"
        assert expected.exists()

    def test_front_matter_contains_title(self, write_tool, wiki_dir):
        write_tool.invoke(
            {
                "path": "objects/mahogany-chair",
                "title": "Mahogany Chair",
                "content": "Content.",
            }
        )
        text = (wiki_dir / "objects" / "mahogany-chair.md").read_text()
        meta, _ = _parse_front_matter(text)
        assert meta["title"] == "Mahogany Chair"

    def test_front_matter_contains_description(self, write_tool, wiki_dir):
        write_tool.invoke(
            {
                "path": "objects/oak-table",
                "title": "Oak Table",
                "content": "Content.",
                "description": "A sturdy oak dining table.",
            }
        )
        text = (wiki_dir / "objects" / "oak-table.md").read_text()
        meta, _ = _parse_front_matter(text)
        assert meta["description"] == "A sturdy oak dining table."

    def test_front_matter_contains_entity_id(self, write_tool, wiki_dir):
        write_tool.invoke(
            {
                "path": "people/test-person",
                "title": "Test Person",
                "content": "Content.",
                "entity_id": "eid-12345",
            }
        )
        text = (wiki_dir / "people" / "test-person.md").read_text()
        meta, _ = _parse_front_matter(text)
        assert meta["entity_id"] == "eid-12345"

    def test_front_matter_contains_entity_type(self, write_tool, wiki_dir):
        write_tool.invoke(
            {
                "path": "people/maker",
                "title": "The Maker",
                "content": "Content.",
                "entity_type": "person",
            }
        )
        text = (wiki_dir / "people" / "maker.md").read_text()
        meta, _ = _parse_front_matter(text)
        assert meta["entity_type"] == "person"

    def test_front_matter_contains_source_page_id(self, write_tool, wiki_dir):
        write_tool.invoke(
            {
                "path": "people/src",
                "title": "Source",
                "content": "Content.",
                "source_page_id": "pid-99",
            }
        )
        text = (wiki_dir / "people" / "src.md").read_text()
        meta, _ = _parse_front_matter(text)
        assert meta["source_page_id"] == "pid-99"

    def test_front_matter_wiki_page_id_is_none_by_default(self, write_tool, wiki_dir):
        write_tool.invoke(
            {
                "path": "places/london",
                "title": "London",
                "content": "Content.",
            }
        )
        text = (wiki_dir / "places" / "london.md").read_text()
        meta, _ = _parse_front_matter(text)
        assert meta["wiki_page_id"] is None

    def test_spaces_in_path_converted_to_hyphens(self, write_tool, wiki_dir):
        write_tool.invoke(
            {
                "path": "people/thomas chippendale",
                "title": "Thomas Chippendale",
                "content": "Content.",
            }
        )
        expected = wiki_dir / "people" / "thomas-chippendale.md"
        assert expected.exists()

    def test_uppercase_path_converted_to_lowercase(self, write_tool, wiki_dir):
        write_tool.invoke(
            {
                "path": "People/Thomas-Chippendale",
                "title": "Thomas Chippendale",
                "content": "Content.",
            }
        )
        expected = wiki_dir / "people" / "thomas-chippendale.md"
        assert expected.exists()

    def test_overwrites_existing_file(self, write_tool, wiki_dir):
        write_tool.invoke({"path": "test/overwrite", "title": "T", "content": "First."})
        write_tool.invoke({"path": "test/overwrite", "title": "T", "content": "Second."})
        text = (wiki_dir / "test" / "overwrite.md").read_text()
        assert "Second." in text
        assert "First." not in text

    def test_return_value_contains_file_path(self, write_tool, wiki_dir):
        result = write_tool.invoke(
            {
                "path": "places/paris",
                "title": "Paris",
                "content": "Content.",
            }
        )
        assert "Article written to:" in result
        assert "paris.md" in result


# ---------------------------------------------------------------------------
# read_article_file
# ---------------------------------------------------------------------------


class TestReadArticleFile:
    def test_returns_file_contents_with_front_matter(self, write_tool, read_tool, wiki_dir):
        write_tool.invoke(
            {
                "path": "people/read-test",
                "title": "Read Test",
                "content": "Article body here.",
                "description": "A test article.",
            }
        )
        result = read_tool.invoke({"path": "people/read-test"})
        assert "Read Test" in result
        assert "A test article." in result
        assert "Article body here." in result

    def test_returns_error_message_for_missing_file(self, read_tool, wiki_dir):
        result = read_tool.invoke({"path": "people/nobody-here"})
        assert "No article file found" in result

    def test_path_normalisation_applies_on_read(self, write_tool, read_tool, wiki_dir):
        write_tool.invoke(
            {
                "path": "places/new-york",
                "title": "New York",
                "content": "The city.",
            }
        )
        # Reading with spaces and uppercase should still find the file
        result = read_tool.invoke({"path": "Places/New York"})
        assert "New York" in result

    def test_returns_entity_id_in_output(self, write_tool, read_tool, wiki_dir):
        write_tool.invoke(
            {
                "path": "people/e-id-test",
                "title": "Entity ID Test",
                "content": "Content.",
                "entity_id": "abc-entity-123",
            }
        )
        result = read_tool.invoke({"path": "people/e-id-test"})
        assert "abc-entity-123" in result


# ---------------------------------------------------------------------------
# list_article_files
# ---------------------------------------------------------------------------


class TestListArticleFiles:
    def test_lists_all_md_files(self, write_tool, list_tool, wiki_dir):
        for i in range(3):
            write_tool.invoke(
                {
                    "path": f"test/article-{i}",
                    "title": f"Article {i}",
                    "content": "Content.",
                }
            )
        result = list_tool.invoke({})
        assert "3 article(s)" in result

    def test_returns_no_files_message_when_empty(self, list_tool, wiki_dir):
        result = list_tool.invoke({})
        assert "No article files found" in result

    def test_lists_nested_articles(self, write_tool, list_tool, wiki_dir):
        write_tool.invoke({"path": "a/b/c/deep-article", "title": "Deep", "content": "x"})
        result = list_tool.invoke({})
        assert "deep-article" in result


# ---------------------------------------------------------------------------
# search_article_files
# ---------------------------------------------------------------------------


class TestSearchArticleFiles:
    def test_finds_files_by_body_content(self, write_tool, search_tool, wiki_dir):
        write_tool.invoke(
            {
                "path": "people/cabinet-maker",
                "title": "Cabinet Maker",
                "content": "Thomas Chippendale was born in 1718 in Otley.",
            }
        )
        result = search_tool.invoke({"query": "Chippendale"})
        assert "cabinet-maker" in result
        assert "Cabinet Maker" in result

    def test_finds_files_by_title(self, write_tool, search_tool, wiki_dir):
        write_tool.invoke(
            {
                "path": "places/bath",
                "title": "City of Bath",
                "content": "Some unrelated content.",
            }
        )
        result = search_tool.invoke({"query": "City of Bath"})
        assert "bath" in result

    def test_search_is_case_insensitive(self, write_tool, search_tool, wiki_dir):
        write_tool.invoke(
            {
                "path": "objects/walnut",
                "title": "Walnut Wood",
                "content": "Walnut was prized by furniture makers.",
            }
        )
        result = search_tool.invoke({"query": "WALNUT"})
        assert "walnut" in result.lower()

    def test_returns_no_match_message_when_not_found(self, write_tool, search_tool, wiki_dir):
        write_tool.invoke(
            {
                "path": "test/placeholder",
                "title": "Placeholder",
                "content": "Some content here.",
            }
        )
        result = search_tool.invoke({"query": "xyznotfoundatall99"})
        assert "No articles found matching" in result

    def test_returns_no_match_when_directory_empty(self, search_tool, wiki_dir):
        result = search_tool.invoke({"query": "anything"})
        assert "No articles found matching" in result


# ---------------------------------------------------------------------------
# create_file_read_tools – read-only subset
# ---------------------------------------------------------------------------


class TestCreateFileReadTools:
    def test_returns_three_read_only_tools(self, wiki_dir):
        tools = create_file_read_tools(str(wiki_dir))
        assert len(tools) == 3

    def test_does_not_include_write_tool(self, wiki_dir):
        tools = create_file_read_tools(str(wiki_dir))
        tool_names = [t.name for t in tools]
        assert "write_article_file" not in tool_names

    def test_includes_read_list_search(self, wiki_dir):
        tools = create_file_read_tools(str(wiki_dir))
        tool_names = [t.name for t in tools]
        assert "read_article_file" in tool_names
        assert "list_article_files" in tool_names
        assert "search_article_files" in tool_names


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestParseFrontMatter:
    def test_parses_yaml_front_matter(self):
        text = "---\ntitle: Hello\nfoo: bar\n---\n\nBody text."
        meta, body = _parse_front_matter(text)
        assert meta["title"] == "Hello"
        assert meta["foo"] == "bar"
        assert body.strip() == "Body text."

    def test_returns_empty_dict_when_no_front_matter(self):
        text = "Just body text with no front matter."
        meta, body = _parse_front_matter(text)
        assert meta == {}
        assert body == text

    def test_handles_malformed_yaml_gracefully(self):
        text = "---\n: invalid: yaml: here\n---\n\nBody."
        meta, body = _parse_front_matter(text)
        assert isinstance(meta, dict)  # either empty or partially parsed, no crash


class TestBuildFrontMatter:
    def test_starts_with_triple_dash(self):
        result = _build_front_matter({"title": "Test"})
        assert result.startswith("---\n")

    def test_ends_with_triple_dash_and_blank_line(self):
        result = _build_front_matter({"title": "Test"})
        assert result.endswith("---\n\n")

    def test_contains_key_value(self):
        result = _build_front_matter({"title": "My Title"})
        assert "title: My Title" in result
