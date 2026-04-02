"""Tests for docswarm.agents.swarm.DocSwarm."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional
from unittest.mock import MagicMock
from unittest.mock import patch

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import ChatResult

from docswarm.agents.swarm import DocSwarm
from docswarm.agents.swarm import _is_masthead_entity
from docswarm.agents.swarm import _safe_name
from docswarm.agents.swarm import _safe_type
from docswarm.agents.tools.db_tools import create_db_tools
from docswarm.agents.tools.file_tools import create_file_read_tools
from docswarm.agents.tools.pdf_tools import create_classification_tools
from docswarm.agents.tools.pdf_tools import create_pdf_tools

# ---------------------------------------------------------------------------
# Shared mock LLM factory
# ---------------------------------------------------------------------------


class _MockLLM(BaseChatModel):
    """Minimal BaseChatModel that returns a fixed AIMessage."""

    response: str = "# Test Article\n\nContent here."

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(
        self,
        messages: List,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.response))])

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self


def _make_mock_llm(response_content: str = "# Test Article\n\nContent here.") -> _MockLLM:
    return _MockLLM(response=response_content)


def _make_mock_wiki_client():
    mock = MagicMock()
    mock.list_pages.return_value = []
    mock.search_pages.return_value = []
    return mock


def _make_swarm(tmp_config, db):
    mock_llm = _make_mock_llm()
    wiki_client = _make_mock_wiki_client()
    with patch("docswarm.agents.swarm.ChatOllama", return_value=mock_llm):
        return DocSwarm(config=tmp_config, db=db, wiki_client=wiki_client)


# ---------------------------------------------------------------------------
# Swarm init
# ---------------------------------------------------------------------------


class TestSwarmInit:
    def test_swarm_can_be_instantiated(self, tmp_config, db):
        """DocSwarm instantiation must not raise even without a real Ollama server."""
        swarm = _make_swarm(tmp_config, db)
        assert swarm is not None
        assert swarm._researcher_llm is not None
        assert swarm._writer_llm is not None
        assert swarm._classify_tool is not None

    def test_swarm_has_classify_tool(self, tmp_config, db):
        swarm = _make_swarm(tmp_config, db)
        assert swarm._classify_tool.name == "classify_page_content"


# ---------------------------------------------------------------------------
# Tool factory counts
# ---------------------------------------------------------------------------


class TestToolFactoryCounts:
    def test_tool_factory_output_counts(self, tmp_config, db):
        """Verify each tool factory produces the expected number of tools."""
        db_tools = create_db_tools(db)
        classification_tools = create_classification_tools(db, config=tmp_config)
        pdf_tools = create_pdf_tools(db)
        file_read_tools = create_file_read_tools(str(tmp_config.wiki_output_dir))

        assert len(db_tools) == 7, f"Expected 7 db_tools, got {len(db_tools)}"
        assert len(classification_tools) == 1, f"Expected 1 classification_tools, got {len(classification_tools)}"
        assert len(pdf_tools) == 2, f"Expected 2 pdf_tools, got {len(pdf_tools)}"
        assert len(file_read_tools) == 3, f"Expected 3 file_read_tools, got {len(file_read_tools)}"

    def test_classification_tool_name(self, tmp_config, db):
        classification_tools = create_classification_tools(db, config=tmp_config)
        assert classification_tools[0].name == "classify_page_content"


# ---------------------------------------------------------------------------
# Tool schema serialization
# ---------------------------------------------------------------------------


class TestToolSchemaSerialization:
    def test_all_tool_schemas_serialize_to_json(self, tmp_config, db):
        """Every tool's schema must be JSON-serialisable."""
        db_tools = create_db_tools(db)
        classification_tools = create_classification_tools(db, config=tmp_config)
        file_read_tools = create_file_read_tools(str(tmp_config.wiki_output_dir))
        all_tools = db_tools + classification_tools + file_read_tools

        schemas = []
        for t in all_tools:
            schema = t.args_schema.schema() if t.args_schema else {}
            json_str = json.dumps(schema)
            schemas.append(json_str)

        assert len(schemas) > 0
        assert all(len(s) > 2 for s in schemas)

    def test_tool_schema_keys_present(self, tmp_config, db):
        """Every tool schema should have 'title' and 'properties' keys."""
        db_tools = create_db_tools(db)
        for t in db_tools:
            schema = t.args_schema.schema() if t.args_schema else {}
            assert (
                "properties" in schema or "title" in schema
            ), f"Tool {t.name!r} schema missing expected keys: {schema}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_safe_type_canonicalises(self):
        assert _safe_type("people") == "person"
        assert _safe_type("organisations") == "organisation"
        assert _safe_type("person") == "person"
        assert _safe_type("concept") == "concept"
        assert _safe_type("") == "misc"

    def test_safe_name_slugifies(self):
        assert _safe_name("Reg Harris") == "reg-harris"
        assert _safe_name("  Café Roubaix! ") == "caf-roubaix"

    def test_masthead_entity_detected(self):
        assert _is_masthead_entity("Editor of the magazine.")
        assert _is_masthead_entity("Publisher and printer.")

    def test_masthead_not_triggered_on_long_info(self):
        long_info = "A famous editor who also happened to be " + "a word " * 20
        assert not _is_masthead_entity(long_info)

    def test_masthead_not_triggered_on_editorial(self):
        assert not _is_masthead_entity("Won the Tour de France in 1953.")


# ---------------------------------------------------------------------------
# Researcher
# ---------------------------------------------------------------------------


class TestResearch:
    def test_research_parses_json(self, tmp_config, db):
        """_research returns parsed entity list from LLM JSON output."""
        researcher_json = json.dumps({
            "entities": [
                {"entity": "Reg Harris", "type": "person", "info": "Champion cyclist.", "source": "Doc, p.1"},
            ]
        })
        mock_llm = _make_mock_llm(researcher_json)
        wiki_client = _make_mock_wiki_client()
        with patch("docswarm.agents.swarm.ChatOllama", return_value=mock_llm):
            swarm = DocSwarm(config=tmp_config, db=db, wiki_client=wiki_client)

        entities = swarm._research("Some page text", "Test Doc", "1")
        assert len(entities) == 1
        assert entities[0]["entity"] == "Reg Harris"
        assert entities[0]["type"] == "person"

    def test_research_handles_bad_json(self, tmp_config, db):
        """_research returns [] on invalid JSON."""
        mock_llm = _make_mock_llm("not valid json at all")
        wiki_client = _make_mock_wiki_client()
        with patch("docswarm.agents.swarm.ChatOllama", return_value=mock_llm):
            swarm = DocSwarm(config=tmp_config, db=db, wiki_client=wiki_client)

        entities = swarm._research("Some text", "Doc", "1")
        assert entities == []

    def test_research_handles_empty_entities(self, tmp_config, db):
        """_research returns [] when entities list is empty."""
        mock_llm = _make_mock_llm(json.dumps({"entities": []}))
        wiki_client = _make_mock_wiki_client()
        with patch("docswarm.agents.swarm.ChatOllama", return_value=mock_llm):
            swarm = DocSwarm(config=tmp_config, db=db, wiki_client=wiki_client)

        entities = swarm._research("Some text", "Doc", "1")
        assert entities == []


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class TestWriteEntity:
    def test_creates_new_article(self, tmp_config, db, sample_page):
        """_write_entity creates a new markdown file."""
        article_text = "**Reg Harris** was a British cycling champion."
        mock_llm = _make_mock_llm(article_text)
        wiki_client = _make_mock_wiki_client()
        with patch("docswarm.agents.swarm.ChatOllama", return_value=mock_llm):
            swarm = DocSwarm(config=tmp_config, db=db, wiki_client=wiki_client)

        entity = {"entity": "Reg Harris", "type": "person", "info": "Champion cyclist.", "source": "Doc, p.1"}
        path = swarm._write_entity(entity, "Test Doc", "1", sample_page["id"])

        assert path is not None
        saved = Path(path)
        assert saved.exists()
        text = saved.read_text()
        assert "Reg Harris" in text
        assert "British cycling champion" in text
        assert saved.parent.name == "person"

    def test_updates_existing_article(self, tmp_config, db, sample_page):
        """_write_entity overwrites an existing file with merged content."""
        root = Path(tmp_config.wiki_output_dir)
        existing_path = root / "person" / "reg-harris.md"
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        existing_path.write_text("---\ntitle: Reg Harris\n---\nOld content.\n")

        updated_text = "**Reg Harris** — old content plus new info about sprint titles."
        mock_llm = _make_mock_llm(updated_text)
        wiki_client = _make_mock_wiki_client()
        with patch("docswarm.agents.swarm.ChatOllama", return_value=mock_llm):
            swarm = DocSwarm(config=tmp_config, db=db, wiki_client=wiki_client)

        entity = {"entity": "Reg Harris", "type": "person", "info": "Won sprint titles.", "source": "Doc, p.5"}
        path = swarm._write_entity(entity, "Test Doc", "5", sample_page["id"])

        assert path is not None
        text = Path(path).read_text()
        assert "sprint titles" in text

    def test_skips_empty_entity(self, tmp_config, db, sample_page):
        """_write_entity returns None for entities with no name or info."""
        swarm = _make_swarm(tmp_config, db)

        assert swarm._write_entity({"entity": "", "type": "person", "info": "x", "source": "s"}, "D", "1", "p") is None
        assert swarm._write_entity({"entity": "X", "type": "person", "info": "", "source": "s"}, "D", "1", "p") is None

    def test_skips_masthead_entity(self, tmp_config, db, sample_page):
        """_write_entity returns None for masthead entities."""
        swarm = _make_swarm(tmp_config, db)

        entity = {"entity": "John Smith", "type": "person", "info": "Editor of the magazine.", "source": "Doc, p.1"}
        assert swarm._write_entity(entity, "Doc", "1", sample_page["id"]) is None

    def test_front_matter_present(self, tmp_config, db, sample_page):
        """Written article has YAML front matter."""
        article_text = "Some article body."
        mock_llm = _make_mock_llm(article_text)
        wiki_client = _make_mock_wiki_client()
        with patch("docswarm.agents.swarm.ChatOllama", return_value=mock_llm):
            swarm = DocSwarm(config=tmp_config, db=db, wiki_client=wiki_client)

        entity = {"entity": "Weinmann", "type": "organisation", "info": "Brake maker.", "source": "Doc, p.2"}
        path = swarm._write_entity(entity, "Doc", "2", sample_page["id"])

        text = Path(path).read_text()
        assert text.startswith("---\n")
        assert "title: Weinmann" in text
        assert "entity_type: organisation" in text


# ---------------------------------------------------------------------------
# run_for_page
# ---------------------------------------------------------------------------


class TestRunForPage:
    def test_skips_advertisement(self, tmp_config, db, sample_page):
        """run_for_page skips pages classified as advertisement."""
        swarm = _make_swarm(tmp_config, db)

        with patch.object(swarm, "_classify", return_value="advertisement"):
            result = swarm.run_for_page(sample_page)

        assert result["skipped"] is True
        assert db.get_next_unstudied_page() is None  # marked studied

    def test_processes_editorial_page(self, tmp_config, db, sample_page):
        """run_for_page calls researcher and writer for editorial pages."""
        entities = [
            {"entity": "Reg Harris", "type": "person", "info": "Champion.", "source": "Doc, p.1"},
        ]
        swarm = _make_swarm(tmp_config, db)

        with (
            patch.object(swarm, "_classify", return_value="editorial"),
            patch.object(swarm, "_research", return_value=entities),
            patch.object(swarm, "_write_entity", return_value="/wiki/person/reg-harris.md"),
        ):
            result = swarm.run_for_page(sample_page)

        assert result["written"] == ["/wiki/person/reg-harris.md"]
        assert db.get_next_unstudied_page() is None  # marked studied

    def test_handles_no_entities(self, tmp_config, db, sample_page):
        """run_for_page handles researcher returning no entities."""
        swarm = _make_swarm(tmp_config, db)

        with (
            patch.object(swarm, "_classify", return_value="editorial"),
            patch.object(swarm, "_research", return_value=[]),
        ):
            result = swarm.run_for_page(sample_page)

        assert result["written"] == []
        assert db.get_next_unstudied_page() is None


# ---------------------------------------------------------------------------
# run_full_wiki stops when all studied
# ---------------------------------------------------------------------------


class TestRunFullWiki:
    def test_stops_immediately_when_no_unstudied_pages(self, tmp_config, db):
        """run_full_wiki_generation returns [] when no pages to process."""
        swarm = _make_swarm(tmp_config, db)
        results = swarm.run_full_wiki_generation()
        assert results == []

    def test_processes_one_page_then_stops(self, tmp_config, db, sample_page, sample_document):
        """After the one unstudied page is processed, run_full_wiki_generation stops."""
        swarm = _make_swarm(tmp_config, db)

        with (
            patch.object(swarm, "_classify", return_value="editorial"),
            patch.object(swarm, "_research", return_value=[]),
        ):
            results = swarm.run_full_wiki_generation()

        assert len(results) == 1
