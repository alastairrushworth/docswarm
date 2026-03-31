"""Tests for docswarm.agents.swarm.DocSwarm."""

from __future__ import annotations

import json
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from docswarm.agents.swarm import DocSwarm
from docswarm.agents.tools.db_tools import create_db_tools
from docswarm.agents.tools.entity_tools import create_entity_tools
from docswarm.agents.tools.file_tools import create_file_read_tools
from docswarm.agents.tools.pdf_tools import create_pdf_tools

# ---------------------------------------------------------------------------
# Shared mock LLM factory
# ---------------------------------------------------------------------------


class _MockLLM(BaseChatModel):
    """Minimal BaseChatModel that returns a fixed AIMessage and supports bind_tools."""

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
    """Return a BaseChatModel subclass that returns a fixed AIMessage.

    Using a proper subclass (not MagicMock) so that LangGraph's
    create_react_agent can use bind_tools, invoke, and stream without errors.
    """
    return _MockLLM(response=response_content)


def _make_mock_wiki_client():
    mock = MagicMock()
    mock.list_pages.return_value = []
    mock.search_pages.return_value = []
    return mock


# ---------------------------------------------------------------------------
# Swarm init
# ---------------------------------------------------------------------------


class TestSwarmInit:
    def test_swarm_can_be_instantiated(self, tmp_config, db):
        """DocSwarm instantiation must not raise even without a real Ollama server."""
        mock_llm = _make_mock_llm()
        wiki_client = _make_mock_wiki_client()

        with patch("docswarm.agents.swarm.ChatOllama", return_value=mock_llm):
            swarm = DocSwarm(config=tmp_config, db=db, wiki_client=wiki_client)

        assert swarm is not None
        assert swarm._graph is not None

    def test_swarm_builds_graph_with_four_nodes(self, tmp_config, db):
        mock_llm = _make_mock_llm()
        wiki_client = _make_mock_wiki_client()

        with patch("docswarm.agents.swarm.ChatOllama", return_value=mock_llm):
            swarm = DocSwarm(config=tmp_config, db=db, wiki_client=wiki_client)

        # The compiled graph has nodes for researcher, writer, reviewer, editor
        graph_nodes = swarm._graph.get_graph().nodes
        node_names = set(graph_nodes.keys())
        for expected in ("researcher", "writer", "reviewer", "editor"):
            assert expected in node_names


# ---------------------------------------------------------------------------
# Tool factory counts
# ---------------------------------------------------------------------------


class TestToolFactoryCounts:
    def test_tool_factory_output_counts(self, tmp_config, db):
        """Verify each tool factory produces the expected number of tools."""
        db_tools = create_db_tools(db)
        entity_tools = create_entity_tools(db)
        pdf_tools = create_pdf_tools(db)
        file_read_tools = create_file_read_tools(str(tmp_config.wiki_output_dir))

        assert len(db_tools) == 7, f"Expected 7 db_tools, got {len(db_tools)}"
        assert len(entity_tools) == 4, f"Expected 4 entity_tools, got {len(entity_tools)}"
        assert len(pdf_tools) == 2, f"Expected 2 pdf_tools, got {len(pdf_tools)}"
        assert len(file_read_tools) == 3, f"Expected 3 file_read_tools, got {len(file_read_tools)}"

    def test_tool_names(self, tmp_config, db):
        db_tools = create_db_tools(db)
        entity_tools = create_entity_tools(db)
        pdf_tools = create_pdf_tools(db)
        file_read_tools = create_file_read_tools(str(tmp_config.wiki_output_dir))

        all_tools = db_tools + entity_tools + pdf_tools + file_read_tools
        names = {t.name for t in all_tools}

        expected_names = {
            # db_tools (7)
            "search_chunks",
            "get_chunk",
            "get_page_text",
            "list_documents",
            "get_document_chunks",
            "get_page_study_status",
            "mark_page_studied",
            # entity_tools (4)
            "save_entity",
            "search_entities",
            "get_entities_for_page",
            "get_entity_mentions",
            # pdf_tools (2)
            "get_page_image_info",
            "get_source_reference",
            # file_read_tools (3)
            "read_article_file",
            "list_article_files",
            "search_article_files",
        }
        assert names == expected_names


# ---------------------------------------------------------------------------
# Tool schema serialization
# ---------------------------------------------------------------------------


class TestToolSchemaSerialization:
    def test_all_researcher_tool_schemas_serialize_to_json(self, tmp_config, db):
        """Every tool's schema must be JSON-serialisable."""
        db_tools = create_db_tools(db)
        entity_tools = create_entity_tools(db)
        pdf_tools = create_pdf_tools(db)
        file_read_tools = create_file_read_tools(str(tmp_config.wiki_output_dir))
        all_tools = db_tools + entity_tools + pdf_tools + file_read_tools

        schemas = []
        for t in all_tools:
            schema = t.args_schema.schema() if t.args_schema else {}
            # Must not raise
            json_str = json.dumps(schema)
            schemas.append(json_str)

        total_bytes = sum(len(s.encode("utf-8")) for s in schemas)
        combined = json.dumps([json.loads(s) for s in schemas])
        combined_bytes = len(combined.encode("utf-8"))

        # Log sizes for diagnostic purposes (visible with pytest -s)
        print(f"\n[schema_size] Total across {len(all_tools)} tools: {total_bytes} bytes")
        print(f"[schema_size] Combined JSON: {combined_bytes} bytes")

        # All schemas must be valid JSON (no assertion on size — we are measuring)
        assert len(schemas) > 0
        assert all(len(s) > 2 for s in schemas)  # each schema is non-trivial

    def test_tool_schema_keys_present(self, tmp_config, db):
        """Every tool schema should have 'title' and 'properties' keys."""
        db_tools = create_db_tools(db)
        for t in db_tools:
            schema = t.args_schema.schema() if t.args_schema else {}
            assert (
                "properties" in schema or "title" in schema
            ), f"Tool {t.name!r} schema missing expected keys: {schema}"


# ---------------------------------------------------------------------------
# Researcher invoke with mock LLM
# ---------------------------------------------------------------------------


class TestResearcherInvokeWithMockLLM:
    def test_invoke_does_not_crash_with_short_message(self, tmp_config, db):
        """Researcher invoke works with a short message and mocked LLM."""
        from langgraph.prebuilt import create_react_agent

        from docswarm.agents.personas import RESEARCHER_PROMPT

        db_tools = create_db_tools(db)
        entity_tools = create_entity_tools(db)
        pdf_tools = create_pdf_tools(db)
        file_read_tools = create_file_read_tools(str(tmp_config.wiki_output_dir))
        research_tools = db_tools + entity_tools + pdf_tools + file_read_tools

        mock_llm = _make_mock_llm()

        # create_react_agent calls bind_tools internally
        researcher = create_react_agent(mock_llm, tools=research_tools, prompt=RESEARCHER_PROMPT)

        initial_message = HumanMessage(content="Research the topic: Victorian furniture.")
        result = researcher.invoke({"messages": [initial_message]})
        assert "messages" in result
        assert len(result["messages"]) > 0

    def test_last_message_is_ai_message(self, tmp_config, db):
        from langgraph.prebuilt import create_react_agent

        from docswarm.agents.personas import RESEARCHER_PROMPT

        tools = create_db_tools(db) + create_entity_tools(db)
        mock_llm = _make_mock_llm()
        researcher = create_react_agent(mock_llm, tools=tools, prompt=RESEARCHER_PROMPT)

        result = researcher.invoke({"messages": [HumanMessage(content="Test")]})
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_messages) >= 1


# ---------------------------------------------------------------------------
# Researcher invoke with all tools
# ---------------------------------------------------------------------------


class TestResearcherInvokeWithAllTools:
    def test_create_react_agent_accepts_all_tools(self, tmp_config, db):
        """Verify create_react_agent works with the full set of available tools."""
        from langgraph.prebuilt import create_react_agent

        from docswarm.agents.personas import RESEARCHER_PROMPT

        db_tools = create_db_tools(db)
        entity_tools = create_entity_tools(db)
        pdf_tools = create_pdf_tools(db)
        file_read_tools = create_file_read_tools(str(tmp_config.wiki_output_dir))
        research_tools = db_tools + entity_tools + pdf_tools + file_read_tools

        mock_llm = _make_mock_llm()
        researcher = create_react_agent(mock_llm, tools=research_tools, prompt=RESEARCHER_PROMPT)

        result = researcher.invoke({"messages": [HumanMessage(content="List what you know.")]})
        assert result is not None
        assert "messages" in result

    def test_invoke_result_contains_expected_fields(self, tmp_config, db):
        from langgraph.prebuilt import create_react_agent

        tools = (
            create_db_tools(db)
            + create_entity_tools(db)
            + create_pdf_tools(db)
            + create_file_read_tools(str(tmp_config.wiki_output_dir))
        )
        mock_llm = _make_mock_llm()
        agent = create_react_agent(mock_llm, tools=tools)
        result = agent.invoke({"messages": [HumanMessage(content="test")]})
        assert isinstance(result, dict)
        assert "messages" in result


# ---------------------------------------------------------------------------
# Researcher invoke with long message
# ---------------------------------------------------------------------------


class TestResearcherInvokeWithLongMessage:
    def test_invoke_with_2500_char_page_text(self, tmp_config, db):
        """Researcher invoke works with a large (~2500 char) initial message."""
        from langgraph.prebuilt import create_react_agent

        db_tools = create_db_tools(db)
        entity_tools = create_entity_tools(db)
        pdf_tools = create_pdf_tools(db)
        file_read_tools = create_file_read_tools(str(tmp_config.wiki_output_dir))
        research_tools = db_tools + entity_tools + pdf_tools + file_read_tools

        # Simulate the real initial message format used in run_for_page
        long_page_text = (
            "Thomas Chippendale (baptised 5 June 1718 – 13 November 1779) was an "
            "English furniture maker and cabinet-maker. He worked in the mid-Georgian, "
            "English Rococo, and Neoclassical styles. He was a highly influential figure "
            "in English furniture design. He published his book The Gentleman and "
            "Cabinet-Maker's Director in 1754, which contained numerous detailed "
            "engraved designs for furniture. The book was enormously influential, "
            "going through three editions. Many other craftsmen in Britain and America "
            "produced furniture in the 'Chippendale style' from these designs. "
            "Chippendale was born in Otley, Yorkshire, to a local craftsman, Thomas "
            "Chippendale senior, and his wife Mary Drake. He is said to have served "
            "his apprenticeship in York. Around 1748 he moved to London. In 1753 he "
            "moved to St Martin's Lane in London. In 1754 he published The Gentleman "
            "and Cabinet-Maker's Director. This was a landmark publication in the "
            "history of English furniture. It contained 161 engraved plates showing "
            "furniture in the Rococo style. The work brought Chippendale immediate "
            "fame and patronage. He is best known for his cabinets, bookcases, beds, "
            "and chairs. His workshop at 60 St Martin's Lane produced furniture for "
            "the leading aristocratic patrons of the day. Among his notable commissions "
            "were furnishings for Harewood House in Yorkshire, Nostell Priory, and "
            "Dumfries House in Scotland. His son Thomas Chippendale the Younger "
            "continued the family business after his death in 1779. "
        ) * 3  # ~2500 chars

        assert len(long_page_text) >= 2500, f"Message too short: {len(long_page_text)} chars"

        initial_message = HumanMessage(
            content=(
                f"You are working on the following source page:\n\n"
                f"Document: Victorian Furniture\n"
                f"Page: 1\n"
                f"Page ID: test-page-id-001\n\n"
                f"--- PAGE TEXT ---\n{long_page_text}\n--- END PAGE TEXT ---\n\n"
                f"Please research this page thoroughly."
            )
        )

        mock_llm = _make_mock_llm()
        agent = create_react_agent(mock_llm, tools=research_tools)
        result = agent.invoke({"messages": [initial_message]})
        assert result is not None
        assert len(result["messages"]) >= 1

    def test_message_size_is_logged(self, tmp_config, db):
        """Diagnostic: print message size for comparison against crash conditions."""

        page_text = "word " * 500  # 2500 chars
        initial_content = (
            f"Document: Test\nPage: 1\nPage ID: abc\n\n"
            f"--- PAGE TEXT ---\n{page_text}\n--- END PAGE TEXT ---"
        )
        print(f"\n[msg_size] Initial message content: {len(initial_content)} chars")
        assert len(initial_content) > 2000


# ---------------------------------------------------------------------------
# Salvage articles
# ---------------------------------------------------------------------------


class TestSalvageArticles:
    def _make_swarm(self, tmp_config, db):
        mock_llm = _make_mock_llm()
        wiki_client = _make_mock_wiki_client()
        with patch("docswarm.agents.swarm.ChatOllama", return_value=mock_llm):
            return DocSwarm(config=tmp_config, db=db, wiki_client=wiki_client)

    def test_creates_stub_articles_from_entities(self, tmp_config, db, sample_page):
        """Salvage creates one file per entity recorded for the page."""
        swarm = self._make_swarm(tmp_config, db)
        # Register two entities for the sample page
        eid1 = db.upsert_entity("Thomas Chippendale", "person")
        db.add_entity_mention(
            eid1,
            sample_page["id"],
            sample_page["document_id"],
            "Famous cabinet-maker from Yorkshire.",
        )
        eid2 = db.upsert_entity("Weinmann", "organisation")
        db.add_entity_mention(
            eid2, sample_page["id"], sample_page["document_id"], "Swiss brake manufacturer."
        )

        path = swarm._salvage_articles(
            messages=[HumanMessage(content="Initial"), AIMessage(content="Done.")],
            page_id=sample_page["id"],
            doc_title="Test Doc",
            page_number=1,
        )
        assert path != ""
        from pathlib import Path

        root = Path(tmp_config.wiki_output_dir)
        person_file = root / "person" / "thomas-chippendale.md"
        org_file = root / "organisation" / "weinmann.md"
        assert person_file.exists()
        assert org_file.exists()
        assert "Famous cabinet-maker" in person_file.read_text()
        assert "Swiss brake manufacturer" in org_file.read_text()

    def test_uses_writer_article_body_when_available(self, tmp_config, db, sample_page):
        """If the writer produced === ARTICLE === blocks, use that content."""
        swarm = self._make_swarm(tmp_config, db)
        eid = db.upsert_entity("Reg Harris", "person")
        db.add_entity_mention(
            eid, sample_page["id"], sample_page["document_id"], "Champion cyclist."
        )

        writer_output = (
            "=== ARTICLE: Reg Harris ===\n"
            "**Reg Harris** was a British cycling champion.\n\n"
            "He won multiple world titles.\n\n"
            '[Source: "Test Doc", p.1]\n'
            "=== END ARTICLE ==="
        )
        path = swarm._salvage_articles(
            messages=[HumanMessage(content="Initial"), AIMessage(content=writer_output)],
            page_id=sample_page["id"],
            doc_title="Test Doc",
            page_number=1,
        )
        assert path != ""
        from pathlib import Path

        saved = Path(tmp_config.wiki_output_dir) / "person" / "reg-harris.md"
        assert saved.exists()
        text = saved.read_text()
        assert "British cycling champion" in text
        assert "world titles" in text

    def test_skips_if_file_already_written(self, tmp_config, db, sample_page):
        """If 'Article written to:' appears in messages, no salvage occurs."""
        swarm = self._make_swarm(tmp_config, db)
        messages = [
            HumanMessage(content="Initial"),
            AIMessage(content="Article written to: wiki/test.md"),
        ]
        path = swarm._salvage_articles(
            messages=messages,
            page_id=sample_page["id"],
            doc_title="Test Doc",
            page_number=1,
        )
        assert path == ""

    def test_returns_empty_when_no_entities(self, tmp_config, db, sample_page):
        """Returns '' when no entities have been recorded for the page."""
        swarm = self._make_swarm(tmp_config, db)
        path = swarm._salvage_articles(
            messages=[HumanMessage(content="Start"), AIMessage(content="Nothing found.")],
            page_id=sample_page["id"],
            doc_title="Empty Doc",
            page_number=1,
        )
        assert path == ""

    def test_skips_entities_with_existing_files(self, tmp_config, db, sample_page):
        """Entities that already have article files are not overwritten."""
        swarm = self._make_swarm(tmp_config, db)
        eid = db.upsert_entity("Existing Entity", "concept")
        db.add_entity_mention(eid, sample_page["id"], sample_page["document_id"], "Context.")

        # Pre-create the file
        from pathlib import Path

        existing = Path(tmp_config.wiki_output_dir) / "concept" / "existing-entity.md"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text("# Already here\n")

        path = swarm._salvage_articles(
            messages=[HumanMessage(content="Initial")],
            page_id=sample_page["id"],
            doc_title="Test",
            page_number=1,
        )
        # Should return "" since the only entity already has a file
        assert path == ""
        # Original content should be untouched
        assert "Already here" in existing.read_text()


# ---------------------------------------------------------------------------
# run_for_page always marks studied
# ---------------------------------------------------------------------------


class TestRunForPageAlwaysMarksStudied:
    def test_page_marked_studied_even_if_swarm_fails(
        self, tmp_config, db, sample_page, sample_document
    ):
        """
        Even when the swarm graph produces no useful output, run_for_page
        must call db.log_page_study so the page is not picked up again.
        """
        mock_llm = _make_mock_llm("Short.")
        wiki_client = _make_mock_wiki_client()

        with patch("docswarm.agents.swarm.ChatOllama", return_value=mock_llm):
            swarm = DocSwarm(config=tmp_config, db=db, wiki_client=wiki_client)

        # Patch _graph.invoke to return a minimal (but valid) state
        from unittest.mock import patch as upatch

        fake_result = {"messages": [AIMessage(content="Short.")], "active_agent": "editor"}
        with upatch.object(swarm._graph, "invoke", return_value=fake_result):
            swarm.run_for_page(sample_page)

        # Page should now be marked studied
        next_page = db.get_next_unstudied_page()
        assert next_page is None


# ---------------------------------------------------------------------------
# run_full_wiki stops when all studied
# ---------------------------------------------------------------------------


class TestRunFullWikiStopsWhenAllStudied:
    def test_stops_immediately_when_no_unstudied_pages(self, tmp_config, db):
        """run_full_wiki_generation returns [] when get_next_unstudied_page is None."""
        mock_llm = _make_mock_llm()
        wiki_client = _make_mock_wiki_client()

        with patch("docswarm.agents.swarm.ChatOllama", return_value=mock_llm):
            swarm = DocSwarm(config=tmp_config, db=db, wiki_client=wiki_client)

        # No pages in DB → get_next_unstudied_page returns None immediately
        results = swarm.run_full_wiki_generation()
        assert results == []

    def test_processes_one_page_then_stops(self, tmp_config, db, sample_page, sample_document):
        """After the one unstudied page is processed, run_full_wiki_generation stops."""
        mock_llm = _make_mock_llm("Short.")
        wiki_client = _make_mock_wiki_client()

        with patch("docswarm.agents.swarm.ChatOllama", return_value=mock_llm):
            swarm = DocSwarm(config=tmp_config, db=db, wiki_client=wiki_client)

        fake_result = {"messages": [AIMessage(content="Short.")], "active_agent": "editor"}

        with patch.object(swarm._graph, "invoke", return_value=fake_result):
            results = swarm.run_full_wiki_generation()

        assert len(results) == 1


# ---------------------------------------------------------------------------
# Reviewer routing
# ---------------------------------------------------------------------------


class TestRouteFromReviewer:
    def _make_swarm(self, tmp_config, db):
        mock_llm = _make_mock_llm()
        wiki_client = _make_mock_wiki_client()
        with patch("docswarm.agents.swarm.ChatOllama", return_value=mock_llm):
            return DocSwarm(config=tmp_config, db=db, wiki_client=wiki_client)

    def test_routes_to_editor_when_no_revision_keywords(self, tmp_config, db):
        swarm = self._make_swarm(tmp_config, db)
        state = {
            "messages": [
                HumanMessage(content="Initial"),
                AIMessage(content="The article looks good. Well written and accurate."),
            ],
            "active_agent": "reviewer",
        }
        result = swarm._route_from_reviewer(state)
        assert result == "editor"

    def test_routes_to_writer_when_revision_needed(self, tmp_config, db):
        swarm = self._make_swarm(tmp_config, db)
        state = {
            "messages": [
                HumanMessage(content="Initial"),
                AIMessage(content="This article needs revision. The dates are incorrect."),
            ],
            "active_agent": "reviewer",
        }
        result = swarm._route_from_reviewer(state)
        assert result == "writer"

    @pytest.mark.parametrize(
        "keyword",
        [
            "revision needed",
            "needs revision",
            "revise",
            "rewrite",
            "return to writer",
            "back to writer",
            "incorrect",
            "inaccurate",
            "unsupported claim",
        ],
    )
    def test_routes_to_writer_for_each_keyword(self, tmp_config, db, keyword):
        swarm = self._make_swarm(tmp_config, db)
        state = {
            "messages": [
                AIMessage(content=f"The article has an issue: {keyword} in section 2."),
            ],
            "active_agent": "reviewer",
        }
        result = swarm._route_from_reviewer(state)
        assert result == "writer"

    def test_routes_to_editor_with_empty_messages(self, tmp_config, db):
        swarm = self._make_swarm(tmp_config, db)
        state = {"messages": [], "active_agent": "reviewer"}
        result = swarm._route_from_reviewer(state)
        assert result == "editor"

    def test_routes_to_editor_with_only_human_messages(self, tmp_config, db):
        swarm = self._make_swarm(tmp_config, db)
        state = {
            "messages": [HumanMessage(content="Please review this article.")],
            "active_agent": "reviewer",
        }
        result = swarm._route_from_reviewer(state)
        assert result == "editor"

    def test_uses_last_ai_message_for_routing(self, tmp_config, db):
        """Routing decision should be based on the most recent AIMessage."""
        swarm = self._make_swarm(tmp_config, db)
        state = {
            "messages": [
                AIMessage(content="revision needed here"),  # older — needs revision
                AIMessage(content="The article looks great."),  # newer — approved
            ],
            "active_agent": "reviewer",
        }
        # Most recent message approves → should route to editor
        result = swarm._route_from_reviewer(state)
        assert result == "editor"
