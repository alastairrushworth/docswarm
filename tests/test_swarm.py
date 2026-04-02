"""Tests for docswarm.agents.swarm.DocSwarm."""

from __future__ import annotations

import json
from typing import Any
from typing import List
from typing import Optional
from unittest.mock import MagicMock
from unittest.mock import patch

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import ChatResult

from docswarm.agents.swarm import DocSwarm
from docswarm.agents.tools.db_tools import create_db_tools
from docswarm.agents.tools.file_tools import create_file_read_tools
from docswarm.agents.tools.pdf_tools import create_classification_tools
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

    def test_swarm_builds_graph_with_three_nodes(self, tmp_config, db):
        mock_llm = _make_mock_llm()
        wiki_client = _make_mock_wiki_client()

        with patch("docswarm.agents.swarm.ChatOllama", return_value=mock_llm):
            swarm = DocSwarm(config=tmp_config, db=db, wiki_client=wiki_client)

        # The compiled graph has nodes for researcher, writer, editor
        graph_nodes = swarm._graph.get_graph().nodes
        node_names = set(graph_nodes.keys())
        for expected in ("researcher", "writer", "editor"):
            assert expected in node_names


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

    def test_tool_names(self, tmp_config, db):
        """Verify the researcher and writer tool lists match what the swarm builds."""
        db_tools = create_db_tools(db)
        classification_tools = create_classification_tools(db, config=tmp_config)
        file_read_tools = create_file_read_tools(str(tmp_config.wiki_output_dir))

        def _pick(tools, *names):
            name_set = set(names)
            return [t for t in tools if t.name in name_set]

        research_tools = (
            classification_tools
            + _pick(db_tools, "search_chunks", "get_page_text", "list_documents")
            + _pick(file_read_tools, "search_article_files")
        )
        writer_tools = _pick(db_tools, "search_chunks", "get_page_text") + _pick(
            file_read_tools, "search_article_files", "read_article_file"
        )

        research_names = {t.name for t in research_tools}
        writer_names = {t.name for t in writer_tools}

        assert research_names == {
            "classify_page_content",
            "search_chunks",
            "get_page_text",
            "list_documents",
            "search_article_files",
        }
        assert writer_names == {
            "search_chunks",
            "get_page_text",
            "search_article_files",
            "read_article_file",
        }


# ---------------------------------------------------------------------------
# Tool schema serialization
# ---------------------------------------------------------------------------


class TestToolSchemaSerialization:
    def test_all_researcher_tool_schemas_serialize_to_json(self, tmp_config, db):
        """Every tool's schema must be JSON-serialisable."""
        db_tools = create_db_tools(db)
        classification_tools = create_classification_tools(db, config=tmp_config)
        file_read_tools = create_file_read_tools(str(tmp_config.wiki_output_dir))
        all_tools = db_tools + classification_tools + file_read_tools

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
    @staticmethod
    def _build_research_tools(tmp_config, db):
        db_tools = create_db_tools(db)
        classification_tools = create_classification_tools(db, config=tmp_config)
        file_read_tools = create_file_read_tools(str(tmp_config.wiki_output_dir))

        def _pick(tools, *names):
            return [t for t in tools if t.name in set(names)]

        return (
            classification_tools
            + _pick(db_tools, "search_chunks", "get_page_text", "list_documents")
            + _pick(file_read_tools, "search_article_files")
        )

    def test_invoke_does_not_crash_with_short_message(self, tmp_config, db):
        """Researcher invoke works with a short message and mocked LLM."""
        from langgraph.prebuilt import create_react_agent

        from docswarm.agents.personas import RESEARCHER_PROMPT

        research_tools = self._build_research_tools(tmp_config, db)
        mock_llm = _make_mock_llm()
        researcher = create_react_agent(mock_llm, tools=research_tools, prompt=RESEARCHER_PROMPT)

        initial_message = HumanMessage(content="Research the topic: Victorian furniture.")
        result = researcher.invoke({"messages": [initial_message]})
        assert "messages" in result
        assert len(result["messages"]) > 0

    def test_last_message_is_ai_message(self, tmp_config, db):
        from langgraph.prebuilt import create_react_agent

        from docswarm.agents.personas import RESEARCHER_PROMPT

        research_tools = self._build_research_tools(tmp_config, db)
        mock_llm = _make_mock_llm()
        researcher = create_react_agent(mock_llm, tools=research_tools, prompt=RESEARCHER_PROMPT)

        result = researcher.invoke({"messages": [HumanMessage(content="Test")]})
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_messages) >= 1


# ---------------------------------------------------------------------------
# Researcher invoke with all tools
# ---------------------------------------------------------------------------


class TestResearcherInvokeWithAllTools:
    def test_create_react_agent_accepts_all_tools(self, tmp_config, db):
        """Verify create_react_agent works with the researcher tool set."""
        from langgraph.prebuilt import create_react_agent

        from docswarm.agents.personas import RESEARCHER_PROMPT

        research_tools = TestResearcherInvokeWithMockLLM._build_research_tools(tmp_config, db)

        mock_llm = _make_mock_llm()
        researcher = create_react_agent(mock_llm, tools=research_tools, prompt=RESEARCHER_PROMPT)

        result = researcher.invoke({"messages": [HumanMessage(content="List what you know.")]})
        assert result is not None
        assert "messages" in result

    def test_invoke_result_contains_expected_fields(self, tmp_config, db):
        from langgraph.prebuilt import create_react_agent

        research_tools = TestResearcherInvokeWithMockLLM._build_research_tools(tmp_config, db)
        mock_llm = _make_mock_llm()
        agent = create_react_agent(mock_llm, tools=research_tools)
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

        research_tools = TestResearcherInvokeWithMockLLM._build_research_tools(tmp_config, db)

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


class TestRunEditor:
    """Tests for the deterministic _run_editor method."""

    def _make_swarm(self, tmp_config, db):
        mock_llm = _make_mock_llm()
        wiki_client = _make_mock_wiki_client()
        with patch("docswarm.agents.swarm.ChatOllama", return_value=mock_llm):
            return DocSwarm(config=tmp_config, db=db, wiki_client=wiki_client)

    def _make_state(self, messages, page_id, doc_title="Test Doc", page_number=1):
        """Build a SwarmState dict with a properly formatted initial message."""
        initial = HumanMessage(
            content=f"Page ID: {page_id}\nDocument: {doc_title}\nPage: {page_number}"
        )
        return {"messages": [initial] + messages, "active_agent": "writer"}

    def test_creates_stub_articles_from_entities(self, tmp_config, db, sample_page):
        """Editor creates one file per entity from researcher entity blocks."""
        swarm = self._make_swarm(tmp_config, db)

        researcher_output = (
            "=== ENTITY: Thomas Chippendale (person) ===\n"
            "Famous cabinet-maker from Yorkshire.\n"
            'Source: "Test Doc", p.1\n'
            "=== END ENTITY ===\n\n"
            "=== ENTITY: Weinmann (organisation) ===\n"
            "Swiss brake manufacturer.\n"
            'Source: "Test Doc", p.1\n'
            "=== END ENTITY ==="
        )
        state = self._make_state(
            [AIMessage(content=researcher_output), AIMessage(content="Done.")],
            page_id=sample_page["id"],
        )
        result = swarm._run_editor(state)
        summary = result["messages"][-1].content
        assert "Article written to:" in summary

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

        researcher_output = (
            "=== ENTITY: Reg Harris (person) ===\n"
            "Champion cyclist.\n"
            'Source: "Test Doc", p.1\n'
            "=== END ENTITY ==="
        )
        writer_output = (
            "=== ARTICLE: Reg Harris ===\n"
            "**Reg Harris** was a British cycling champion.\n\n"
            "He won multiple world titles.\n\n"
            '[Source: "Test Doc", p.1]\n'
            "=== END ARTICLE ==="
        )
        state = self._make_state(
            [AIMessage(content=researcher_output), AIMessage(content=writer_output)],
            page_id=sample_page["id"],
        )
        result = swarm._run_editor(state)
        summary = result["messages"][-1].content
        assert "Article written to:" in summary

        from pathlib import Path

        saved = Path(tmp_config.wiki_output_dir) / "person" / "reg-harris.md"
        assert saved.exists()
        text = saved.read_text()
        assert "British cycling champion" in text
        assert "world titles" in text

    def test_writes_nothing_when_no_entities(self, tmp_config, db, sample_page):
        """Returns 'No articles written.' when no entities exist for the page."""
        swarm = self._make_swarm(tmp_config, db)
        state = self._make_state(
            [AIMessage(content="Nothing found.")], page_id=sample_page["id"]
        )
        result = swarm._run_editor(state)
        summary = result["messages"][-1].content
        assert summary == "No articles written."

    def test_skips_entities_with_existing_files(self, tmp_config, db, sample_page):
        """Entities that already have article files are not overwritten."""
        swarm = self._make_swarm(tmp_config, db)

        researcher_output = (
            "=== ENTITY: Existing Entity (concept) ===\n"
            "Some context.\n"
            "=== END ENTITY ==="
        )

        from pathlib import Path

        existing = Path(tmp_config.wiki_output_dir) / "concept" / "existing-entity.md"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text("# Already here\n")

        state = self._make_state(
            [AIMessage(content=researcher_output)], page_id=sample_page["id"]
        )
        result = swarm._run_editor(state)
        summary = result["messages"][-1].content
        assert summary == "No articles written."
        assert "Already here" in existing.read_text()

    def test_filters_masthead_entities(self, tmp_config, db, sample_page):
        """Entities with masthead roles in short context are filtered out."""
        swarm = self._make_swarm(tmp_config, db)

        researcher_output = (
            "=== ENTITY: John Smith (person) ===\n"
            "Editor of the magazine.\n"
            "=== END ENTITY ==="
        )
        state = self._make_state(
            [AIMessage(content=researcher_output)], page_id=sample_page["id"]
        )
        result = swarm._run_editor(state)
        summary = result["messages"][-1].content
        assert summary == "No articles written."


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
