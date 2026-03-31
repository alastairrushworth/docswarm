"""LangGraph multi-agent swarm for wiki generation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

from docswarm.agents.personas import EDITOR_PROMPT
from docswarm.agents.personas import RESEARCHER_PROMPT
from docswarm.agents.personas import REVIEWER_PROMPT
from docswarm.agents.personas import WRITER_PROMPT
from docswarm.agents.tools.db_tools import create_db_tools
from docswarm.agents.tools.entity_tools import create_entity_tools
from docswarm.agents.tools.file_tools import _build_front_matter
from docswarm.agents.tools.file_tools import create_file_read_tools
from docswarm.agents.tools.file_tools import create_file_tools
from docswarm.agents.tools.pdf_tools import create_classification_tools
from docswarm.logger import get_logger

if TYPE_CHECKING:
    from docswarm.config import Config
    from docswarm.storage.database import DatabaseManager
    from docswarm.wiki.client import WikiJSClient

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Swarm state
# ---------------------------------------------------------------------------


class SwarmState(MessagesState):
    """State shared across all nodes in the swarm graph.

    Extends :class:`langgraph.graph.MessagesState` (which provides a
    ``messages`` field) with ``active_agent`` to track which persona is
    currently acting.
    """

    active_agent: str


# ---------------------------------------------------------------------------
# DocSwarm
# ---------------------------------------------------------------------------


class DocSwarm:
    """Orchestrates a four-agent LangGraph swarm for wiki generation.

    The swarm consists of:

    * **Researcher** – searches the source document database and collects
      relevant passages with references.
    * **Writer** – drafts a wiki article from the research findings.
    * **Reviewer** – fact-checks the draft against the source material.
    * **Editor** – applies final formatting and publishes to Wiki.js.

    The routing logic is linear by default (researcher → writer → reviewer →
    editor) with the reviewer able to send the article back to the writer for
    revision.
    """

    def __init__(
        self,
        config: "Config",
        db: "DatabaseManager",
        wiki_client: "WikiJSClient",
    ) -> None:
        """Initialise the swarm.

        Args:
            config: Application configuration (used for model name and API
                key).
            db: Open database manager.
            wiki_client: Authenticated Wiki.js client.
        """
        self.config = config
        self.db = db
        self.wiki_client = wiki_client

        log.info("Initialising swarm with model: %s", config.model)
        self._llm = ChatOllama(
            model=config.model,
            base_url=config.ollama_base_url,
            reasoning=False,
        )

        # Build tool lists
        db_tools = create_db_tools(db)
        entity_tools = create_entity_tools(db)
        classification_tools = create_classification_tools(
            db,
            model=config.model,
            ollama_base_url=config.ollama_base_url,
        )
        file_tools = create_file_tools(config.wiki_output_dir)
        file_read_tools = create_file_read_tools(config.wiki_output_dir)

        def _pick(tools: list, *names: str) -> list:
            """Return only the tools whose .name is in *names."""
            name_set = set(names)
            return [t for t in tools if t.name in name_set]

        # Keep per-agent tool counts small — small models (qwen3.5:4b) crash with >8-10 tools.
        # Researcher: classify + search + entity extraction + check existing articles (8 tools)
        research_tools = (
            classification_tools
            + _pick(db_tools, "search_chunks", "get_page_text", "list_documents")
            + _pick(entity_tools, "save_entity", "search_entities", "get_entities_for_page")
            + _pick(file_read_tools, "search_article_files")
        )
        # Writer: search source material + read existing articles for context (4 tools)
        writer_tools = _pick(db_tools, "search_chunks", "get_page_text") + _pick(
            file_read_tools, "search_article_files", "read_article_file"
        )
        # Reviewer: verify claims against source DB (2 tools)
        reviewer_tools = _pick(db_tools, "search_chunks", "get_page_text")
        # Editor: list existing articles + write article file + mark page studied (3 tools)
        editor_tools = (
            _pick(file_read_tools, "list_article_files")
            + _pick(file_tools, "write_article_file")
            + _pick(db_tools, "mark_page_studied")
        )

        for agent_name, tools in [
            ("researcher", research_tools),
            ("writer", writer_tools),
            ("reviewer", reviewer_tools),
            ("editor", editor_tools),
        ]:
            names = [t.name for t in tools]
            log.info("  %s (%d tools): %s", agent_name, len(names), ", ".join(names))

        # Build individual ReAct agents
        self._researcher = create_react_agent(
            self._llm,
            tools=research_tools,
            prompt=RESEARCHER_PROMPT,
        )
        self._writer = create_react_agent(
            self._llm,
            tools=writer_tools,
            prompt=WRITER_PROMPT,
        )
        self._reviewer = create_react_agent(
            self._llm,
            tools=reviewer_tools,
            prompt=REVIEWER_PROMPT,
        )
        self._editor = create_react_agent(
            self._llm,
            tools=editor_tools,
            prompt=EDITOR_PROMPT,
        )

        self._graph = self.build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(self) -> StateGraph:
        """Construct and compile the LangGraph StateGraph.

        The graph nodes correspond to the four agents.  Routing is determined
        by the ``active_agent`` field in :class:`SwarmState`.

        Returns:
            A compiled LangGraph runnable.
        """
        builder = StateGraph(SwarmState)

        # Register nodes (each node calls the corresponding ReAct agent)
        builder.add_node("researcher", self._run_researcher)
        builder.add_node("writer", self._run_writer)
        builder.add_node("reviewer", self._run_reviewer)
        builder.add_node("editor", self._run_editor)

        # Edges
        builder.add_edge(START, "researcher")
        builder.add_conditional_edges(
            "researcher",
            self._route_from_researcher,
            {"writer": "writer", END: END},
        )
        builder.add_edge("writer", "reviewer")
        builder.add_conditional_edges(
            "reviewer",
            self._route_from_reviewer,
            {"editor": "editor", "writer": "writer"},
        )
        builder.add_edge("editor", END)

        return builder.compile()

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _log_new_messages(agent_name: str, old_count: int, messages: list) -> None:
        """Log new messages produced by an agent invocation."""
        for msg in messages[old_count:]:
            if isinstance(msg, AIMessage):
                # Log text content
                text = msg.content
                if isinstance(text, list):
                    text = " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b) for b in text
                    )
                if text:
                    preview = text[:500].replace("\n", " ↵ ")
                    log.debug("[%s] LLM → %s", agent_name, preview)
                # Log tool calls
                for tc in getattr(msg, "tool_calls", []) or []:
                    log.debug(
                        "[%s] TOOL CALL → %s(%s)",
                        agent_name,
                        tc.get("name", "?"),
                        ", ".join(f"{k}={v!r}" for k, v in list(tc.get("args", {}).items())[:3]),
                    )
            elif isinstance(msg, ToolMessage):
                preview = str(msg.content)[:300].replace("\n", " ↵ ")
                log.debug("[%s] TOOL RESULT (%s) → %s", agent_name, msg.name, preview)

    def _run_researcher(self, state: SwarmState) -> dict:
        log.info("[researcher] starting")
        old_count = len(state["messages"])
        result = self._researcher.invoke({"messages": state["messages"]})
        self._log_new_messages("researcher", old_count, result["messages"])
        log.info("[researcher] done (%d messages)", len(result["messages"]))
        return {"messages": result["messages"], "active_agent": "researcher"}

    def _run_writer(self, state: SwarmState) -> dict:
        log.info("[writer] starting")
        old_count = len(state["messages"])
        result = self._writer.invoke({"messages": state["messages"]})
        self._log_new_messages("writer", old_count, result["messages"])
        log.info("[writer] done (%d messages)", len(result["messages"]))
        return {"messages": result["messages"], "active_agent": "writer"}

    def _run_reviewer(self, state: SwarmState) -> dict:
        log.info("[reviewer] starting")
        old_count = len(state["messages"])
        result = self._reviewer.invoke({"messages": state["messages"]})
        self._log_new_messages("reviewer", old_count, result["messages"])
        log.info("[reviewer] done (%d messages)", len(result["messages"]))
        return {"messages": result["messages"], "active_agent": "reviewer"}

    def _run_editor(self, state: SwarmState) -> dict:
        log.info("[editor] starting")
        old_count = len(state["messages"])
        result = self._editor.invoke({"messages": state["messages"]})
        self._log_new_messages("editor", old_count, result["messages"])
        log.info("[editor] done (%d messages)", len(result["messages"]))
        return {"messages": result["messages"], "active_agent": "editor"}

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Remove <think>...</think> blocks that some models emit textually."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _is_ad_skip(messages: list) -> bool:
        """Return True if the last AI message signals an advertisement skip."""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = msg.content
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b) for b in content
                    )
                content = DocSwarm._strip_think_tags(str(content)).lower()
                return "advertisement" in content and "skipping" in content
        return False

    def _route_from_researcher(self, state: SwarmState) -> str:
        """Skip the rest of the pipeline if the researcher classified the page as an ad."""
        messages = state.get("messages", [])
        if self._is_ad_skip(messages):
            log.info("Researcher flagged page as advertisement — skipping pipeline")
            return END
        return "writer"

    def _route_from_reviewer(self, state: SwarmState) -> Literal["editor", "writer"]:
        """Decide whether to send a reviewed article to the editor or back to the writer.

        The reviewer's last message is inspected for keywords indicating that
        revision is needed.  If no such keywords are found the article is
        forwarded to the editor.

        Args:
            state: Current swarm state.

        Returns:
            ``"editor"`` or ``"writer"``.
        """
        messages = state.get("messages", [])
        last_ai_content = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = msg.content
                if isinstance(content, list):
                    # Handle multi-part content blocks
                    texts = [
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    ]
                    last_ai_content = " ".join(texts)
                else:
                    last_ai_content = str(content)
                last_ai_content = self._strip_think_tags(last_ai_content)
                break

        revision_patterns = [
            r"revision needed",
            r"needs revision",
            r"\brevise\b",
            r"\brewrite\b",
            r"return to writer",
            r"back to writer",
            r"\bincorrect\b",
            r"\binaccurate\b",
            r"unsupported claim",
            r"advertising content",
            r"\bad-derived\b",
            r"promotional content",
            r"\bdrop this\b",
            r"\bshould be dropped\b",
        ]
        lower = last_ai_content.lower()
        if any(re.search(p, lower) for p in revision_patterns):
            return "writer"
        return "editor"

    # ------------------------------------------------------------------
    # Public run API
    # ------------------------------------------------------------------

    def run(self, topic: str) -> dict:
        """Run the full swarm pipeline for a single topic.

        The researcher is given an initial prompt to find material about
        *topic* in the source documents.  The swarm then progresses through
        writer → reviewer → editor automatically.

        Args:
            topic: Topic string to research and write a wiki article about.

        Returns:
            Final swarm state dict.
        """
        log.info("Starting swarm for topic: %s", topic)
        initial_message = HumanMessage(
            content=(
                f"Please research the following topic from the source documents "
                f"and produce a comprehensive wiki article: {topic}\n\n"
                f"Start by searching the database for relevant passages."
            )
        )
        initial_state: SwarmState = {
            "messages": [initial_message],
            "active_agent": "researcher",
        }
        result = self._graph.invoke(initial_state)
        log.info(
            "Swarm complete for topic: %s (%d messages)", topic, len(result.get("messages", []))
        )
        return result

    def run_for_page(self, page: dict) -> dict:
        """Run the swarm for a specific source page.

        The researcher is primed with the page's full text so it starts from
        real content rather than having to discover what to work on.

        Args:
            page: Page dict from the database (must include ``id``,
                ``document_id``, ``page_number``, and ``raw_text``).

        Returns:
            Final swarm state dict.
        """
        page_id = page["id"]
        page_number = page.get("page_number", "?")
        raw_text = page.get("raw_text") or ""

        doc = self.db.get_document(page["document_id"])
        doc_title = doc.get("title", page["document_id"]) if doc else page["document_id"]

        log.info("Starting swarm for page_id=%s (%s p.%s)", page_id, doc_title, page_number)

        initial_message = HumanMessage(
            content=(
                f"Below is a source page from a scanned magazine/book. Your job is to "
                f"identify the distinct entities (people, organisations, events, places, "
                f"objects, concepts) mentioned on this page and produce a SEPARATE focused "
                f"wiki article for each one. Articles must be about specific subjects, "
                f"NOT about the magazine page itself.\n\n"
                f"Document: {doc_title}\n"
                f"Page: {page_number}\n"
                f"Page ID: {page_id}\n\n"
                f"--- PAGE TEXT ---\n{raw_text}\n--- END PAGE TEXT ---\n\n"
                f"Use search tools to find supporting material from other pages."
            )
        )
        initial_state: SwarmState = {
            "messages": [initial_message],
            "active_agent": "researcher",
        }
        result = self._graph.invoke(initial_state)
        messages = result.get("messages", [])
        log.info("Swarm complete for page_id=%s (%d messages)", page_id, len(messages))

        # Always mark the page as studied — do not rely on the LLM to call the tool.
        # Skip salvage entirely if the researcher flagged the page as an ad.
        article_path = ""
        if not self._is_ad_skip(messages):
            article_path = self._salvage_articles(
                messages=messages,
                page_id=page_id,
                doc_title=doc_title,
                page_number=page_number,
            )
        self.db.log_page_study(
            page_id=page_id,
            document_id=page["document_id"],
            wiki_article_path=article_path,
        )
        return result

    def _salvage_articles(
        self,
        messages: list,
        page_id: str,
        doc_title: str,
        page_number: int | str,
    ) -> str:
        """Ensure every entity found on this page has an article file.

        First checks whether the editor already wrote files.  Then checks
        the entity database for entities mentioned on this page and creates
        a stub article for any entity that does not yet have a file.

        If the writer produced usable article sections in the messages
        (delimited by ``=== ARTICLE: ... ===``), those are used as content.
        Otherwise a stub is generated from the entity's context text.

        Returns:
            The first article path written, or empty string if nothing saved.
        """
        root = Path(self.config.wiki_output_dir)

        # Check if editor already wrote files during this run
        for msg in messages:
            if isinstance(msg, AIMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if "Article written to:" in content:
                    log.debug("Editor appears to have written file(s) — no salvage needed")
                    return ""

        # Try to extract per-entity articles from writer output.
        # The writer is prompted to delimit articles with === ARTICLE: name ===
        article_bodies: dict[str, str] = {}
        all_ai_text = ""
        for msg in messages:
            if isinstance(msg, AIMessage):
                c = msg.content if isinstance(msg.content, str) else ""
                if isinstance(msg.content, list):
                    c = " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b) for b in msg.content
                    )
                all_ai_text += c + "\n"

        # Parse === ARTICLE: name === ... === END ARTICLE === blocks
        article_pattern = re.compile(
            r"===\s*ARTICLE:\s*([^\n=]+?)\s*===\s*\n(.*?)===\s*END\s+ARTICLE\s*===",
            re.DOTALL | re.IGNORECASE,
        )
        for match in article_pattern.finditer(all_ai_text):
            name = match.group(1).strip()
            body = match.group(2).strip()
            if body and "ARTICLE NEEDED" not in body and "entity_id" not in body.lower():
                article_bodies[name.lower()] = body

        # Get all entities the researcher recorded for this page
        entities = self.db.get_entities_for_page(page_id)
        if not entities:
            log.warning("No entities found for page_id=%s — nothing to salvage", page_id)
            return ""

        saved_paths = []
        for entity in entities:
            name = entity.get("name", "")
            entity_type = entity.get("entity_type", "misc")
            context = entity.get("context_text", "")

            # Canonicalise entity type to avoid folder duplication
            _CANONICAL_TYPES = {
                "people": "person",
                "persons": "person",
                "organisations": "organisation",
                "orgs": "organisation",
                "org": "organisation",
                "company": "organisation",
                "places": "place",
                "location": "place",
                "locations": "place",
                "events": "event",
                "objects": "object",
                "thing": "object",
                "things": "object",
                "concepts": "concept",
                "idea": "concept",
            }
            raw_type = re.sub(r"[^a-z0-9]+", "-", entity_type.lower()).strip("-") or "misc"
            safe_type = _CANONICAL_TYPES.get(raw_type, raw_type)
            safe_name = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
            if not safe_name:
                log.warning("Entity name %r has no alphanumeric chars — skipping", name)
                continue
            path = f"{safe_type}/{safe_name}"
            file_path = root / (path + ".md")

            # Skip if file already exists (editor or previous page already created it)
            if file_path.exists():
                log.debug("Article already exists: %s", file_path)
                continue

            # Use writer's article body if available, otherwise create a stub
            body = article_bodies.get(name.lower(), "")
            if not body:
                # Build a stub article from entity context
                body = f"**{name}** is a {entity_type}.<sup>[1]</sup>\n"
                if context:
                    body += f"\n{context}<sup>[1]</sup>\n"
                body += f'\n## References\n\n1. "{doc_title}", p.{page_number}\n'

            file_path.parent.mkdir(parents=True, exist_ok=True)
            meta = {
                "title": name,
                "description": f"{name} — {entity_type}",
                "entity_id": entity.get("id", ""),
                "entity_type": entity_type,
                "source_page_id": page_id,
                "wiki_page_id": None,
            }
            file_path.write_text(_build_front_matter(meta) + body, encoding="utf-8")
            log.info("Salvaged article → %s", file_path)
            saved_paths.append(path)

        if not saved_paths:
            log.info("All entities for page_id=%s already have articles", page_id)
            return ""

        log.info("Salvaged %d article(s) for page_id=%s", len(saved_paths), page_id)
        return ", ".join(saved_paths)

    def run_full_wiki_generation(self) -> list[dict]:
        """Generate wiki articles for all unstudied pages in the database.

        Repeatedly fetches the next unstudied page and runs the swarm until
        all pages have been covered.

        Returns:
            List of final state dicts, one per page processed.
        """
        results = []
        while True:
            page = self.db.get_next_unstudied_page()
            if page is None:
                log.info("All pages have been studied.")
                break
            result = self.run_for_page(page)
            results.append(result)
        return results
