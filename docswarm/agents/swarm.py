"""LangGraph multi-agent swarm for wiki generation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

from docswarm.agents.personas import RESEARCHER_PROMPT
from docswarm.agents.personas import WRITER_PROMPT
from docswarm.agents.tools.db_tools import create_db_tools
from docswarm.agents.tools.entity_tools import create_entity_tools
from docswarm.agents.tools.file_tools import _build_front_matter
from docswarm.agents.tools.file_tools import create_file_read_tools
from docswarm.agents.tools.pdf_tools import create_classification_tools
from docswarm.logger import get_logger

if TYPE_CHECKING:
    from docswarm.config import Config
    from docswarm.storage.database import DatabaseManager
    from docswarm.wiki.client import WikiJSClient

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical mapping for entity type folder names.
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

# Masthead / credits role words — entities whose context is dominated by these
# are likely staff credits rather than editorial subjects.
_MASTHEAD_ROLES = {
    "manager",
    "director",
    "editor",
    "department",
    "secretary",
    "photographer",
    "printer",
    "typesetter",
    "publisher",
    "correspondent",
    "photographic",
}

# Article block regex used by both the deterministic editor and salvage.
_ARTICLE_PATTERN = re.compile(
    r"===\s*ARTICLE:\s*([^\n=]+?)\s*===\s*\n(.*?)===\s*END\s+ARTICLE\s*===",
    re.DOTALL | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(s: str) -> str:
    """Lowercase, strip non-alphanumeric characters for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _safe_type(entity_type: str) -> str:
    """Canonicalise an entity type string to a folder name."""
    raw = re.sub(r"[^a-z0-9]+", "-", entity_type.lower()).strip("-") or "misc"
    return _CANONICAL_TYPES.get(raw, raw)


def _safe_name(name: str) -> str:
    """Slugify an entity name for use as a filename."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _is_masthead_entity(entity: dict) -> bool:
    """Return True if an entity looks like a masthead credit, not an editorial subject."""
    context = (entity.get("context_text") or "").lower()
    if len(context.split()) > 20:
        return False  # substantial context — probably editorial
    return bool(_MASTHEAD_ROLES & set(context.split()))


def _get_page_classification(messages: list) -> str:
    """Extract the page classification from the classify_page_content tool result.

    Returns one of: "advertisement", "editorial", "mixed", or "unknown".
    """
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "classify_page_content":
            content = str(msg.content).lower()
            if "classification:" in content:
                after = content.split("classification:", 1)[1].strip()
                for label in ("advertisement", "editorial", "mixed"):
                    if after.startswith(label):
                        return label
    return "unknown"


def _collect_ai_text(messages: list) -> str:
    """Concatenate all AIMessage content into a single string."""
    parts = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            c = msg.content if isinstance(msg.content, str) else ""
            if isinstance(msg.content, list):
                c = " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in msg.content
                )
            if c:
                parts.append(c)
    return "\n".join(parts)


# Fallback pattern: markdown heading like "# Entity Name" or "## Entity Name"
# followed by body text until the next top-level heading or end of text.
_MD_HEADING_PATTERN = re.compile(
    r"^#{1,2}\s+(?:\d+\.\s+)?(.+?)$\n(.*?)(?=^#{1,2}\s|\Z)",
    re.MULTILINE | re.DOTALL,
)

# Headings to skip when using fallback parsing
_SKIP_HEADINGS = {
    "references", "key facts", "summary", "entities identified",
    "entities", "notes", "sources", "introduction", "overview",
    "page analysis", "page summary", "article status",
    "new articles created", "existing articles", "entities needing articles",
}

# Regex patterns in headings that indicate meta-content, not real articles
_META_HEADING_RE = re.compile(
    r"entit(?:y|ies)|summary|page\s*\d|page\s*analysis|article\s*status|"
    r"supporting\s*material|new\s*article|existing\s*article",
    re.IGNORECASE,
)


def _extract_article_blocks(messages: list) -> dict[str, tuple[str, str]]:
    """Parse === ARTICLE: name === blocks from all AI messages.

    Falls back to markdown heading parsing if no delimited blocks are found.

    Returns:
        Dict mapping normalized name → (original_title, body).
    """
    all_ai_text = _collect_ai_text(messages)

    # Primary: look for explicit delimiters
    blocks: dict[str, tuple[str, str]] = {}
    for match in _ARTICLE_PATTERN.finditer(all_ai_text):
        title = match.group(1).strip()
        body = match.group(2).strip()
        if body and "ARTICLE NEEDED" not in body and "entity_id" not in body.lower():
            blocks[_normalize(title)] = (title, body)

    if blocks:
        return blocks

    # Fallback: parse markdown headings as article boundaries
    for match in _MD_HEADING_PATTERN.finditer(all_ai_text):
        title = match.group(1).strip().strip("*").strip()
        body = match.group(2).strip()
        if not body or len(body) < 20:
            continue
        if _normalize(title) in _SKIP_HEADINGS:
            continue
        if _META_HEADING_RE.search(title):
            continue
        if "ARTICLE NEEDED" in body or "entity_id" in body.lower():
            continue
        blocks[_normalize(title)] = (title, body)

    if blocks:
        log.info("No delimited blocks found; extracted %d from markdown headings", len(blocks))

    return blocks


def _match_entity_to_article(
    entity_name: str, article_blocks: dict[str, tuple[str, str]]
) -> tuple[str, str] | None:
    """Find the best matching article block for an entity name.

    Tries exact normalized match, then substring containment.
    Returns (title, body) or None.
    """
    norm = _normalize(entity_name)
    # Exact match
    if norm in article_blocks:
        return article_blocks[norm]
    # Entity name is a substring of an article title
    for key, val in article_blocks.items():
        if norm in key or key in norm:
            return val
    return None


# ---------------------------------------------------------------------------
# Swarm state
# ---------------------------------------------------------------------------


class SwarmState(MessagesState):
    """State shared across all nodes in the swarm graph."""

    active_agent: str


# ---------------------------------------------------------------------------
# DocSwarm
# ---------------------------------------------------------------------------


class DocSwarm:
    """Orchestrates a three-stage pipeline for wiki generation.

    * **Researcher** (LLM) – classifies, extracts entities, searches.
    * **Writer** (LLM) – drafts wiki articles from research.
    * **Editor** (deterministic Python) – parses writer output, writes files.
    """

    def __init__(
        self,
        config: "Config",
        db: "DatabaseManager",
        wiki_client: "WikiJSClient",
    ) -> None:
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
        file_read_tools = create_file_read_tools(config.wiki_output_dir)

        def _pick(tools: list, *names: str) -> list:
            name_set = set(names)
            return [t for t in tools if t.name in name_set]

        research_tools = (
            classification_tools
            + _pick(db_tools, "search_chunks", "get_page_text", "list_documents")
            + _pick(entity_tools, "save_entity", "search_entities", "get_entities_for_page")
            + _pick(file_read_tools, "search_article_files")
        )
        writer_tools = _pick(db_tools, "search_chunks", "get_page_text") + _pick(
            file_read_tools, "search_article_files", "read_article_file"
        )

        for agent_name, tools in [
            ("researcher", research_tools),
            ("writer", writer_tools),
        ]:
            names = [t.name for t in tools]
            log.info("  %s (%d tools): %s", agent_name, len(names), ", ".join(names))
        log.info("  editor (deterministic, no LLM)")

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

        self._graph = self.build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(self) -> StateGraph:
        builder = StateGraph(SwarmState)
        builder.add_node("researcher", self._run_researcher)
        builder.add_node("writer", self._run_writer)
        builder.add_node("editor", self._run_editor)

        builder.add_edge(START, "researcher")
        builder.add_conditional_edges(
            "researcher",
            self._route_from_researcher,
            {"writer": "writer", END: END},
        )
        builder.add_edge("writer", "editor")
        builder.add_edge("editor", END)

        return builder.compile()

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _log_new_messages(agent_name: str, old_count: int, messages: list) -> None:
        for msg in messages[old_count:]:
            if isinstance(msg, AIMessage):
                text = msg.content
                if isinstance(text, list):
                    text = " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b) for b in text
                    )
                if text:
                    preview = text[:500].replace("\n", " ↵ ")
                    log.debug("[%s] LLM → %s", agent_name, preview)
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

    def _run_editor(self, state: SwarmState) -> dict:
        """Deterministic editor: parse writer output and write article files.

        No LLM call — this is pure Python.
        """
        log.info("[editor] starting")
        messages = state["messages"]

        # Extract page_id from the initial human message
        page_id = ""
        doc_title = ""
        page_number = "?"
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = str(msg.content)
                m = re.search(r"Page ID:\s*(\S+)", content)
                if m:
                    page_id = m.group(1)
                m = re.search(r"Document:\s*(.+)", content)
                if m:
                    doc_title = m.group(1).strip()
                m = re.search(r"Page:\s*(\S+)", content)
                if m:
                    page_number = m.group(1)
                break

        # Parse article blocks from writer output
        article_blocks = _extract_article_blocks(messages)
        log.info("[editor] found %d article block(s) from writer", len(article_blocks))
        if not article_blocks:
            # Log raw writer text for debugging
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    raw = msg.content if isinstance(msg.content, str) else str(msg.content)
                    log.debug("[editor] last writer AI text: %s", raw[:800].replace("\n", " ↵ "))
                    break

        # Get entities for this page
        entities = self.db.get_entities_for_page(page_id) if page_id else []

        # Filter out masthead entities
        filtered = [e for e in entities if not _is_masthead_entity(e)]
        if len(filtered) < len(entities):
            log.info(
                "[editor] filtered %d masthead entit(ies)", len(entities) - len(filtered)
            )

        root = Path(self.config.wiki_output_dir)
        written_paths = []
        matched_blocks = set()  # track which article blocks we've used

        # Phase 1: Write articles for entities that have matching writer output
        for entity in filtered:
            name = entity.get("name", "")
            entity_type = entity.get("entity_type", "misc")
            context = entity.get("context_text", "")
            slug = _safe_name(name)
            if not slug:
                continue

            etype = _safe_type(entity_type)
            path = f"{etype}/{slug}"
            file_path = root / (path + ".md")

            if file_path.exists():
                log.debug("[editor] article already exists: %s", file_path)
                continue

            # Try to match with writer's article blocks
            match = _match_entity_to_article(name, article_blocks)
            if match:
                title, body = match
                matched_blocks.add(_normalize(title))
            else:
                # Stub from entity context
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
            log.info("[editor] wrote article → %s", file_path)
            written_paths.append(str(file_path))

        # Phase 2: Write any unmatched article blocks (writer produced content
        # for entities the researcher didn't save — still valuable)
        for norm_key, (title, body) in article_blocks.items():
            if norm_key in matched_blocks:
                continue
            slug = _safe_name(title)
            if not slug:
                continue
            if _META_HEADING_RE.search(title):
                log.debug("[editor] skipping meta-content block: %s", title)
                continue
            # Infer type from content or default to "misc"
            etype = "misc"
            for label in ("person", "organisation", "place", "event", "object", "concept"):
                if label in body.lower()[:200]:
                    etype = label
                    break
            path = f"{etype}/{slug}"
            file_path = root / (path + ".md")
            if file_path.exists():
                continue
            file_path.parent.mkdir(parents=True, exist_ok=True)
            meta = {
                "title": title,
                "description": title,
                "entity_id": "",
                "entity_type": etype,
                "source_page_id": page_id,
                "wiki_page_id": None,
            }
            file_path.write_text(_build_front_matter(meta) + body, encoding="utf-8")
            log.info("[editor] wrote unmatched article → %s", file_path)
            written_paths.append(str(file_path))

        summary = (
            f"Article written to: {', '.join(written_paths)}"
            if written_paths
            else "No articles written."
        )
        log.info("[editor] done — wrote %d article(s)", len(written_paths))

        new_messages = list(messages) + [AIMessage(content=summary)]
        return {"messages": new_messages, "active_agent": "editor"}

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Remove <think>...</think> blocks that some models emit textually."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _route_from_researcher(self, state: SwarmState) -> str:
        """Skip the pipeline only for pages classified as 'advertisement'.

        Uses the classify_page_content tool result directly, not the LLM's
        prose interpretation, to avoid false positives on mixed pages.
        """
        messages = state.get("messages", [])
        classification = _get_page_classification(messages)
        if classification == "advertisement":
            log.info("Page classified as advertisement — skipping pipeline")
            return END
        log.info("Page classified as %s — proceeding", classification)
        return "writer"

    # ------------------------------------------------------------------
    # Public run API
    # ------------------------------------------------------------------

    def run(self, topic: str) -> dict:
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

        # Determine article path from editor output
        article_path = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and "Article written to:" in str(msg.content):
                article_path = str(msg.content).replace("Article written to: ", "")
                break

        self.db.log_page_study(
            page_id=page_id,
            document_id=page["document_id"],
            wiki_article_path=article_path,
        )
        return result

    def run_full_wiki_generation(self) -> list[dict]:
        results = []
        while True:
            page = self.db.get_next_unstudied_page()
            if page is None:
                log.info("All pages have been studied.")
                break
            result = self.run_for_page(page)
            results.append(result)
        return results
