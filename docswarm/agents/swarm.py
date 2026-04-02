"""Multi-agent swarm for wiki generation from scanned documents."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from docswarm.agents.personas import RESEARCHER_PROMPT
from docswarm.agents.personas import WRITER_PROMPT
from docswarm.agents.tools.file_tools import _build_front_matter
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

# Masthead / credits role words — entities whose info is dominated by these
# are likely staff credits rather than editorial subjects.
_MASTHEAD_ROLES = {
    "manager", "director", "editor", "department", "secretary",
    "photographer", "printer", "typesetter", "publisher",
    "correspondent", "photographic",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks that some models emit textually."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _safe_type(entity_type: str) -> str:
    """Canonicalise an entity type string to a folder name."""
    raw = re.sub(r"[^a-z0-9]+", "-", entity_type.lower()).strip("-") or "misc"
    return _CANONICAL_TYPES.get(raw, raw)


def _safe_name(name: str) -> str:
    """Slugify an entity name for use as a filename."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _is_masthead_entity(info: str) -> bool:
    """Return True if entity info looks like a masthead credit."""
    words = info.lower().split()
    if len(words) > 20:
        return False  # substantial info — probably editorial
    return bool(_MASTHEAD_ROLES & set(words))


# ---------------------------------------------------------------------------
# DocSwarm
# ---------------------------------------------------------------------------


class DocSwarm:
    """Orchestrates a two-stage pipeline for wiki generation.

    * **Researcher** (LLM, JSON mode) – classifies page, extracts entity dicts.
    * **Writer** (LLM) – creates or updates a wiki article per entity.
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

        if config.use_ollama:
            log.info("Initialising swarm with Ollama model: %s", config.model)
            self._researcher_llm = ChatOllama(
                model=config.model,
                base_url=config.ollama_base_url,
                format="json",
                reasoning=False,
            )
            self._writer_llm = ChatOllama(
                model=config.model,
                base_url=config.ollama_base_url,
                reasoning=False,
            )
        else:
            log.info("Initialising swarm with OpenAI model: %s", config.openai_model)
            self._researcher_llm = ChatOpenAI(
                model=config.openai_model,
                api_key=config.openai_api_key,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
            self._writer_llm = ChatOpenAI(
                model=config.openai_model,
                api_key=config.openai_api_key,
            )

        # Classification tool — called directly, not through an agent
        classification_tools = create_classification_tools(db, config=config)
        self._classify_tool = classification_tools[0]

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify(self, page_id: str) -> str:
        """Classify a page as advertisement/editorial/mixed."""
        result = self._classify_tool.invoke({"page_id": page_id})
        log.info("[classify] page=%s → %s", page_id, result)
        content = str(result).lower()
        if "classification:" in content:
            after = content.split("classification:", 1)[1].strip()
            for label in ("advertisement", "editorial", "mixed"):
                if after.startswith(label):
                    return label
        return "unknown"

    # ------------------------------------------------------------------
    # Researcher
    # ------------------------------------------------------------------

    def _research(self, raw_text: str, doc_title: str, page_number: str) -> list[dict]:
        """Call the researcher LLM in JSON mode to extract entities."""
        user_content = (
            f"Document: {doc_title}\n"
            f"Page: {page_number}\n\n"
            f"--- PAGE TEXT ---\n{raw_text}\n--- END PAGE TEXT ---"
        )
        messages = [
            SystemMessage(content=RESEARCHER_PROMPT),
            HumanMessage(content=user_content),
        ]
        log.debug("[researcher] INPUT: %d chars of page text", len(raw_text))
        response = self._researcher_llm.invoke(messages)
        log.debug("[researcher] OUTPUT: %s", str(response.content)[:500])

        try:
            data = json.loads(response.content)
            entities = data.get("entities", []) if isinstance(data, dict) else data
            if not isinstance(entities, list):
                log.warning("[researcher] expected list, got %s", type(entities).__name__)
                return []
            return entities
        except (json.JSONDecodeError, TypeError) as e:
            log.warning("[researcher] failed to parse JSON: %s — %s", e, str(response.content)[:200])
            return []

    # ------------------------------------------------------------------
    # Writer
    # ------------------------------------------------------------------

    def _write_entity(
        self,
        entity: dict,
        doc_title: str,
        page_number: str,
        page_id: str,
    ) -> str | None:
        """Write or update a wiki article for a single entity. Returns the file path or None."""
        name = entity.get("entity", "").strip()
        entity_type = _safe_type(entity.get("type", "misc"))
        info = entity.get("info", "").strip()
        source = entity.get("source", f'"{doc_title}", p.{page_number}')

        if not name or not info:
            return None

        if _is_masthead_entity(info):
            log.debug("[writer] skipping masthead entity: %s", name)
            return None

        slug = _safe_name(name)
        if not slug:
            return None

        root = Path(self.config.wiki_output_dir)
        file_path = root / entity_type / (slug + ".md")

        # Read existing article if present
        existing_content = ""
        if file_path.exists():
            existing_content = file_path.read_text(encoding="utf-8")
            # Strip front matter for the LLM — we regenerate it
            if existing_content.startswith("---"):
                parts = existing_content.split("---", 2)
                if len(parts) >= 3:
                    existing_content = parts[2].strip()

        # Build writer input
        user_content = (
            f"Entity: {name}\n"
            f"Type: {entity_type}\n"
            f"New information: {info}\n"
            f"Source: {source}\n"
        )
        if existing_content:
            user_content += f"\nExisting article:\n{existing_content}"

        messages = [
            SystemMessage(content=WRITER_PROMPT),
            HumanMessage(content=user_content),
        ]
        log.debug("[writer] INPUT entity=%s existing=%d chars", name, len(existing_content))
        response = self._writer_llm.invoke(messages)
        article_body = response.content.strip()
        log.debug("[writer] OUTPUT: %s", article_body[:300])

        if not article_body:
            return None

        # Write to disk
        file_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "title": name,
            "description": f"{name} — {entity_type}",
            "entity_type": entity_type,
            "source_page_id": page_id,
            "wiki_page_id": None,
        }
        file_path.write_text(_build_front_matter(meta) + article_body + "\n", encoding="utf-8")
        action = "updated" if existing_content else "created"
        log.info("[writer] %s article → %s", action, file_path)
        return str(file_path)

    # ------------------------------------------------------------------
    # Public run API
    # ------------------------------------------------------------------

    def run_for_page(self, page: dict) -> dict:
        """Process a single page: classify, research, write articles."""
        page_id = page["id"]
        page_number = str(page.get("page_number", "?"))
        raw_text = page.get("raw_text") or ""

        doc = self.db.get_document(page["document_id"])
        doc_title = doc.get("title", page["document_id"]) if doc else page["document_id"]

        log.info("Processing page_id=%s (%s p.%s)", page_id, doc_title, page_number)

        # 1. Classify
        classification = self._classify(page_id)
        if classification == "advertisement":
            log.info("Page classified as advertisement — skipping")
            self.db.log_page_study(
                page_id=page_id,
                document_id=page["document_id"],
                wiki_article_path="",
            )
            return {"skipped": True, "reason": "advertisement"}

        log.info("Page classified as %s — extracting entities", classification)

        # 2. Research — extract entities as JSON
        entities = self._research(raw_text, doc_title, page_number)
        log.info("Researcher found %d entity(ies)", len(entities))

        # 3. Write/update article for each entity
        written_paths = []
        for entity in entities:
            path = self._write_entity(entity, doc_title, page_number, page_id)
            if path:
                written_paths.append(path)

        log.info("Wrote %d article(s) for page %s", len(written_paths), page_id)
        self.db.log_page_study(
            page_id=page_id,
            document_id=page["document_id"],
            wiki_article_path=", ".join(written_paths),
        )
        return {"written": written_paths}

    def run_full_wiki_generation(self) -> list[dict]:
        """Process all unstudied pages."""
        results = []
        while True:
            page = self.db.get_next_unstudied_page()
            if page is None:
                log.info("All pages have been studied.")
                break
            result = self.run_for_page(page)
            results.append(result)
        return results
