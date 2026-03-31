"""LangChain tools for entity extraction and tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from docswarm.storage.database import DatabaseManager


def create_entity_tools(db: "DatabaseManager") -> list:
    """Create entity-tracking tools with the database instance bound.

    These tools are used primarily by the Researcher agent to extract and
    record named entities from source pages, and to check how well each
    entity is already covered in the wiki.

    Args:
        db: An initialised :class:`~docswarm.storage.database.DatabaseManager`.

    Returns:
        List of LangChain tool callables.
    """

    @tool
    def save_entity(
        name: str, entity_type: str, page_id: str, context_text: str = ""
    ) -> str:
        """Save a named entity found on a source page.

        Call this once per entity identified on the page being researched.
        If the entity already exists in the database the existing record is
        returned; only the mention (page_id + context) is always added as new.

        Entity types to use: person, place, event, object, organisation, concept.

        Args:
            name: Canonical name of the entity (e.g. 'Thomas Chippendale').
            entity_type: Category — person, place, event, object, organisation,
                or concept.
            page_id: UUID of the source page (provided in the task prompt).
            context_text: Short excerpt (1-3 sentences) showing how the entity
                appears on the page.

        Returns:
            Confirmation with the entity ID and whether it is new or existing.
        """
        existing = db.get_entity_by_name(name)
        is_new = existing is None

        entity_id = db.upsert_entity(name=name, entity_type=entity_type)

        page = db.get_page(page_id)
        document_id = page["document_id"] if page else ""
        db.add_entity_mention(
            entity_id=entity_id,
            page_id=page_id,
            document_id=document_id,
            context_text=context_text,
        )

        status = "NEW entity created" if is_new else "EXISTING entity — mention added"
        return (
            f"{status}\n"
            f"Entity ID:   {entity_id}\n"
            f"Name:        {name}\n"
            f"Type:        {entity_type}\n"
            f"Page ID:     {page_id}\n"
        )

    @tool
    def search_entities(query: str, limit: int = 10) -> str:
        """Search for entities already tracked in the database by name.

        Use this before saving a new entity to avoid duplicates, and to check
        how many times an entity has already been mentioned.

        Args:
            query: Name or partial name to search for.
            limit: Maximum number of results (default 10).

        Returns:
            Formatted list of matching entities with IDs and types.
        """
        results = db.search_entities(query, limit=limit)
        if not results:
            return f"No entities found matching: {query!r}"
        lines = [f"=== {len(results)} entity match(es) for {query!r} ===\n"]
        for e in results:
            lines.append(
                f"ID:   {e.get('id')}\n"
                f"Name: {e.get('name')}\n"
                f"Type: {e.get('entity_type')}\n"
            )
        return "\n".join(lines)

    @tool
    def get_entities_for_page(page_id: str) -> str:
        """List all entities that have already been extracted from a specific page.

        Use this to avoid re-extracting entities from a page that has already
        been partially processed.

        Args:
            page_id: UUID of the page.

        Returns:
            Formatted list of entities with their types and context excerpts.
        """
        entities = db.get_entities_for_page(page_id)
        if not entities:
            return f"No entities recorded for page {page_id!r} yet."
        lines = [f"=== {len(entities)} entity/entities for page {page_id!r} ===\n"]
        for e in entities:
            lines.append(
                f"ID:      {e.get('id')}\n"
                f"Name:    {e.get('name')}\n"
                f"Type:    {e.get('entity_type')}\n"
                f"Context: {e.get('context_text', '')}\n"
            )
        return "\n".join(lines)

    @tool
    def get_entity_mentions(entity_id: str) -> str:
        """Retrieve all source pages where a given entity has been mentioned.

        Use this to gather supporting material from across the corpus for an
        entity that appears on multiple pages.

        Args:
            entity_id: UUID of the entity.

        Returns:
            Formatted list of mentions with document, page number, and context.
        """
        mentions = db.get_entity_mentions(entity_id)
        if not mentions:
            return f"No mentions found for entity {entity_id!r}."
        lines = [f"=== {len(mentions)} mention(s) for entity {entity_id!r} ===\n"]
        for m in mentions:
            doc = db.get_document(m.get("document_id", ""))
            doc_title = doc.get("title", m.get("document_id", "?")) if doc else "?"
            lines.append(
                f"Document: {doc_title}\n"
                f"Page:     {m.get('page_number', '?')}\n"
                f"Page ID:  {m.get('page_id')}\n"
                f"Context:  {m.get('context_text', '')}\n"
            )
        return "\n".join(lines)

    return [save_entity, search_entities, get_entities_for_page, get_entity_mentions]
