"""Tests for docswarm.agents.tools.entity_tools.

Uses the real db fixture backed by a temporary DuckLake database.
"""

from __future__ import annotations

import pytest

from docswarm.agents.tools.entity_tools import create_entity_tools


@pytest.fixture
def entity_tools(db):
    return create_entity_tools(db)


@pytest.fixture
def tools_by_name(entity_tools):
    return {t.name: t for t in entity_tools}


# ---------------------------------------------------------------------------
# save_entity
# ---------------------------------------------------------------------------

class TestSaveEntityTool:
    def test_creates_new_entity_on_first_call(self, tools_by_name, sample_page, db):
        result = tools_by_name["save_entity"].invoke({
            "name": "Thomas Chippendale",
            "entity_type": "person",
            "page_id": sample_page["id"],
            "context_text": "Famous English cabinet-maker.",
        })
        assert "NEW entity created" in result
        entity = db.get_entity_by_name("Thomas Chippendale")
        assert entity is not None

    def test_reuses_existing_entity_on_second_call(self, tools_by_name, sample_page, db):
        # First call creates the entity
        tools_by_name["save_entity"].invoke({
            "name": "Windsor Castle",
            "entity_type": "place",
            "page_id": sample_page["id"],
        })
        # Second call with same name should reuse
        result = tools_by_name["save_entity"].invoke({
            "name": "Windsor Castle",
            "entity_type": "place",
            "page_id": sample_page["id"],
        })
        assert "EXISTING entity" in result

        # Should still be only one entity row
        results = db.search_entities("Windsor Castle")
        assert len(results) == 1

    def test_always_creates_a_new_mention(self, tools_by_name, sample_page, db):
        """Even on the second call for the same entity, a new mention row is created."""
        invoke_args = {
            "name": "Georgian Period",
            "entity_type": "event",
            "page_id": sample_page["id"],
            "context_text": "Georgian period furniture.",
        }
        tools_by_name["save_entity"].invoke(invoke_args)
        tools_by_name["save_entity"].invoke(invoke_args)

        entity = db.get_entity_by_name("Georgian Period")
        assert entity is not None
        mentions = db.get_entity_mentions(entity["id"])
        assert len(mentions) == 2

    def test_result_contains_entity_id(self, tools_by_name, sample_page):
        result = tools_by_name["save_entity"].invoke({
            "name": "Mahogany",
            "entity_type": "object",
            "page_id": sample_page["id"],
        })
        assert "Entity ID:" in result

    def test_result_contains_name_and_type(self, tools_by_name, sample_page):
        result = tools_by_name["save_entity"].invoke({
            "name": "London",
            "entity_type": "place",
            "page_id": sample_page["id"],
        })
        assert "London" in result
        assert "place" in result


# ---------------------------------------------------------------------------
# search_entities
# ---------------------------------------------------------------------------

class TestSearchEntitiesTool:
    def test_finds_entity_by_partial_name(self, tools_by_name, sample_page, db):
        db.upsert_entity("Thomas Chippendale", "person")
        db.upsert_entity("Thomas Sheraton", "person")

        result = tools_by_name["search_entities"].invoke({"query": "Thomas"})
        assert "Chippendale" in result
        assert "Sheraton" in result

    def test_returns_no_match_message_when_empty(self, tools_by_name, db):
        result = tools_by_name["search_entities"].invoke({"query": "xyznotentity99"})
        assert "No entities found" in result

    def test_result_includes_entity_type(self, tools_by_name, db):
        db.upsert_entity("Crystal Palace", "place")
        result = tools_by_name["search_entities"].invoke({"query": "Crystal"})
        assert "place" in result

    def test_result_includes_entity_id(self, tools_by_name, db):
        db.upsert_entity("Walnut Wood", "object")
        result = tools_by_name["search_entities"].invoke({"query": "Walnut"})
        assert "ID:" in result


# ---------------------------------------------------------------------------
# get_entities_for_page
# ---------------------------------------------------------------------------

class TestGetEntitiesForPageTool:
    def test_returns_entities_for_page(self, tools_by_name, sample_page, sample_document, db):
        eid = db.upsert_entity("Mahogany Chair", "object")
        db.add_entity_mention(eid, sample_page["id"], sample_document["id"], "chair context")

        result = tools_by_name["get_entities_for_page"].invoke({"page_id": sample_page["id"]})
        assert "Mahogany Chair" in result

    def test_returns_empty_message_when_no_entities(self, tools_by_name, sample_page):
        result = tools_by_name["get_entities_for_page"].invoke({"page_id": sample_page["id"]})
        assert "No entities recorded" in result

    def test_result_includes_entity_type(self, tools_by_name, sample_page, sample_document, db):
        eid = db.upsert_entity("Victorian Era", "event")
        db.add_entity_mention(eid, sample_page["id"], sample_document["id"], "event context")
        result = tools_by_name["get_entities_for_page"].invoke({"page_id": sample_page["id"]})
        assert "event" in result

    def test_result_includes_context_text(self, tools_by_name, sample_page, sample_document, db):
        eid = db.upsert_entity("Inlay Work", "concept")
        db.add_entity_mention(
            eid, sample_page["id"], sample_document["id"], "Inlay work was popular."
        )
        result = tools_by_name["get_entities_for_page"].invoke({"page_id": sample_page["id"]})
        assert "Inlay work was popular." in result


# ---------------------------------------------------------------------------
# get_entity_mentions
# ---------------------------------------------------------------------------

class TestGetEntityMentionsTool:
    def test_returns_all_mentions(self, tools_by_name, sample_page, sample_document, db):
        # Add second page
        page2_id = db.insert_page({
            "document_id": sample_document["id"],
            "page_number": 2,
            "width_pts": 595.0, "height_pts": 842.0,
            "image_path": "/tmp/p2.png", "raw_text": "Second page.",
            "ocr_confidence": 90.0, "word_count": 2,
        })
        eid = db.upsert_entity("Roving Concept", "concept")
        db.add_entity_mention(eid, sample_page["id"], sample_document["id"], "Mention 1")
        db.add_entity_mention(eid, page2_id, sample_document["id"], "Mention 2")

        result = tools_by_name["get_entity_mentions"].invoke({"entity_id": eid})
        assert "Mention 1" in result
        assert "Mention 2" in result
        assert "2 mention(s)" in result

    def test_returns_no_mentions_message_when_empty(self, tools_by_name, db):
        result = tools_by_name["get_entity_mentions"].invoke({
            "entity_id": "00000000-0000-0000-0000-000000000000"
        })
        assert "No mentions found" in result

    def test_result_includes_document_title(self, tools_by_name, sample_page, sample_document, db):
        eid = db.upsert_entity("Oak Furniture", "object")
        db.add_entity_mention(eid, sample_page["id"], sample_document["id"], "Oak is strong.")
        result = tools_by_name["get_entity_mentions"].invoke({"entity_id": eid})
        assert "Test Document" in result


# ---------------------------------------------------------------------------
# Tool list completeness
# ---------------------------------------------------------------------------

class TestEntityToolsCreation:
    def test_creates_four_tools(self, db):
        tools = create_entity_tools(db)
        assert len(tools) == 4

    def test_tool_names(self, db):
        tools = create_entity_tools(db)
        names = {t.name for t in tools}
        expected = {
            "save_entity", "search_entities",
            "get_entities_for_page", "get_entity_mentions",
        }
        assert names == expected
