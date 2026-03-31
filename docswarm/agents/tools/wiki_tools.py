"""LangChain tools wrapping the Wiki.js client for use by the Editor agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import tool

from docswarm.logger import get_logger

if TYPE_CHECKING:
    from docswarm.storage.database import DatabaseManager
    from docswarm.wiki.client import WikiJSClient

log = get_logger(__name__)


def create_wiki_tools(wiki_client: "WikiJSClient", db: "DatabaseManager") -> list:
    """Create a list of wiki-management LangChain tools.

    The tools wrap :class:`~docswarm.wiki.client.WikiJSClient` and also call
    :meth:`~docswarm.storage.database.DatabaseManager.upsert_wiki_article` so
    the local database tracks every page that is created or updated.

    Args:
        wiki_client: An initialised :class:`~docswarm.wiki.client.WikiJSClient`.
        db: An initialised :class:`~docswarm.storage.database.DatabaseManager`.

    Returns:
        List of LangChain tool callables.
    """

    @tool
    def list_wiki_pages() -> str:
        """List all pages currently published on the wiki.

        Returns:
            Formatted list of pages with their IDs, titles, and paths.
        """
        try:
            pages = wiki_client.list_pages()
        except Exception as exc:
            return f"Error listing wiki pages: {exc}"

        if not pages:
            return "No pages found on the wiki."

        lines = [f"=== {len(pages)} wiki page(s) ===\n"]
        for page in pages:
            lines.append(
                f"ID:      {page.get('id')}\n"
                f"Title:   {page.get('title')}\n"
                f"Path:    {page.get('path')}\n"
                f"Updated: {page.get('updatedAt', 'N/A')}\n"
            )
        return "\n".join(lines)

    @tool
    def get_wiki_page(page_id: int) -> str:
        """Retrieve the full content of a wiki page by its integer ID.

        Args:
            page_id: Wiki.js numeric page ID.

        Returns:
            Formatted page data including title, path, description, and content.
        """
        try:
            page = wiki_client.get_page(page_id)
        except Exception as exc:
            return f"Error fetching wiki page {page_id}: {exc}"

        if not page:
            return f"Page {page_id} not found."

        lines = [
            f"=== Wiki page {page_id} ===",
            f"Title:       {page.get('title')}",
            f"Path:        {page.get('path')}",
            f"Description: {page.get('description', 'N/A')}",
            "",
            "--- Content ---",
            page.get("content", ""),
        ]
        return "\n".join(lines)

    @tool
    def create_wiki_page(title: str, content: str, path: str, description: str = "") -> str:
        """Create a new wiki page and record it in the local database.

        The page path should follow the pattern ``topic/subtopic`` using
        lowercase letters and hyphens in place of spaces.

        Args:
            title: Page title.
            content: Markdown content for the page.
            path: URL path for the page (e.g. ``"history/victorian-era"``).
            description: Short description shown in search results.

        Returns:
            Confirmation string with the new page ID and path, or an error
            message.
        """
        try:
            page = wiki_client.create_page(
                title=title, content=content, path=path, description=description
            )
        except Exception as exc:
            return f"Error creating wiki page: {exc}"

        wiki_page_id = page.get("id")

        # Track in local DB
        try:
            db.upsert_wiki_article(
                {
                    "wiki_page_id": wiki_page_id,
                    "title": title,
                    "path": path,
                    "content": content,
                }
            )
        except Exception:
            log.debug("Failed to track wiki page in local DB", exc_info=True)

        return (
            f"Wiki page created successfully.\n"
            f"ID:    {wiki_page_id}\n"
            f"Title: {page.get('title')}\n"
            f"Path:  {page.get('path')}"
        )

    @tool
    def update_wiki_page(page_id: int, title: str, content: str, description: str = "") -> str:
        """Update an existing wiki page and refresh the local database record.

        Args:
            page_id: Wiki.js numeric page ID of the page to update.
            title: New title.
            content: New markdown content.
            description: New description.

        Returns:
            Confirmation string or an error message.
        """
        try:
            page = wiki_client.update_page(
                page_id=page_id,
                title=title,
                content=content,
                description=description,
            )
        except Exception as exc:
            return f"Error updating wiki page {page_id}: {exc}"

        # Refresh local DB record
        try:
            db.upsert_wiki_article(
                {
                    "wiki_page_id": page_id,
                    "title": title,
                    "path": page.get("path", ""),
                    "content": content,
                }
            )
        except Exception:
            log.debug("Failed to track wiki page update in local DB", exc_info=True)

        return (
            f"Wiki page updated successfully.\n"
            f"ID:    {page.get('id')}\n"
            f"Title: {page.get('title')}\n"
            f"Path:  {page.get('path')}"
        )

    @tool
    def delete_wiki_page(page_id: int) -> str:
        """Delete a wiki page by its integer ID.

        Args:
            page_id: Wiki.js numeric page ID of the page to delete.

        Returns:
            Confirmation message or an error description.
        """
        try:
            wiki_client.delete_page(page_id)
        except Exception as exc:
            return f"Error deleting wiki page {page_id}: {exc}"
        return f"Wiki page {page_id} deleted successfully."

    @tool
    def search_wiki(query: str) -> str:
        """Search the wiki for pages matching a query string.

        Args:
            query: Search query.

        Returns:
            Formatted list of matching pages, or a message if none found.
        """
        try:
            results = wiki_client.search_pages(query)
        except Exception as exc:
            return f"Error searching wiki: {exc}"

        if not results:
            return f"No wiki pages found matching: {query!r}"

        lines = [f"=== {len(results)} result(s) for {query!r} ===\n"]
        for page in results:
            lines.append(
                f"ID:          {page.get('id')}\n"
                f"Title:       {page.get('title')}\n"
                f"Path:        {page.get('path')}\n"
                f"Description: {page.get('description', 'N/A')}\n"
            )
        return "\n".join(lines)

    read_tools = [list_wiki_pages, get_wiki_page, search_wiki]
    write_tools = [create_wiki_page, update_wiki_page, delete_wiki_page]
    return read_tools + write_tools


def create_wiki_read_tools(wiki_client: "WikiJSClient") -> list:
    """Return only the read-only wiki tools (list, get, search).

    Use this for agents that need to check wiki coverage but must not
    create or modify pages.

    Args:
        wiki_client: An initialised :class:`~docswarm.wiki.client.WikiJSClient`.

    Returns:
        List of read-only LangChain tool callables.
    """
    # Use a no-op db stub since read tools don't touch the database
    all_tools = create_wiki_tools(wiki_client, db=None)
    read_names = {"list_wiki_pages", "get_wiki_page", "search_wiki"}
    return [t for t in all_tools if t.name in read_names]
