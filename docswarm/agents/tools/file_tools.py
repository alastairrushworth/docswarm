"""LangChain tools for reading and writing wiki article markdown files."""

from __future__ import annotations

from pathlib import Path

import yaml
from langchain_core.tools import tool

from docswarm.logger import get_logger

log = get_logger(__name__)


def _parse_front_matter(text: str) -> tuple[dict, str]:
    """Split a markdown file into front matter dict and body.

    Args:
        text: Full file contents.

    Returns:
        ``(front_matter_dict, body)`` tuple.  If no front matter is present,
        the dict is empty and body is the full text.
    """
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                meta = {}
            return meta, parts[2].lstrip("\n")
    return {}, text


def _build_front_matter(meta: dict) -> str:
    """Render a dict as a YAML front matter block."""
    return "---\n" + yaml.dump(meta, default_flow_style=False, allow_unicode=True) + "---\n\n"


def create_file_tools(wiki_output_dir: str) -> list:
    """Create file-system tools for reading and writing markdown articles.

    All paths are relative to *wiki_output_dir* and follow the pattern
    ``entity-type/entity-name`` (e.g. ``people/thomas-chippendale``).
    The ``.md`` extension is added automatically.

    Args:
        wiki_output_dir: Root directory for generated articles.

    Returns:
        List of LangChain tool callables.
    """
    root = Path(wiki_output_dir)
    root.mkdir(parents=True, exist_ok=True)

    @tool
    def write_article_file(
        path: str,
        title: str,
        content: str,
        description: str = "",
        entity_id: str = "",
        entity_type: str = "",
        source_page_id: str = "",
    ) -> str:
        """Write a wiki article to a local markdown file.

        Creates the file (and any parent directories) at
        ``{wiki_output_dir}/{path}.md``.  If the file already exists it is
        overwritten.  The file includes YAML front matter so that sync_wiki.py
        can push it to Wiki.js later.

        Path convention: ``entity-type/entity-name`` in lowercase with hyphens,
        e.g. ``people/thomas-chippendale`` or ``events/great-exhibition-1851``.

        Args:
            path: Article path without extension (e.g. ``people/thomas-chippendale``).
            title: Article title.
            content: Full markdown body of the article.
            description: One-sentence summary shown in search results.
            entity_id: UUID of the entity in the docswarm database.
            entity_type: Entity type (person, place, event, object, etc.).
            source_page_id: UUID of the source page this article was derived from.

        Returns:
            Confirmation with the file path written.
        """
        clean_path = path.strip("/").replace(" ", "-").lower()
        file_path = (root / (clean_path + ".md")).resolve()
        if not str(file_path).startswith(str(root.resolve())):
            return f"Invalid path (escapes wiki directory): {path!r}"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "title": title,
            "description": description,
            "entity_id": entity_id,
            "entity_type": entity_type,
            "source_page_id": source_page_id,
            "wiki_page_id": None,  # populated by sync_wiki.py after publishing
        }
        full_text = _build_front_matter(meta) + content

        file_path.write_text(full_text, encoding="utf-8")
        log.info("Wrote article: %s", file_path)
        return f"Article written to: {file_path}"

    @tool
    def read_article_file(path: str) -> str:
        """Read an existing wiki article file.

        Use this to check the current content of an article before deciding
        whether it needs updating.

        Args:
            path: Article path without extension (e.g. ``people/thomas-chippendale``).

        Returns:
            The full file contents including front matter, or an error message
            if the file does not exist.
        """
        clean_path = path.strip("/").replace(" ", "-").lower()
        file_path = (root / (clean_path + ".md")).resolve()
        if not str(file_path).startswith(str(root.resolve())):
            return f"Invalid path (escapes wiki directory): {path!r}"
        if not file_path.exists():
            return f"No article file found at: {file_path}"
        text = file_path.read_text(encoding="utf-8")
        meta, body = _parse_front_matter(text)
        lines = [
            f"=== Article: {file_path} ===",
            f"Title:       {meta.get('title', 'N/A')}",
            f"Description: {meta.get('description', 'N/A')}",
            f"Entity ID:   {meta.get('entity_id', 'N/A')}",
            f"Entity type: {meta.get('entity_type', 'N/A')}",
            f"Wiki page ID:{meta.get('wiki_page_id', 'not yet synced')}",
            "",
            "--- Content ---",
            body,
        ]
        return "\n".join(lines)

    @tool
    def list_article_files() -> str:
        """List all markdown article files in the wiki output directory.

        Returns:
            Formatted list of article paths and their titles.
        """
        files = sorted(root.glob("**/*.md"))
        if not files:
            return f"No article files found in {root}."
        lines = [f"=== {len(files)} article(s) in {root} ===\n"]
        for f in files:
            text = f.read_text(encoding="utf-8")
            meta, _ = _parse_front_matter(text)
            rel = f.relative_to(root).with_suffix("")
            synced = (
                f"wiki_page_id={meta['wiki_page_id']}" if meta.get("wiki_page_id") else "not synced"
            )
            lines.append(f"{rel}  |  {meta.get('title', '(no title)')}  |  {synced}")
        return "\n".join(lines)

    @tool
    def search_article_files(query: str) -> str:
        """Search local article files for content matching a query string.

        Use this to check whether an entity already has a written article
        before creating a new one.

        Args:
            query: Keywords to search for (case-insensitive).

        Returns:
            Matching article paths and a short excerpt around the match.
        """
        query_lower = query.lower()
        files = sorted(root.glob("**/*.md"))
        matches = []
        for f in files:
            text = f.read_text(encoding="utf-8")
            meta, body = _parse_front_matter(text)
            # Search title + body
            title_prefix = meta.get("title", "") + " "
            searchable = (title_prefix + body).lower()
            if query_lower in searchable:
                rel = str(f.relative_to(root).with_suffix(""))
                # Find a short excerpt (adjust index for title prefix)
                idx = searchable.find(query_lower)
                body_idx = max(0, idx - len(title_prefix))
                excerpt = body[max(0, body_idx - 60) : body_idx + 120].replace("\n", " ").strip()
                matches.append((rel, meta.get("title", "?"), excerpt))

        if not matches:
            return f"No articles found matching: {query!r}"
        lines = [f"=== {len(matches)} article(s) matching {query!r} ===\n"]
        for path, title, excerpt in matches:
            lines.append(f"Path:    {path}\nTitle:   {title}\nExcerpt: ...{excerpt}...\n")
        return "\n".join(lines)

    return [write_article_file, read_article_file, list_article_files, search_article_files]


def create_file_read_tools(wiki_output_dir: str) -> list:
    """Return only the read-only file tools (read, list, search).

    Use this for agents that need to check existing article coverage but
    must not write files.

    Args:
        wiki_output_dir: Root directory for generated articles.

    Returns:
        List of read-only LangChain tool callables.
    """
    all_tools = create_file_tools(wiki_output_dir)
    return [t for t in all_tools if t.name != "write_article_file"]
