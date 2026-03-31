"""Sync local markdown articles in wiki/ to a live Wiki.js instance.

Reads all .md files under DOCSWARM_WIKI_OUTPUT_DIR, parses their YAML front
matter, and creates or updates the corresponding Wiki.js pages via the
GraphQL API.  After a successful sync the wiki_page_id is written back into
each file's front matter so subsequent runs perform updates rather than creates.

Usage:
    python3 sync_wiki.py

Requires DOCSWARM_WIKI_URL and DOCSWARM_WIKI_API_KEY to be set in .env or
the environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    from docswarm.config import Config
    from docswarm.wiki.client import WikiJSClient
    from docswarm.agents.tools.file_tools import _build_front_matter, _parse_front_matter
    from docswarm.logger import get_logger

    log = get_logger(__name__)
    config = Config.from_env()

    if not config.wiki_url:
        print("ERROR: DOCSWARM_WIKI_URL is not set in .env")
        sys.exit(1)
    if not config.wiki_api_key:
        print("ERROR: DOCSWARM_WIKI_API_KEY is not set in .env")
        sys.exit(1)

    root = Path(config.wiki_output_dir)
    if not root.exists():
        print(f"No wiki output directory found at: {root}")
        sys.exit(0)

    files = sorted(root.glob("**/*.md"))
    if not files:
        print(f"No markdown files found in {root}")
        sys.exit(0)

    print(f"Found {len(files)} article file(s) in {root}")
    wiki = WikiJSClient(base_url=config.wiki_url, api_key=config.wiki_api_key)

    stats = {"created": 0, "updated": 0, "skipped": 0, "errors": 0}

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        meta, body = _parse_front_matter(text)

        title = meta.get("title") or file_path.stem.replace("-", " ").title()
        description = meta.get("description", "")
        wiki_page_id = meta.get("wiki_page_id")
        rel_path = str(file_path.relative_to(root).with_suffix(""))

        try:
            if wiki_page_id:
                # Update existing page
                log.info("Updating: %s (wiki_page_id=%s)", rel_path, wiki_page_id)
                wiki.update_page(
                    page_id=int(wiki_page_id),
                    title=title,
                    content=body,
                    description=description,
                )
                print(f"  UPDATED  {rel_path}")
                stats["updated"] += 1
            else:
                # Create new page
                log.info("Creating: %s", rel_path)
                page = wiki.create_page(
                    title=title,
                    content=body,
                    path=rel_path,
                    description=description,
                )
                new_id = page.get("id")
                print(f"  CREATED  {rel_path}  (wiki_page_id={new_id})")
                stats["created"] += 1

                # Write wiki_page_id back into front matter
                meta["wiki_page_id"] = new_id
                file_path.write_text(_build_front_matter(meta) + body, encoding="utf-8")

        except Exception as exc:
            log.error("Failed to sync %s: %s", rel_path, exc)
            print(f"  ERROR    {rel_path}: {exc}")
            stats["errors"] += 1

    wiki.close()

    print(
        f"\nSync complete — created: {stats['created']}, "
        f"updated: {stats['updated']}, "
        f"errors: {stats['errors']}"
    )
