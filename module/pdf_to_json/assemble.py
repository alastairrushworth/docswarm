"""Assemble per-page intermediate representations into a schema-conformant Document."""
from __future__ import annotations

from datetime import date
from typing import Any

from .schema import Article, Cost, Document, Issue, MagazineMeta, Publisher


def _str(x: Any, default: str = "") -> str:
    return x if isinstance(x, str) else default


def _int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _to_date(x: Any) -> date:
    if isinstance(x, date):
        return x
    if isinstance(x, str):
        try:
            return date.fromisoformat(x)
        except ValueError:
            pass
    return date(1900, 1, 1)


def _meta_from(ir_meta: dict[str, Any]) -> MagazineMeta:
    issue_d = ir_meta.get("issue") or {}
    pub_d = ir_meta.get("publisher") or {}
    cost_d = ir_meta.get("cost") or {}
    return MagazineMeta(
        editor=_str(ir_meta.get("editor")),
        issue=Issue(
            date=_to_date(issue_d.get("date")),
            volume=_int(issue_d.get("volume")),
            number=_int(issue_d.get("number")),
        ),
        publisher=Publisher(
            name=_str(pub_d.get("name")),
            address=_str(pub_d.get("address")),
        ),
        cost=Cost(
            issue=cost_d.get("issue"),
            annual=cost_d.get("annual"),
            semiannual=cost_d.get("semiannual"),
        ),
    )


def _article_from(a: dict[str, Any]) -> Article | None:
    title = _str(a.get("title")).strip()
    if not title:
        return None
    text = a.get("text") or []
    if isinstance(text, str):
        text = [text]
    text = [t for t in (str(s) for s in text) if t.strip()]
    pages = a.get("pages") or []
    if isinstance(pages, int):
        pages = [pages]
    pages = [int(p) for p in pages if isinstance(p, (int, str)) and str(p).lstrip("-").isdigit()]
    kind = a.get("kind", "prose")
    if kind not in ("prose", "verse"):
        kind = "prose"
    return Article(title=title, text=text, pages=pages or [1], kind=kind)


def assemble(ir_meta: dict[str, Any], ir_articles: list[dict[str, Any]]) -> Document:
    """Build a Document from intermediate representations.

    Articles are ordered as supplied (caller is responsible for printed-sequence order).
    """
    meta = _meta_from(ir_meta or {})
    articles: list[Article] = []
    for a in ir_articles or []:
        art = _article_from(a or {})
        if art is not None:
            articles.append(art)
    return Document(magazine=meta, articles=articles)


def empty_document() -> Document:
    """Best-effort fallback when extraction produced nothing."""
    return Document(
        magazine=MagazineMeta(
            editor="",
            issue=Issue(date=date(1900, 1, 1), volume=0, number=0),
            publisher=Publisher(name="", address=""),
            cost=Cost(),
        ),
        articles=[],
    )
