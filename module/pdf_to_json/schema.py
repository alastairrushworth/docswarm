"""Source of truth for the output schema. Refine here, not in DESIGN.md."""
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel


class Issue(BaseModel):
    date: date                # ISO; example: 1892-06-03
    volume: int
    number: int


class Publisher(BaseModel):
    name: str
    address: str


class Cost(BaseModel):
    issue:      Optional[str] = None
    annual:     Optional[str] = None
    semiannual: Optional[str] = None


class MagazineMeta(BaseModel):
    editor:    str
    issue:     Issue
    publisher: Publisher
    cost:      Cost


class Article(BaseModel):
    title: str
    text:  list[str]
    pages: list[int]
    kind:  Literal["prose", "verse"] = "prose"


class Document(BaseModel):
    magazine: MagazineMeta
    articles: list[Article]
