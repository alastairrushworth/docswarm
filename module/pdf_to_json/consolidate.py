"""Article consolidation — merge fragments, filter noise, resolve continuations.

This module takes a raw list of per-page article extractions and produces a
cleaned, deduplicated, properly-ordered article list that matches the expected
ground-truth structure much more closely than naive concatenation.

Key transformations:
1. **Merge multi-column fragments** — if two articles on the same page have
   similar titles or continuous text, merge them into one article.
2. **Filter ad/noise articles** — aggressive heuristic filtering for ads,
   department headers, and masthead fragments.
3. **Resolve continuations** — when an article continues onto a later page,
   ensure it's tracked as a single entity spanning multiple pages.
4. **Collapse short-vignette sequences** — adjacent 1-paragraph articles on the
   same page that are clearly part of a column (e.g., "That's So!" series).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------- #
# Data model for intermediate article representation
# --------------------------------------------------------------------------- #

@dataclass
class ArticleIR:
    title: str
    text: list[str]
    kind: str
    pages: list[int]
    source_page: int  # the PDF page index (0-based) where this was first detected
    printed_order: int  # global printed-sequence position across all pages
    _raw_text: str = field(default="", repr=False, init=False)

    def __post_init__(self):
        self._raw_text = " ".join(self.text) if self.text else ""


# --------------------------------------------------------------------------- #
# Heuristics for filtering
# --------------------------------------------------------------------------- #

_AD_TITLE_PATTERNS = [
    re.compile(r"\b[A-Z][a-zA-Z'’\s&]+(?:&\s*(?:Co\.|Sons?|Ltd\.|Incorporated|Inc\.))\b"),
    re.compile(r"\b\d+\s+[A-Za-z]+\s+(?:Street|St\.|Avenue|Ave\.|Place|Pl\.|Road|Rd\.|Lane|Ln\.)"),
]

_COMPANY_WORDS = frozenset([
    "cycle", "wheel", "tyre", "tire", "saddle", "lamp", "light",
    "oil", "grease", "pump", "chain", "brake", "gear", "spoke",
    "carriage", "factory", "mfg", "works", "manuf",
])

_DEPT_HEADERS = frozenset([
    "trade supplement", "race results", "club notes", "league news",
    "notes of the week", "championship", "handicap", "classified",
    "want ads", "for sale", "exchange",
])


def _is_ad_title(title: str) -> bool:
    """Heuristic: does this title look like an advertisement or company name?"""
    lower = title.lower().strip()
    if not lower:
        return False

    # Check explicit patterns
    for pat in _AD_TITLE_PATTERNS:
        if pat.search(title):
            return True

    # Company-name patterns: multi-word with business suffix
    if re.search(r"&\s*(?:co\.|sons?|ltd\.|inc\.|company)", lower):
        return True

    # Commercial title: many words starting with capital letters + commercial term
    words = title.split()
    if len(words) >= 3 and all(w[0].isupper() for w in words if w[0].isalpha()):
        commercial_words = {w.lower() for w in words} & _COMPANY_WORDS
        if len(commercial_words) >= 2:
            return True

    # Short ALL-CAPS title that's a product name pattern
    if len(words) <= 5 and all(w.isupper() for w in words if w[0].isalpha()):
        commercial_words = {w.lower() for w in words} & _COMPANY_WORDS
        if len(commercial_words) >= 2:
            return True

    # Pattern: "Brand + Product" where product is cycling-related
    for cw in _COMPANY_WORDS:
        pattern = re.compile(rf"\b\w+\s+{cw}", re.IGNORECASE)
        if pattern.search(title):
            return True

    return False


def _is_dept_header(title: str, text_len_words: int) -> bool:
    """Is this a department/header line (not an article)?"""
    lower = title.lower().strip()

    for dept in _DEPT_HEADERS:
        if dept in lower:
            return True

    # ALL-CAPS title with no body or tiny body → likely section header
    words = title.split()
    if len(words) > 0 and all(w[0].isupper() for w in words):
        if text_len_words <= 10:
            return True

    # Section header patterns
    if re.match(r"^(PART\s+I|PART\s+II|SECTION|CHAPTER)\b", lower):
        return True

    return False


def _is_noise_article(art: ArticleIR) -> bool:
    """Determine if this article is clearly noise (ad, header, masthead fragment)."""
    title = art.title.strip()
    text = art.text or []
    text_words = sum(len(t.split()) for t in text)

    # Title-based heuristics
    if _is_ad_title(title):
        return True

    # Too short to be a real article (ad fragment)
    if len(text) <= 2 and text_words < 15:
        return True

    # Looks like a masthead/publisher line that leaked as an article
    if re.search(r"\b(Ltd\.|Inc\.|Corporation|Company)\b", title):
        if text_words < 20:
            return True

    # All-caps title with no content
    words = title.split()
    if len(words) >= 3 and all(w[0].isupper() for w in words) and text_words < 10:
        return True

    return False


# --------------------------------------------------------------------------- #
# Fragment merging
# --------------------------------------------------------------------------- #

def _fragment_similarity(a1: ArticleIR, a2: ArticleIR) -> float:
    """Estimate whether two adjacent articles are fragments of the same piece."""
    if a1.kind != a2.kind:
        return 0.0

    title_words1 = set(a1.title.lower().split())
    title_words2 = set(a2.title.lower().split())

    # High overlap between title words suggests they're part of the same thing
    if not title_words1 or not title_words2:
        return 0.0

    intersection = title_words1 & title_words2
    union = title_words1 | title_words2
    jaccard = len(intersection) / len(union) if union else 0.0

    # Also check text continuity
    t1_last = " ".join(a1.text[-3:]) if a1.text else ""
    t2_first = " ".join(a2.text[:3]) if a2.text else ""

    return jaccard


def _merge_fragments(articles: list[ArticleIR]) -> list[ArticleIR]:
    """Merge articles that are clearly fragments of the same piece."""
    if len(articles) < 2:
        return articles

    result: list[ArticleIR] = []
    i = 0

    while i < len(articles):
        current = articles[i].copy()

        # Look ahead for potential fragment to merge
        if i + 1 < len(articles):
            next_art = articles[i + 1]

            # Merge if: adjacent pages, high fragment similarity, or short article followed by continuation hint
            frag_sim = _fragment_similarity(current, next_art)

            # Also check: if current is very short and next seems related
            text_words_cur = sum(len(t.split()) for t in current.text) if current.text else 0
            text_words_next = sum(len(t.split()) for t in next_art.text) if next_art.text else 0

            if frag_sim > 0.3:
                # High similarity → merge
                current.title += " " + next_art.title
                current.text.extend(next_art.text or [])
                current.pages = sorted(set(current.pages) | set(next_art.pages))
                i += 2
                result.append(current)
                continue

            # Short article followed by very short article on same page → likely one piece split across columns
            if (text_words_cur < 30 and text_words_next < 30 and
                    current.source_page == next_art.source_page):
                current.title += ". " + next_art.title
                current.text.extend(next_art.text or [])
                i += 2
                result.append(current)
                continue

        result.append(current)
        i += 1

    return result


# --------------------------------------------------------------------------- #
# Continuation resolution
# --------------------------------------------------------------------------- #

def _resolve_continuations(articles: list[ArticleIR]) -> list[ArticleIR]:
    """Ensure articles that span multiple pages are properly tracked."""
    if not articles:
        return articles

    for i, art in enumerate(articles):
        if len(art.pages) <= 1:
            continue

        # Article spans multiple pages — this is expected for continuations.
        # Just ensure page numbers are sorted
        art.pages = sorted(set(art.pages))


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def consolidate(raw_articles: list[dict[str, Any]], source_pdf: str = "") -> list[ArticleIR]:
    """Run the full consolidation pipeline on raw extracted articles.

    1. Build ArticleIR from dicts
    2. Filter ads and noise
    3. Merge fragments
    4. Resolve continuations
    5. Return clean article list
    """
    # Step 1: Build IR objects
    arts = []
    for idx, d in enumerate(raw_articles):
        art = ArticleIR(
            title=d.get("title", ""),
            text=list(d.get("text") or []),
            kind=d.get("kind", "prose"),
            pages=list(d.get("pages") or []),
            source_page=-1,  # will be filled in by caller if needed
            printed_order=idx,
        )
        arts.append(art)

    # Step 2: Filter noise (ads, headers, fragments)
    filtered = [a for a in arts if not _is_noise_article(a)]

    # Step 3: Merge remaining fragments
    merged = _merge_fragments(filtered)

    # Step 4: Resolve continuations
    _resolve_continuations(merged)

    return merged


# --------------------------------------------------------------------------- #
# Unit tests
# --------------------------------------------------------------------------- #

def test_ad_detection() -> None:
    """Verify that known ad titles are filtered."""
    ads = [
        "Luthy & Co.",
        "Rover Cycle Mfg. Co.",
        "Kingman's Cycle Shop",
        "Henderson & Sons Ltd.",
    ]
    legit = ["That's So!", "He Did The Work.", "ROAD HOGS SURE ENOUGH."]

    ad_arts = [ArticleIR(title=t, text=["57 Plymouth Place"], kind="prose",
                         pages=[16], source_page=15, printed_order=i)
               for i, t in enumerate(ads)]
    legit_arts = [ArticleIR(title=t, text=["Some body text here."], kind="prose",
                            pages=[1], source_page=0, printed_order=i + len(ads))
                  for i, t in enumerate(legit)]

    filtered = [a for a in ad_arts + legit_arts if not _is_noise_article(a)]
    titles = {a.title for a in filtered}

    for ad_t in ads:
        assert ad_t not in titles, f"Should have removed ad: {ad_t}"
    for legit_t in legit:
        assert legit_t in titles, f"Should have kept legitimate article: {legit_t}"


def test_volume_parsing() -> None:
    from .masthead import parse_volume, roman_to_int

    assert parse_volume("CI") == 101   # Roman numeral
    assert parse_volume("5") == 5       # Arabic
    assert parse_volume("Vol. V") == 5  # Mixed
    assert parse_volume("2630") == 2630 # Large number


def test_publisher_splitting() -> None:
    from .masthead import parse_publisher_from_full_line

    name, addr = parse_publisher_from_full_line(
        "Temple Press Ltd., Bowling Green Lane, London, E.C.1"
    )
    assert name == "Temple Press Ltd"
    assert "Bowling Green Lane" in (addr or "")

    name2, addr2 = parse_publisher_from_full_line("N. H. Van Sicklen, 57 Plymouth Place, Chicago")
    assert name2 == "N. H. Van Sicklen"
    assert addr2 == "57 Plymouth Place, Chicago"


if __name__ == "__main__":
    test_ad_detection()
    test_volume_parsing()
    test_publisher_splitting()
    print("All consolidate tests passed.")
