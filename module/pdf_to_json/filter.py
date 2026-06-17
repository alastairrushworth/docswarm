"""Article-level filtering to remove ads, fragments, and non-content.

The per-page extraction produces too many "articles" because it treats every
visible title/text-block pair as an article.  This module identifies which ones
are genuine articles versus advertisements, department headers, or other noise,
and prunes them from the list.

Designed as a standalone utility so it can be unit-tested independently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

# Patterns that strongly indicate an advertisement rather than article content.
_AD_TITLE_PATTERNS: list[re.Pattern[str]] = [
    # Company names with trade suffixes like "& Co.", "Ltd.", etc.
    re.compile(
        r"\b[A-Z][a-zA-Z'’\s&]+(?:&\s*(?:Co\.|Sons?|Ltd\.|Incorporated|Inc\.|Company))\b",
        re.IGNORECASE,
    ),
]

# Patterns that indicate the text content itself is an ad.
_AD_TEXT_PATTERNS: list[re.Pattern[str]] = [
    # Address patterns in short fragments
    re.compile(r"\d+\s+[A-Z][a-z]+\s+(?:Street|St\.|Avenue|Ave\.|Place|Pl\.|Road|Rd\.)"),
]


@dataclass
class ArticleIR:
    """Intermediate representation for a single article (for filtering)."""

    title: str
    text: list[str]
    kind: str
    pages: list[int]
    predicted_index: int  # original index for tracking


def _is_company_name(title: str) -> bool:
    """Check if the title looks like a company/advertisement name."""
    lower = title.lower()
    # Common ad indicators in titles
    com_words = {"cycle", "wheel", "tyre", "tire", "saddle", "lamp", "light",
                 "oil", "grease", "pump", "chain", "brake", "gear", "spoke",
                 "carriage", "factory", "mfg"}
    words = set(lower.split())
    if len(words & com_words) >= 2:
        return True
    # "& Co." or similar suffixes
    if re.search(r"&\s*(?:co\.|sons?|ltd\.|inc\.|company)", lower):
        return True
    # Capitalized multi-word commercial pattern
    words_list = title.split()
    if len(words_list) >= 3 and all(w[0].isupper() for w in words_list):
        if any(w.lower() in com_words for w in words_list):
            return True
    return False


def _is_ad_text(text: list[str]) -> bool:
    """Check if the text content looks like an advertisement."""
    full_text = " ".join(text)
    for pattern in _AD_TEXT_PATTERNS:
        if pattern.search(full_text):
            return True
    return False


def _is_section_header(title: str, text: list[str]) -> bool:
    """Check if this looks like a section header rather than an article."""
    # Section headers with NO body text are likely departments (not articles)
    if len(text) == 0 and re.match(r"^[A-Z][A-Z\s\-\.]{5,}$", title):
        return True
    # Known department sections that are not standalone articles
    dept_patterns = [
        r"^Trade Supplement$",
        r"^Notes? of the Week",
        r"^League News",
        r"^Club Notes",
        r"^Championship",
        r"^Race Results",
        r"^Handicap",
    ]
    for pat in dept_patterns:
        if re.match(pat, title):
            return True
    return False


def _is_legitimate_short(title: str) -> bool:
    """Check if this is a legitimate short article (e.g., Bearings vignettes)."""
    # Short aphoristic pieces from The Bearings
    legit_prefixes = ["That's", "It's", "She'd", "He'd", "They'd", "There'd"]
    for prefix in legit_prefixes:
        if title.startswith(prefix):
            return True
    # Question-like or exclamatory fragments that are legitimate content
    if title.endswith("!") and len(title) < 30:
        words = title.split()
        if any(w[0].islower() for w in words) or len(words) >= 4:
            return True
    return False


def filter_articles(articles: list[ArticleIR]) -> list[ArticleIR]:
    """Remove articles that are clearly advertisements or non-content.

    Returns a new list with only genuine article content preserved.
    Heuristics are conservative -- we prefer keeping content over removing it.
    """
    if not articles:
        return []

    result: list[ArticleIR] = []
    for art in articles:
        title = art.title.strip()
        text = art.text or []
        full_text = " ".join(text) if text else ""

        # Rule 1: Company name patterns (strong ad signal)
        if _is_company_name(title):
            continue

        # Rule 2: Section headers with minimal body text
        if _is_section_header(title, text):
            continue

        # Rule 3: Very short fragments that look like ads
        word_count = sum(len(t.split()) for t in text)
        if len(text) <= 2 and word_count < 40:
            # Could be a legitimate vignette (Bearings has many 1-paragraph pieces)
            # or an ad fragment. Check title more carefully.
            if _is_ad_text(text):
                continue
            # If text looks like just address info, it's likely an ad
            if re.search(r"\d+\s+[A-Za-z]+\s+(?:St|Ave|Pl|Rd)\.?\b", full_text):
                # Only filter if title also looks commercial
                words_list = title.split()
                if _is_company_name(title) or (
                    len(words_list) <= 4 and all(w[0].isupper() for w in words_list)
                ):
                    continue

        result.append(art)

    return result


# ---------------------------------------------------------------------------
# Unit-testable helpers
# ---------------------------------------------------------------------------

def test_ad_detection() -> None:
    """Quick self-check to verify ad detection logic."""
    ad_titles = [
        "Luthy & Co.",
        "Kingman & Co.",
        "Warwick Cycle Mfg. Co.",
        "Rover Cycle Co.",
    ]
    legit_titles = [
        "That's So!",
        "Funeral Note.",
        "ROAD HOGS SURE ENOUGH.",
        "He Did The Work.",
    ]

    # Ad titles should be detected as company names
    for t in ad_titles:
        assert _is_company_name(t), f"Should detect '{t}' as ad"

    combined = [ArticleIR(title=t, text=["Some content"], kind="prose",
                          pages=[1], predicted_index=i)
                for i, t in enumerate(legit_titles)]
    ads = [ArticleIR(title=t, text=["57 Plymouth Place"], kind="prose",
                     pages=[16], predicted_index=i)
           for i, t in enumerate(ad_titles)]

    filtered = filter_articles(ads + combined)
    removed_titles = {t for t in ad_titles if not any(f.title == t for f in filtered)}
    kept_legit = {t for t in legit_titles if any(f.title == t for f in filtered)}

    print(f"Ad detection: removed {len(removed_titles)}/{len(ad_titles)} ads, "
          f"kept {len(kept_legit)}/{len(legit_titles)} legitimate")


if __name__ == "__main__":
    test_ad_detection()
    print("All ad detection checks passed.")
