"""Deterministic post-filter that redacts hints overlapping with ground truth.

The judge prompt is the first line of defense; this is the safety net.
"""
from __future__ import annotations

import re
from typing import Iterable

WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _ngrams(tokens: list[str], n: int) -> set[str]:
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i:i + n]).lower() for i in range(len(tokens) - n + 1)}


def _tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text)


def _truth_phrases(truth_strings: Iterable[str], n: int = 4) -> set[str]:
    out: set[str] = set()
    for s in truth_strings:
        if not s:
            continue
        toks = _tokenize(str(s))
        out |= _ngrams(toks, n)
    return out


def overlap_ratio(hint: str, truth_strings: Iterable[str], n: int = 4) -> float:
    """Fraction of n-grams in `hint` that appear verbatim in any truth string."""
    hint_tokens = _tokenize(hint)
    hint_ngrams = _ngrams(hint_tokens, n)
    if not hint_ngrams:
        return 0.0
    truth = _truth_phrases(truth_strings, n)
    if not truth:
        return 0.0
    overlap = hint_ngrams & truth
    return len(overlap) / len(hint_ngrams)


def filter_hints(
    hints: list[str],
    truth_strings: Iterable[str],
    threshold: float,
) -> list[str]:
    """Drop hints whose overlap with ground truth exceeds threshold."""
    truth_strings = list(truth_strings)
    out: list[str] = []
    for h in hints:
        if overlap_ratio(h, truth_strings) >= threshold:
            out.append("[redacted: overlapped with ground truth]")
        else:
            out.append(h)
    return out


def collect_truth_strings(truth: dict) -> list[str]:
    """Extract every string from a ground-truth Document dict for overlap checks."""
    out: list[str] = []

    def walk(x):
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(truth)
    return out
