"""Surface-form normalization + fuzzy field similarity.

The judge should grade *information*, not transcription form. Differences in
punctuation, capitalization, whitespace, accents, and minor abbreviation/OCR
noise must not read as "wrong". These helpers collapse surface form so the
deterministic comparators can score essence. Stdlib only — no new deps.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(s: object) -> str:
    """Casefold, strip accents/punctuation, collapse whitespace.

    "L. J. Berger"            -> "l j berger"
    "57 Plymouth Place, Chi." -> "57 plymouth place chi"
    """
    if s is None:
        return ""
    text = unicodedata.normalize("NFKD", str(s))
    text = text.encode("ascii", "ignore").decode("ascii")  # drop accents
    text = text.casefold()
    text = _NON_ALNUM.sub(" ", text)
    return text.strip()


def field_similarity(a: object, b: object) -> float:
    """Form-insensitive similarity in [0, 1] for short metadata strings.

    Blends character sequence ratio with token-set Jaccard and takes the max,
    so reorderings and punctuation-only differences both score high while a
    genuinely different or truncated value still loses credit.
    """
    na, nb = normalize(a), normalize(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    seq = SequenceMatcher(None, na, nb).ratio()
    ta, tb = set(na.split()), set(nb.split())
    jac = len(ta & tb) / len(ta | tb) if (ta or tb) else 0.0
    return max(seq, jac)
