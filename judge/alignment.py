"""Hungarian alignment between predicted and truth articles."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .similarity import combined_similarity


def align_articles(
    pred: list[dict[str, Any]],
    truth: list[dict[str, Any]],
    floor: float = 0.0,
) -> list[tuple[int, int, float]]:
    """Return list of (pred_index, truth_index, similarity) for matched pairs.

    `floor` is the minimum combined similarity for a pair to count as the same
    article. Pairs below it are dropped, so a genuinely missing or extra article
    is not force-matched to an unrelated one (which would otherwise pollute the
    title/text/pages components and hide the count error).
    """
    if not pred or not truth:
        return []
    n, m = len(pred), len(truth)
    sim = np.zeros((n, m), dtype=float)
    for i, p in enumerate(pred):
        for j, t in enumerate(truth):
            sim[i, j] = combined_similarity(p, t)
    cost = 1.0 - sim
    rows, cols = linear_sum_assignment(cost)
    return [
        (int(r), int(c), float(sim[r, c]))
        for r, c in zip(rows, cols)
        if sim[r, c] >= floor
    ]
