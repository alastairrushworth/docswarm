"""Hungarian alignment between predicted and truth articles."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .similarity import combined_similarity


def align_articles(
    pred: list[dict[str, Any]],
    truth: list[dict[str, Any]],
) -> list[tuple[int, int, float]]:
    """Return list of (pred_index, truth_index, similarity) for matched pairs."""
    if not pred or not truth:
        return []
    n, m = len(pred), len(truth)
    sim = np.zeros((n, m), dtype=float)
    for i, p in enumerate(pred):
        for j, t in enumerate(truth):
            sim[i, j] = combined_similarity(p, t)
    cost = 1.0 - sim
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c), float(sim[r, c])) for r, c in zip(rows, cols)]
