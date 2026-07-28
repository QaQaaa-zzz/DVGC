"""Balanced sampling utilities for conservative Descent RSI."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np


def iterative_balanced_weights(rows: Sequence[Mapping], fields=("candidate_id", "layer", "region"),
                               iterations: int = 100) -> np.ndarray:
    """Rake positive weights to equal marginals for every declared field."""
    if not rows: raise ValueError("balanced support cannot be empty")
    weights = np.ones(len(rows), np.float64)
    for _ in range(int(iterations)):
        for field in fields:
            groups = defaultdict(list)
            for index, row in enumerate(rows): groups[str(row[field])].append(index)
            target = 1.0 / len(groups)
            for indices in groups.values():
                mass = float(weights[indices].sum())
                if mass <= 0: raise ValueError("balanced support has zero-mass group")
                weights[indices] *= target / mass
        weights /= weights.sum()
    return weights


def marginal_masses(rows: Sequence[Mapping], weights: Sequence[float], field: str) -> dict[str, float]:
    result = defaultdict(float)
    for row, weight in zip(rows, weights, strict=True): result[str(row[field])] += float(weight)
    return dict(sorted(result.items()))
