from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True, slots=True)
class Match:
    worker_index: int
    task_index: int
    utility: float


def maximum_weight_feasible_matching(
    utilities: np.ndarray,
    feasible: np.ndarray,
) -> list[Match]:
    """Return a one-to-one matching, with one zero-utility idle slot per worker."""

    utilities = np.asarray(utilities, dtype=np.float64)
    feasible = np.asarray(feasible, dtype=bool)
    if utilities.ndim != 2 or feasible.shape != utilities.shape:
        raise ValueError("utilities and feasible must be equally shaped matrices")
    if np.isnan(utilities).any():
        raise ValueError("utilities must not contain NaN")

    worker_count, task_count = utilities.shape
    if worker_count == 0 or task_count == 0:
        return []

    masked = np.where(feasible, utilities, -1e12)
    augmented = np.concatenate(
        [masked, np.zeros((worker_count, worker_count), dtype=np.float64)],
        axis=1,
    )
    rows, columns = linear_sum_assignment(-augmented)

    matches: list[Match] = []
    for row, column in zip(rows.tolist(), columns.tolist()):
        if (
            column < task_count
            and feasible[row, column]
            and utilities[row, column] > 0.0
        ):
            matches.append(Match(row, column, float(utilities[row, column])))
    return matches
