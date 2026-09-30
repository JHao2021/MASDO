from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from masdo.matching.solver import maximum_weight_feasible_matching


@dataclass(frozen=True, slots=True)
class JointAction:
    action_indices: np.ndarray
    task_matches: tuple[tuple[int, int], ...]


def constrained_joint_action(
    task_q: np.ndarray,
    idle_q: np.ndarray,
    feasible: np.ndarray,
) -> JointAction:
    """Maximize total value under one-worker/one-task constraints."""

    task_q = np.asarray(task_q, dtype=np.float64)
    idle_q = np.asarray(idle_q, dtype=np.float64)
    feasible = np.asarray(feasible, dtype=bool)
    if task_q.ndim != 2 or feasible.shape != task_q.shape:
        raise ValueError("task_q and feasible must be equally shaped matrices")
    if idle_q.shape != (task_q.shape[0],):
        raise ValueError("idle_q must contain one value per worker")
    advantages = task_q - idle_q[:, None]
    matches = maximum_weight_feasible_matching(advantages, feasible)
    idle_index = task_q.shape[1]
    action_indices = np.full(task_q.shape[0], idle_index, dtype=np.int64)
    task_matches: list[tuple[int, int]] = []
    for match in matches:
        action_indices[match.worker_index] = match.task_index
        task_matches.append((match.worker_index, match.task_index))
    return JointAction(action_indices, tuple(task_matches))
