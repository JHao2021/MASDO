from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Iterable

import numpy as np

from masdo.environment.entities import Task
from masdo.environment.grid import SpatialGrid


@dataclass(frozen=True, slots=True)
class MetricSnapshot:
    completion_revenue: float
    balanced_coverage: float
    travel_efficiency: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.completion_revenue,
                self.balanced_coverage,
                self.travel_efficiency,
            ],
            dtype=np.float64,
        )


class PlatformMetricTracker:
    """Track cumulative service metrics and their stepwise reward increments."""

    def __init__(self, grid: SpatialGrid, d_ref: float) -> None:
        if d_ref <= 0:
            raise ValueError("d_ref must be positive")
        self.grid = grid
        self.d_ref = float(d_ref)
        self._released: dict[str, Task] = {}
        self._completed_distance: dict[str, float] = {}

    def release(self, tasks: Iterable[Task]) -> None:
        for task in tasks:
            existing = self._released.get(task.task_id)
            if existing is not None and existing != task:
                raise ValueError(f"conflicting task definition for {task.task_id}")
            self._released[task.task_id] = task

    def complete(self, task: Task, travel_cost: float) -> None:
        if task.task_id not in self._released:
            raise ValueError("a task must be released before completion")
        if task.task_id in self._completed_distance:
            raise ValueError(f"task {task.task_id} completed more than once")
        if travel_cost < 0:
            raise ValueError("travel cost must be nonnegative")
        self._completed_distance[task.task_id] = float(travel_cost)

    @property
    def released_count(self) -> int:
        return len(self._released)

    @property
    def released_value(self) -> float:
        """Total value in the currently released metric denominator."""

        return float(sum(task.value for task in self._released.values()))

    @property
    def completed_count(self) -> int:
        return len(self._completed_distance)

    def regional_counts(self) -> tuple[np.ndarray, np.ndarray]:
        released = np.zeros(self.grid.num_regions, dtype=np.float64)
        completed = np.zeros(self.grid.num_regions, dtype=np.float64)
        for task in self._released.values():
            region = self.grid.region_id(task.location)
            released[region] += 1.0
            if task.task_id in self._completed_distance:
                completed[region] += 1.0
        return released, completed

    @staticmethod
    def _balanced_coverage_from_counts(
        released: np.ndarray,
        completed: np.ndarray,
    ) -> float:
        active = released > 0
        if not np.any(active):
            return 0.0
        rates = completed[active] / released[active]
        if np.allclose(rates, 0.0):
            return 0.0
        adequacy = float(np.mean(rates))
        jain = float(
            np.square(np.sum(rates))
            / (rates.size * np.sum(np.square(rates)))
        )
        return adequacy * jain

    def regional_service_rates(self) -> np.ndarray:
        released, completed = self.regional_counts()
        rates = np.zeros_like(released)
        active = released > 0
        rates[active] = completed[active] / released[active]
        return rates

    def marginal_balanced_coverage_gain(self, region: int) -> float:
        if region < 0 or region >= self.grid.num_regions:
            raise ValueError("region index is outside the spatial grid")
        return float(self.marginal_balanced_coverage_gains()[region])

    def marginal_balanced_coverage_gains(self) -> np.ndarray:
        released, completed = self.regional_counts()
        current = self._balanced_coverage_from_counts(released, completed)
        gains = np.zeros(self.grid.num_regions, dtype=np.float64)
        for region in range(self.grid.num_regions):
            if released[region] <= completed[region]:
                continue
            completed_after = completed.copy()
            completed_after[region] += 1.0
            future = self._balanced_coverage_from_counts(
                released, completed_after
            )
            gains[region] = future - current
        return gains

    def snapshot(self) -> MetricSnapshot:
        return self.snapshot_excluding(())

    def snapshot_excluding(
        self, excluded_completed_task_ids: Iterable[str]
    ) -> MetricSnapshot:
        excluded = set(excluded_completed_task_ids)
        if not self._released:
            return MetricSnapshot(0.0, 0.0, 0.0)

        released_value = self.released_value
        completed_value = sum(
            self._released[task_id].value
            for task_id in self._completed_distance
            if task_id not in excluded
        )
        completion_revenue = completed_value / released_value

        released_counts = np.zeros(self.grid.num_regions, dtype=np.float64)
        completed_counts = np.zeros(self.grid.num_regions, dtype=np.float64)
        for task in self._released.values():
            region = self.grid.region_id(task.location)
            released_counts[region] += 1.0
            if (
                task.task_id in self._completed_distance
                and task.task_id not in excluded
            ):
                completed_counts[region] += 1.0
        balanced_coverage = self._balanced_coverage_from_counts(
            released_counts, completed_counts
        )

        efficiency_sum = sum(
            exp(-distance / self.d_ref)
            for task_id, distance in self._completed_distance.items()
            if task_id not in excluded
        )
        travel_efficiency = efficiency_sum / len(self._released)

        values = (completion_revenue, balanced_coverage, travel_efficiency)
        if any(value < -1e-12 or value > 1.0 + 1e-12 for value in values):
            raise AssertionError(f"metric left [0, 1]: {values}")
        return MetricSnapshot(*(float(np.clip(value, 0.0, 1.0)) for value in values))

    def deletion_contribution(
        self, completed_task_ids: Iterable[str]
    ) -> np.ndarray:
        task_ids = tuple(completed_task_ids)
        unknown = set(task_ids).difference(self._completed_distance)
        if unknown:
            raise ValueError(
                f"cannot delete tasks that are not completed: {sorted(unknown)}"
            )
        return self.snapshot().as_array() - self.snapshot_excluding(
            task_ids
        ).as_array()

    def delta_from(self, previous: MetricSnapshot) -> np.ndarray:
        return self.snapshot().as_array() - previous.as_array()
