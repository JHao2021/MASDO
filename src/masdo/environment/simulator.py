from __future__ import annotations

from dataclasses import dataclass
from math import ceil, hypot
from typing import Iterable

import numpy as np

from masdo.environment.entities import Completion, Scenario, Task, Worker
from masdo.environment.grid import SpatialGrid
from masdo.evaluation.metrics import MetricSnapshot, PlatformMetricTracker


@dataclass(slots=True)
class _WorkerState:
    definition: Worker
    location: tuple[float, float]
    busy_until: int = 0
    current_task_id: str | None = None
    current_distance: float = 0.0


class SpatialCrowdsourcingEnv:
    """Finite-horizon dynamic assignment environment used by every system."""

    def __init__(
        self,
        scenario: Scenario,
        travel_speed: float,
        d_ref: float,
        grid: SpatialGrid | None = None,
    ) -> None:
        if travel_speed <= 0:
            raise ValueError("travel_speed must be positive")
        self.scenario = scenario
        self.travel_speed = float(travel_speed)
        self.grid = grid or SpatialGrid.fit(scenario.tasks, rows=2, cols=2)
        self.d_ref = float(d_ref)
        self._tasks = {task.task_id: task for task in scenario.tasks}
        self._workers = {worker.worker_id: worker for worker in scenario.workers}
        self.reset()

    def reset(self) -> dict[str, object]:
        self.time = 0
        self.done = False
        self.metric_tracker = PlatformMetricTracker(self.grid, self.d_ref)
        self.task_status = {task_id: "unreleased" for task_id in self._tasks}
        self.worker_state = {
            worker_id: _WorkerState(worker, worker.location)
            for worker_id, worker in self._workers.items()
        }
        self.completions: list[Completion] = []
        self.assignment_count = 0
        self.cumulative_reward = np.zeros(3, dtype=np.float64)
        self._activate_releases()
        return self.observation()

    def observation(self) -> dict[str, object]:
        return {
            "time": self.time,
            "available_worker_ids": self.available_worker_ids(),
            "active_task_ids": self.active_task_ids(),
            "metrics": self.metric_tracker.snapshot().as_array(),
        }

    def available_worker_ids(self) -> list[str]:
        available: list[str] = []
        for worker_id, state in self.worker_state.items():
            worker = state.definition
            if (
                worker.release_time <= self.time < worker.end_time
                and state.current_task_id is None
                and state.busy_until <= self.time
            ):
                available.append(worker_id)
        return sorted(available)

    def active_task_ids(self) -> list[str]:
        return sorted(
            task_id
            for task_id, status in self.task_status.items()
            if status == "pending"
            and self._tasks[task_id].release_time <= self.time
            and self.time <= self._tasks[task_id].deadline
        )

    def distance(self, worker_id: str, task_id: str) -> float:
        worker_location = self.worker_state[worker_id].location
        task_location = self._tasks[task_id].location
        return hypot(
            worker_location[0] - task_location[0],
            worker_location[1] - task_location[1],
        )

    def travel_steps(self, worker_id: str, task_id: str) -> int:
        return int(ceil(self.distance(worker_id, task_id) / self.travel_speed))

    def is_feasible(self, worker_id: str, task_id: str) -> bool:
        state = self.worker_state.get(worker_id)
        task = self._tasks.get(task_id)
        if state is None or task is None:
            return False
        worker = state.definition
        if not (
            worker.release_time <= self.time < worker.end_time
            and state.current_task_id is None
            and state.busy_until <= self.time
        ):
            return False
        if not (
            self.task_status[task_id] == "pending"
            and task.release_time <= self.time <= task.deadline
        ):
            return False
        arrival = self.time + self.travel_steps(worker_id, task_id)
        return arrival <= task.deadline and arrival <= worker.end_time

    def task(self, task_id: str) -> Task:
        return self._tasks[task_id]

    def worker_location(self, worker_id: str) -> tuple[float, float]:
        return self.worker_state[worker_id].location

    def worker_end_time(self, worker_id: str) -> float:
        return self._workers[worker_id].end_time

    def worker_is_available(self, worker_id: str) -> bool:
        state = self.worker_state[worker_id]
        worker = state.definition
        return (
            worker.release_time <= self.time < worker.end_time
            and state.current_task_id is None
            and state.busy_until <= self.time
        )

    def worker_busy_remaining(self, worker_id: str) -> int:
        state = self.worker_state[worker_id]
        return max(0, state.busy_until - self.time)

    def worker_current_task_id(self, worker_id: str) -> str | None:
        """Return the worker's in-flight task without exposing mutable state."""

        return self.worker_state[worker_id].current_task_id

    def pair_matrices(
        self,
    ) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray]:
        worker_ids = self.available_worker_ids()
        task_ids = self.active_task_ids()
        shape = (len(worker_ids), len(task_ids))
        feasible = np.zeros(shape, dtype=bool)
        distances = np.zeros(shape, dtype=np.float64)
        values = np.zeros(shape, dtype=np.float64)
        for worker_index, worker_id in enumerate(worker_ids):
            for task_index, task_id in enumerate(task_ids):
                distances[worker_index, task_index] = self.distance(worker_id, task_id)
                values[worker_index, task_index] = self._tasks[task_id].value
                arrival = self.time + int(ceil(distances[worker_index, task_index] / self.travel_speed))
                feasible[worker_index, task_index] = (
                    arrival <= self._tasks[task_id].deadline
                    and arrival <= self._workers[worker_id].end_time
                )
        return worker_ids, task_ids, feasible, distances, values

    def step(
        self,
        assignments: Iterable[tuple[str, str]],
    ) -> tuple[dict[str, object], np.ndarray, bool, dict[str, object]]:
        if self.done:
            raise RuntimeError("cannot step a terminated environment")
        assignments = list(assignments)
        worker_ids = [worker_id for worker_id, _ in assignments]
        task_ids = [task_id for _, task_id in assignments]
        if len(worker_ids) != len(set(worker_ids)):
            raise ValueError("a worker cannot receive multiple tasks in one step")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("a task cannot be assigned to multiple workers")
        for worker_id, task_id in assignments:
            if not self.is_feasible(worker_id, task_id):
                raise ValueError(f"infeasible assignment: {worker_id} -> {task_id}")

        previous = self.metric_tracker.snapshot()
        for worker_id, task_id in assignments:
            state = self.worker_state[worker_id]
            task = self._tasks[task_id]
            distance = self.distance(worker_id, task_id)
            state.busy_until = self.time + self.travel_steps(worker_id, task_id)
            state.current_task_id = task_id
            state.current_distance = distance
            self.task_status[task_id] = "assigned"
            self.assignment_count += 1

        self._complete_arrivals()
        self.time += 1
        self._complete_arrivals()
        self._activate_releases()
        self._expire_tasks()
        current = self.metric_tracker.snapshot()
        reward = current.as_array() - previous.as_array()
        self.cumulative_reward += reward
        self.done = self.time >= self.scenario.horizon
        info = {
            "metrics": current.as_array(),
            "scalar_metrics": current,
            "completion_count": len(self.completions),
            "assignment_count": self.assignment_count,
        }
        return self.observation(), reward, self.done, info

    def _activate_releases(self) -> None:
        newly_released: list[Task] = []
        for task_id, task in self._tasks.items():
            if (
                self.task_status[task_id] == "unreleased"
                and task.release_time <= self.time
                and task.release_time < self.scenario.horizon
            ):
                self.task_status[task_id] = (
                    "pending" if task.deadline >= self.time else "expired"
                )
                newly_released.append(task)
        self.metric_tracker.release(newly_released)

    def _complete_arrivals(self) -> None:
        for worker_id, state in self.worker_state.items():
            if state.current_task_id is None or state.busy_until > self.time:
                continue
            task = self._tasks[state.current_task_id]
            if state.busy_until <= task.deadline:
                self.task_status[task.task_id] = "completed"
                self.metric_tracker.complete(task, state.current_distance)
                self.completions.append(
                    Completion(
                        task_id=task.task_id,
                        worker_id=worker_id,
                        completion_time=state.busy_until,
                        travel_cost=state.current_distance,
                    )
                )
            else:
                self.task_status[task.task_id] = "expired"
            state.location = task.location
            state.current_task_id = None
            state.current_distance = 0.0

    def _expire_tasks(self) -> None:
        for task_id, task in self._tasks.items():
            if self.task_status[task_id] == "pending" and task.deadline < self.time:
                self.task_status[task_id] = "expired"

    @property
    def final_metrics(self) -> MetricSnapshot:
        return self.metric_tracker.snapshot()

    @property
    def mean_travel_distance(self) -> float:
        if not self.completions:
            return 0.0
        return float(np.mean([item.travel_cost for item in self.completions]))


class DistanceLimitedEnv(SpatialCrowdsourcingEnv):
    """Apply the same distance bound to feasibility checks and pair matrices."""
    def __init__(self, *args, max_distance=None, **kwargs):
        self.max_distance = max_distance
        super().__init__(*args, **kwargs)

    def is_feasible(self, worker_id, task_id):
        return (super().is_feasible(worker_id, task_id)
                and (self.max_distance is None
                     or self.distance(worker_id, task_id) <= self.max_distance))

    def pair_matrices(self):
        workers, tasks, feasible, distances, values = super().pair_matrices()
        if self.max_distance is not None:
            feasible &= distances <= self.max_distance
        return workers, tasks, feasible, distances, values
