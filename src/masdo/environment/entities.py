from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

Location = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    location: Location
    release_time: float
    deadline: float
    value: float

    def __post_init__(self) -> None:
        if self.deadline < self.release_time:
            raise ValueError("task deadline must not precede its release")
        if self.value <= 0:
            raise ValueError("task value must be positive")


@dataclass(frozen=True, slots=True)
class Worker:
    worker_id: str
    location: Location
    release_time: float = 0.0
    end_time: float = float("inf")

    def __post_init__(self) -> None:
        if self.end_time < self.release_time:
            raise ValueError("worker end_time must not precede its release")


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    workers: tuple[Worker, ...]
    tasks: tuple[Task, ...]
    horizon: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("scenario horizon must be positive")
        if len({worker.worker_id for worker in self.workers}) != len(self.workers):
            raise ValueError("worker ids must be unique")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("task ids must be unique")


@dataclass(frozen=True, slots=True)
class Completion:
    task_id: str
    worker_id: str
    completion_time: float
    travel_cost: float
