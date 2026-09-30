from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from masdo.environment.entities import Location, Task


@dataclass(frozen=True, slots=True)
class SpatialGrid:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    rows: int = 4
    cols: int = 4

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("grid dimensions must be positive")
        if self.max_x <= self.min_x or self.max_y <= self.min_y:
            raise ValueError("grid bounds must have positive extent")

    @property
    def num_regions(self) -> int:
        return self.rows * self.cols

    def region_id(self, location: Location) -> int:
        x, y = location
        x_ratio = (x - self.min_x) / (self.max_x - self.min_x)
        y_ratio = (y - self.min_y) / (self.max_y - self.min_y)
        col = min(self.cols - 1, max(0, int(x_ratio * self.cols)))
        row = min(self.rows - 1, max(0, int(y_ratio * self.rows)))
        return row * self.cols + col

    @classmethod
    def fit(
        cls,
        tasks: Iterable[Task],
        rows: int = 4,
        cols: int = 4,
        minimum_extent: float = 1e-6,
    ) -> "SpatialGrid":
        locations = [task.location for task in tasks]
        if not locations:
            raise ValueError("cannot fit a grid without tasks")
        xs, ys = zip(*locations)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        if max_x - min_x < minimum_extent:
            max_x = min_x + minimum_extent
        if max_y - min_y < minimum_extent:
            max_y = min_y + minimum_extent
        return cls(min_x, max_x, min_y, max_y, rows=rows, cols=cols)
