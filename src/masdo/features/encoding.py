from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from masdo.environment.simulator import SpatialCrowdsourcingEnv


@dataclass(frozen=True, slots=True)
class EncodedState:
    worker_features: np.ndarray
    pair_features: np.ndarray
    feasible: np.ndarray
    task_ids: tuple[str, ...]


class StateEncoder:
    """Encode all visible tasks, growing the task axis when necessary."""

    worker_feature_dim = 7
    metric_edge_base_dim = 3

    def __init__(
        self,
        worker_ids: tuple[str, ...],
        horizon: int,
        max_tasks: int,
        max_task_value: float,
        region_count: int = 16,
    ) -> None:
        if horizon <= 0 or max_tasks <= 0 or max_task_value <= 0 or region_count <= 0:
            raise ValueError("encoder scales must be positive")
        self.worker_ids = worker_ids
        self.horizon = int(horizon)
        self.max_tasks = int(max_tasks)
        self.max_task_value = float(max_task_value)
        self.region_count = int(region_count)
        self.model_pair_feature_dim = 12
        self.pair_feature_dim = self.model_pair_feature_dim + self.metric_edge_base_dim

    @staticmethod
    def _region_ids(locations: np.ndarray, grid) -> np.ndarray:
        """Vectorized equivalent of ``SpatialGrid.region_id``."""

        values = np.asarray(locations, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("locations must have shape [items, 2]")
        x_ratio = (values[:, 0] - grid.min_x) / (grid.max_x - grid.min_x)
        y_ratio = (values[:, 1] - grid.min_y) / (grid.max_y - grid.min_y)
        columns = np.floor(x_ratio * grid.cols).astype(np.int64)
        rows = np.floor(y_ratio * grid.rows).astype(np.int64)
        np.clip(columns, 0, grid.cols - 1, out=columns)
        np.clip(rows, 0, grid.rows - 1, out=rows)
        return rows * grid.cols + columns

    def encode(self, env: SpatialCrowdsourcingEnv) -> EncodedState:
        if tuple(sorted(env.worker_state)) != self.worker_ids:
            raise ValueError("environment workers do not match encoder worker ids")
        if env.grid.num_regions != self.region_count:
            raise ValueError("Environment grid does not match the encoder region count")
        active_ids = env.active_task_ids()
        active_ids.sort(
            key=lambda task_id: (
                env.task(task_id).deadline,
                -env.task(task_id).value,
                task_id,
            )
        )
        task_ids = tuple(active_ids)
        capacity = max(self.max_tasks, len(task_ids))
        worker_count = len(self.worker_ids)

        regional_rates = env.metric_tracker.regional_service_rates()
        regional_gains = env.metric_tracker.marginal_balanced_coverage_gains()
        gain_scale = max(1e-12, float(np.max(np.abs(regional_gains))))
        worker_features = np.zeros(
            (worker_count, self.worker_feature_dim), dtype=np.float32
        )
        pair_features = np.zeros(
            (worker_count, capacity, self.pair_feature_dim),
            dtype=np.float32,
        )
        feasible = np.zeros(
            (worker_count, capacity), dtype=bool
        )
        time_fraction = env.time / self.horizon
        worker_locations = np.asarray(
            [env.worker_location(worker_id) for worker_id in self.worker_ids],
            dtype=np.float64,
        )
        worker_available = np.asarray(
            [env.worker_is_available(worker_id) for worker_id in self.worker_ids],
            dtype=bool,
        )
        worker_busy = np.asarray(
            [env.worker_busy_remaining(worker_id) for worker_id in self.worker_ids],
            dtype=np.float64,
        )
        worker_end = np.asarray(
            [env.worker_end_time(worker_id) for worker_id in self.worker_ids],
            dtype=np.float64,
        )
        worker_regions = self._region_ids(worker_locations, env.grid)
        worker_features[:, 0:2] = worker_locations
        worker_features[:, 2] = worker_available.astype(np.float32)
        worker_features[:, 3] = worker_busy / self.horizon
        worker_features[:, 4] = time_fraction
        worker_features[:, 5] = regional_rates[worker_regions]
        worker_features[:, 6] = regional_gains[worker_regions] / gain_scale

        if task_ids:
            tasks = [env.task(task_id) for task_id in task_ids]
            task_locations = np.asarray(
                [task.location for task in tasks], dtype=np.float64
            )
            task_values = np.asarray([task.value for task in tasks], dtype=np.float64)
            task_deadlines = np.asarray(
                [task.deadline for task in tasks], dtype=np.float64
            )
            task_releases = np.asarray(
                [task.release_time for task in tasks], dtype=np.float64
            )
            task_regions = self._region_ids(task_locations, env.grid)
            delta = task_locations[None, :, :] - worker_locations[:, None, :]
            distances = np.linalg.norm(delta, axis=-1)
            travel_steps = np.ceil(distances / env.travel_speed).astype(np.int64)
            arrival = env.time + travel_steps
            task_count = len(task_ids)
            feasible[:, :task_count] = (
                worker_available[:, None]
                & (arrival <= task_deadlines[None, :])
                & (arrival <= worker_end[:, None])
            )
            if getattr(env, 'max_distance', None) is not None:
                feasible[:, :task_count] &= distances <= env.max_distance
            target = pair_features[:, :task_count]
            target[..., 0:2] = task_locations[None, :, :]
            target[..., 2:4] = delta
            target[..., 4] = distances
            target[..., 5] = np.exp(-distances / env.d_ref)
            target[..., 6] = task_values[None, :] / self.max_task_value
            target[..., 7] = np.maximum(0.0, task_deadlines - env.time)[None, :] / self.horizon
            target[..., 8] = np.maximum(0.0, env.time - task_releases)[None, :] / self.horizon
            target[..., 9] = travel_steps / self.horizon
            target[..., 10] = regional_rates[task_regions][None, :]
            target[..., 11] = (regional_gains[task_regions] / gain_scale)[None, :]
            # Observable one-completion metric potentials. These differ
            # from step rewards when service is delayed or coverage changes.
            released_value = float(env.metric_tracker.released_value)
            released_count = int(env.metric_tracker.released_count)
            if released_value <= 0.0 or released_count <= 0:
                raise AssertionError(
                    "active tasks require positive released metric denominators"
                )
            edge_base = target[
                ..., self.model_pair_feature_dim : self.pair_feature_dim
            ]
            edge_base[..., 0] = task_values[None, :] / released_value
            edge_base[..., 1] = regional_gains[task_regions][None, :]
            edge_base[..., 2] = np.exp(-distances / env.d_ref) / released_count
            edge_base *= feasible[:, :task_count, None]

        return EncodedState(
            worker_features=worker_features,
            pair_features=pair_features,
            feasible=feasible,
            task_ids=task_ids,
        )
