"""Load anonymized, prepared assignment instances from JSON."""
import json
import math

from masdo.environment.entities import Scenario, Task, Worker


def load_scenario(path):
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    scale = float(data["coordinate_scale_km"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("coordinate_scale_km must be positive")
    horizon = data["horizon"]
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")

    def location(value):
        result = tuple(float(x) for x in value)
        if len(result) != 2 or not all(math.isfinite(x) for x in result):
            raise ValueError("location must contain two finite planar coordinates")
        return result

    def step(value):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("Scenario times must be nonnegative integer steps")
        return value

    workers = tuple(
        Worker(worker_id=str(row["worker_id"]), location=location(row["location"]),
               release_time=step(row.get("release_time", 0)),
               end_time=step(row.get("end_time", horizon)))
        for row in data["workers"]
    )
    tasks = tuple(
        Task(task_id=str(row["task_id"]), location=location(row["location"]),
             release_time=step(row["release_time"]), deadline=step(row["deadline"]),
             value=float(row["value"]))
        for row in data["tasks"]
    )
    if not workers or not tasks or any(not math.isfinite(task.value) for task in tasks):
        raise ValueError("Provide workers and tasks with finite positive values")
    if any(task.release_time >= horizon for task in tasks):
        raise ValueError("All scenario tasks must be released within the operating period")
    # Arrival at the task location completes service.
    # Unrelated source metadata is not carried into the environment or output.
    return Scenario("evaluation", workers, tasks, horizon), scale
