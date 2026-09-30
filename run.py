"""Evaluate a MASDO deployment on one prepared operating period."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def project_path(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Use a project-relative file path")
    return ROOT / path


def main():
    parser = argparse.ArgumentParser(description="Evaluate MASDO on a prepared scenario")
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()

    import numpy as np
    import torch

    from masdo.deployment.checkpoint import load_deployment
    from masdo.deployment.inputs import load_scenario
    from masdo.environment.grid import SpatialGrid
    from masdo.evaluation.rollout import rollout

    with project_path(args.config).open(encoding="utf-8") as handle:
        config = json.load(handle)
    threads = config.get("cpu_threads", 1)
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    torch.set_num_threads(threads)
    objective = np.asarray(config["objective"], dtype=np.float64)
    if (objective.shape != (3,) or not np.isfinite(objective).all()
            or (objective < 0).any() or not np.isclose(objective.sum(), 1.0)):
        raise ValueError("objective must contain three nonnegative weights summing to one")
    distance_km = float(config["reachable_distance_km"])
    if not math.isfinite(distance_km) or distance_km <= 0:
        raise ValueError("reachable_distance_km must be positive")

    scenario, coordinate_scale_km = load_scenario(project_path(config["scenario"]))
    # Load only tensors and primitive configuration fields, not Python objects.
    bundle = torch.load(project_path(config["checkpoint"]), map_location="cpu", weights_only=True)
    executor, organizer, environment = load_deployment(bundle, config.get("device", "cpu"))
    if not math.isclose(coordinate_scale_km, environment["coordinate_scale_km"], rel_tol=1e-10, abs_tol=0.):
        raise ValueError("Scenario and checkpoint must use the same coordinate scale")
    grid = SpatialGrid(**environment["grid_bounds"],
                       rows=int(environment["grid_rows"]), cols=int(environment["grid_cols"]))
    result = rollout(
        scenario, environment, grid, distance_km / coordinate_scale_km,
        objective, int(config.get("seed", 0)), executor, organizer,
    )
    summary = {
        "method": "MASDO",
        "objective": objective.tolist(),
        "seed": int(config.get("seed", 0)),
        "OPU": result["U"],
        "TFR": result["C"],
        "TCR": result["G"],
        "MTE": result["E"],
        "steps": result["steps"],
        "skill_counts": result["skill_counts"],
        "skill_compositions": result["compositions"],
        "decision_ms_per_step": 1000 * result["decision_seconds"] / max(1, result["steps"]),
        "rollout_seconds": result["wall_seconds"],
    }
    output = project_path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite a previous evaluation implicitly.
    with output.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
