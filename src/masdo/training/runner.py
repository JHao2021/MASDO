"""Train the skill executor, then the per-step skill organizer."""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import torch

from masdo.assignment.learning import VectorTDLearner
from masdo.assignment.policy import VectorSkillExecutor
from masdo.deployment.checkpoint import deployment_bundle, DEPLOYMENT_FORMAT
from masdo.deployment.inputs import load_scenario
from masdo.discovery.learning import AuxiliaryTDLearner, SequenceDiscovery
from masdo.environment.grid import SpatialGrid
from masdo.evaluation.rollout import rollout
from masdo.orchestration.learning import SkillTDLearner
from masdo.orchestration.policy import SkillOrganizer
from .collection import collect
from .replay import TrainingReplay


def project_path(root, value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Use a project-relative file path")
    return Path(root) / path


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def positive_float(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def training_objectives(settings):
    if isinstance(settings, dict):
        count = positive_integer(settings["count"], "objective count")
        alpha = np.asarray(settings.get("dirichlet_alpha", [1., 1., 1.]), np.float64)
        if alpha.shape != (3,) or not np.isfinite(alpha).all() or (alpha <= 0).any():
            raise ValueError("dirichlet_alpha must contain three positive values")
        values = np.random.default_rng(settings.get("seed", 0)).dirichlet(alpha, count)
    else:
        values = np.asarray(settings, np.float64)
    if (values.ndim != 2 or values.shape[1] != 3 or len(values) == 0
            or not np.isfinite(values).all() or (values < 0).any()
            or not np.allclose(values.sum(1), 1.)):
        raise ValueError("Training objectives must be three-component simplex vectors")
    return values


def prepare_environment(scenarios, scales, settings):
    if not np.allclose(scales, scales[0], rtol=1e-10, atol=0):
        raise ValueError("Training scenarios must share one coordinate system and scale")
    scale = scales[0]
    tasks = [task for scenario in scenarios for task in scenario.tasks]
    rows = positive_integer(settings.get("grid_rows", 8), "grid_rows")
    cols = positive_integer(settings.get("grid_cols", 8), "grid_cols")
    if "grid_bounds" in settings:
        grid = SpatialGrid(**settings["grid_bounds"], rows=rows, cols=cols)
    else:
        grid = SpatialGrid.fit(tasks, rows=rows, cols=cols)
    max_tasks = positive_integer(settings.get("max_tasks", max(len(s.tasks) for s in scenarios)), "max_tasks")
    capacity = positive_integer(settings.get("encoder_max_tasks", max_tasks), "encoder_max_tasks")
    environment = {
        "coordinate_scale_km": scale,
        "travel_speed": positive_float(settings["travel_speed_km_per_step"], "travel_speed_km_per_step") / scale,
        "d_ref": positive_float(settings["distance_reference_km"], "distance_reference_km") / scale,
        "max_tasks": max_tasks, "encoder_max_tasks": capacity,
        "max_task_value": positive_float(settings.get("max_task_value", max(t.value for t in tasks)), "max_task_value"),
        "grid_bounds": {key: getattr(grid, key) for key in ("min_x", "max_x", "min_y", "max_y")},
        "grid_rows": rows, "grid_cols": cols,
    }
    return environment, grid


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def train(config, root, *, resume=None):
    """Train from prepared training scenarios and export a deployment checkpoint.

    ``resume`` is a project-relative training checkpoint. Training checkpoints
    preserve optimizers, target networks, progress and random generator states.
    """
    config = deepcopy(config)
    device = config.get("device", "cpu")
    torch.set_num_threads(positive_integer(config.get("cpu_threads", 1), "cpu_threads"))
    seed = config.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    context_steps = positive_integer(config.get("discovery_context_steps", 32), "discovery_context_steps")
    future_steps = positive_integer(config.get("discovery_future_steps", 8), "discovery_future_steps")
    paths = config["train_scenarios"]
    if not isinstance(paths, list) or not paths:
        raise ValueError("train_scenarios must be a nonempty list of prepared JSON files")
    loaded = [load_scenario(project_path(root, path)) for path in paths]
    scenarios, scales = zip(*loaded)
    environment, grid = prepare_environment(scenarios, scales, config["environment"])
    distance = positive_float(config["reachable_distance_km"], "reachable_distance_km") / scales[0]
    objectives = training_objectives(config["objectives"])
    ac, organization = config["discovery_assignment"], config["organization"]
    ac_episodes = positive_integer(ac["episodes"], "discovery_assignment.episodes")
    b_episodes = positive_integer(organization["episodes"], "organization.episodes")
    batch_size = positive_integer(ac["batch_size"], "batch_size")
    sequence_batch = positive_integer(ac["sequence_batch_size"], "sequence_batch_size")
    capacity = positive_integer(ac["replay_capacity"], "replay_capacity")
    updates = positive_integer(ac["updates_per_episode"], "updates_per_episode")
    fits = positive_integer(ac["predictor_updates_per_episode"], "predictor_updates_per_episode")
    checkpoint_every = positive_integer(config.get("checkpoint_every", 1), "checkpoint_every")
    noise_start, noise_end = float(ac["exploration_noise_start"]), float(ac["exploration_noise_end"])
    if not all(math.isfinite(x) and x >= 0 for x in (noise_start, noise_end)):
        raise ValueError("Exploration noise scales must be finite and nonnegative")
    executor = VectorSkillExecutor(**config.get("executor", {})).to(device)
    vector = VectorTDLearner(executor, **ac.get("vector_learning", {}))
    auxiliary = AuxiliaryTDLearner(executor, **ac.get("auxiliary_learning", {}))
    if auxiliary.target_interval != vector.target_interval:
        raise ValueError('Vector and auxiliary TD must share the target refresh interval')
    discovery = SequenceDiscovery(device=device, **ac.get("sequence_learning", {}))
    organizer, skill_learner = None, None
    replay = TrainingReplay(capacity)
    progress = {"discovery_assignment": 0, "organization": 0}

    def create_organizer():
        model = SkillOrganizer(executor.embedding.weight, executor_hidden=executor.hidden,
                                  **config.get("organizer", {})).to(device)
        return model, SkillTDLearner(model, **organization.get("learning", {}))

    output = project_path(root, config["output_dir"])
    if resume is not None:
        resume_path = project_path(root, resume)
        if resume_path.resolve() != (output / "training.pt").resolve():
            raise ValueError("Resume the training.pt file in the configured output directory")
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        if (saved.get("format") != DEPLOYMENT_FORMAT or saved.get("purpose") != "training"
                or saved["config"] != config or saved["environment"] != environment
                or saved["objectives"] != objectives.tolist()):
            raise ValueError("Training checkpoint and configuration differ")
        vector.load_state_dict(saved["vector"])
        auxiliary.load_state_dict(saved["auxiliary"])
        discovery.load_state_dict(saved["discovery"])
        replay.load_state_dict(saved["replay"])
        progress = saved["progress"]
        if saved["organization"] is not None:
            organizer, skill_learner = create_organizer()
            skill_learner.load_state_dict(saved["organization"])
        rng.bit_generator.state = saved["rng"]
        torch.set_rng_state(saved["torch_rng"])
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        # Discard log entries beyond the last committed episode checkpoint.
        log_path = output / "metrics.jsonl"
        if log_path.stat().st_size < saved["log_bytes"]:
            raise ValueError("Training log is shorter than its checkpoint")
        with log_path.open("r+b") as handle:
            handle.truncate(saved["log_bytes"])
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Output directory is not empty; resume it or choose another output_dir")
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "config.json", config)
        write_json(output / "objectives.json", objectives.tolist())

    def checkpoint():
        save_state(output / "training.pt", {
            "format": DEPLOYMENT_FORMAT, "purpose": "training", "config": config,
            "environment": environment, "objectives": objectives.tolist(), "progress": dict(progress),
            "vector": vector.state_dict(), "auxiliary": auxiliary.state_dict(),
            "discovery": discovery.state_dict(), "replay": replay.state_dict(),
            "organization": None if skill_learner is None else skill_learner.state_dict(),
            "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
            "log_bytes": (output / "metrics.jsonl").stat().st_size,
        })

    def sample_conditions():
        index = int(rng.integers(len(scenarios)))
        objective = objectives[int(rng.integers(len(objectives)))].copy()
        return index, objective, int(rng.integers(2**32))

    with (output / "metrics.jsonl").open("a" if resume else "x", encoding="utf-8") as log:
        def record(stage, episode, event, **values):
            receipt = dict(stage=stage, episode=episode, event=event, **values)
            log.write(json.dumps(receipt, allow_nan=False) + "\n")
            log.flush()

        if resume is None:
            checkpoint()
        executor.train().requires_grad_(True)
        for episode in range(progress["discovery_assignment"], ac_episodes):
            noise = noise_start + (noise_end-noise_start)*episode/max(1, ac_episodes-1)
            index, objective, rollout_seed = sample_conditions()
            acting = collect(scenarios[index], environment, grid, distance, objective, rollout_seed,
                             executor, trajectory_id=str(episode), context_steps=context_steps,
                             future_steps=future_steps, exploration_noise=noise, replay_capacity=capacity)
            replay.add_fit(acting.sequences, rng)
            if replay.fit_sequences:
                for _ in range(fits):
                    indices = rng.choice(len(replay.fit_sequences), min(sequence_batch, len(replay.fit_sequences)), replace=False)
                    receipt = discovery.fit([replay.fit_sequences[int(i)] for i in indices],
                                            calibration_sequences=replay.fit_sequences)
                    record("discovery_assignment", episode, "predictor_update", **receipt)
            # Attribute each delayed suffix bonus to its original assignment step,
            # and divide the sum by total team size, not the number available.
            step_bonuses = {}
            if acting.sequences and not discovery.normalizer.fitted.item():
                raise ValueError('No fitting trajectories with available workers; add training scenarios')
            for start in range(0, len(acting.sequences), sequence_batch):
                seq_batch = acting.sequences[start:start+sequence_batch]
                scores = discovery.score(seq_batch)["bonus"]
                for sequence, bonus in zip(seq_batch, scores):
                    step_bonuses[sequence.step] = step_bonuses.get(sequence.step, 0.) + float(bonus)/len(scenarios[index].workers)
            replay.add_policy(acting.transitions, [step_bonuses.get(step, 0.) for step in acting.transition_steps])
            for _ in range(updates):
                indices = rng.choice(len(replay.transitions), min(batch_size, len(replay.transitions)), replace=False)
                batch = [replay.transitions[int(i)] for i in indices]
                bonuses = [replay.bonuses[int(i)] for i in indices]
                # Update skill-specific parameters before the full value network.
                record("discovery_assignment", episode, "skill_update", **auxiliary.update(batch, bonuses))
                record("discovery_assignment", episode, "vector_update", **vector.update(batch))
                if vector.updates % vector.target_interval == 0:
                    # Both losses select/evaluate future actions with the same
                    # frozen assignment parameters after the complete round.
                    auxiliary.target.load_state_dict(executor.state_dict())
            record("discovery_assignment", episode, "rollout", scenario=index, exploration_noise=noise,
                   step_bonuses=step_bonuses, replay_size=len(replay.transitions), **acting.metrics)
            del acting
            progress["discovery_assignment"] = episode+1
            if (episode+1) % checkpoint_every == 0 or episode+1 == ac_episodes:
                checkpoint()
            print(f"Discovery/assignment: {episode+1}/{ac_episodes}", flush=True)

        executor.eval().requires_grad_(False)
        if organizer is None:
            organizer, skill_learner = create_organizer()
        organizer.train()
        for episode in range(progress["organization"], b_episodes):
            index, objective, rollout_seed = sample_conditions()
            result = rollout(
                scenarios[index], environment, grid, distance, objective, rollout_seed,
                executor, organizer, learner=skill_learner,
                on_update=lambda receipt: record("organization", episode, "skill_selection_update", **receipt),
            )
            record("organization", episode, "rollout", scenario=index, objective=objective.tolist(), **result)
            progress["organization"] = episode+1
            if (episode+1) % checkpoint_every == 0 or episode+1 == b_episodes:
                checkpoint()
            print(f"Organization: {episode+1}/{b_episodes}", flush=True)

    organizer.eval()
    checkpoint()
    save_state(output / "model.pt", deployment_bundle(executor, organizer, environment))
    return {"method": "MASDO", "progress": progress, "vector_updates": vector.updates,
            "skill_updates": auxiliary.updates, "predictor_updates": discovery.updates,
            "organization_updates": skill_learner.updates,
            "checkpoint": str(Path(config["output_dir"]) / "model.pt")}
