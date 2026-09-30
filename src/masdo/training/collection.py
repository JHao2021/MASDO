"""Collect skill-conditioned trajectories and bounded transition replay."""
from dataclasses import dataclass, replace
import time

import numpy as np
import torch

from masdo.assignment.policy import (
    Frame, VectorOutput, VectorTransition, executed_observation,
    maximum_value_actions, reference_next_skills,
)
from masdo.discovery.sequences import TrajectoryRecorder
from masdo.environment.simulator import DistanceLimitedEnv
from masdo.features.encoding import StateEncoder
from masdo.features.service import service_inputs


@dataclass
class CollectedTrajectory:
    transitions: list
    transition_steps: list
    sequences: list
    metrics: dict


@torch.no_grad()
def collect(scenario, environment, grid, distance, objective, seed, executor, *,
            trajectory_id, context_steps=32, future_steps=8, exploration_noise=.1, replay_capacity=256):
    """Keep the goal fixed; choose uniform skills for available workers each step.

    Reservoir sampling limits stored transitions without truncating the episode.
    Replay sampling has a separate random stream from environment decisions.
    """
    if replay_capacity < 0 or not np.isfinite(exploration_noise) or exploration_noise < 0:
        raise ValueError("Invalid collection settings")
    started = time.perf_counter()
    action_seed, replay_seed = np.random.SeedSequence(seed).spawn(2)
    rng, replay_rng = np.random.default_rng(action_seed), np.random.default_rng(replay_seed)
    objective = np.asarray(objective, np.float64)
    env = DistanceLimitedEnv(scenario, travel_speed=environment["travel_speed"],
                            d_ref=environment["d_ref"], grid=grid, max_distance=distance)
    encoder = StateEncoder(
        tuple(sorted(w.worker_id for w in scenario.workers)), scenario.horizon,
        environment["encoder_max_tasks"], environment["max_task_value"],
        region_count=grid.num_regions,
    )
    worker_index = {worker: i for i, worker in enumerate(encoder.worker_ids)}
    n = len(worker_index)
    recorder = TrajectoryRecorder(trajectory_id, context_steps, future_steps)
    history = executor.initial_history(n)
    events, transitions, transition_steps = [], [], []
    skills = np.zeros(n, dtype=np.int64)
    skill_counts = np.zeros(8, np.int64)
    reward_sum = np.zeros(3, np.float64)

    def encode(step, current_skills):
        state = encoder.encode(env)
        clock = np.broadcast_to(
            [max(0., 1-step/scenario.horizon), step/scenario.horizon], (n, 2),
        ).copy().astype(np.float32)
        frame = Frame(state.worker_features.copy(), service_inputs(env, state),
                      state.feasible[:, :len(state.task_ids)].copy(),
                      objective.copy(), current_skills.copy(), clock)
        frame.validate()
        return state, frame

    step = 0
    state, frame = encode(step, skills)
    frame = replace(frame, skills=reference_next_skills(
        frame.skills, available=frame.workers[:, 2] > 0, rng=rng))
    while not env.done:
        available = frame.workers[:, 2] > 0
        skill_counts += np.bincount(frame.skills[available], minlength=8)
        output = executor(frame, history)
        if exploration_noise:
            # Add zero-mean noise to learned projected edge scores.
            # Equal perturbations in each component project to the same scalar.
            noise = rng.normal(0., exploration_noise, frame.feasible.shape)
            output = VectorOutput(output.edges + executor.tensor(noise[..., None]),
                                  output.offset, output.feasible)
        actions = maximum_value_actions(output, objective)
        assignments = [(encoder.worker_ids[i], state.task_ids[j])
                       for i, j in enumerate(actions) if j >= 0]
        completion_cursor = len(env.completions)
        _, reward, done, _ = env.step(assignments)
        reward = np.asarray(reward, np.float64)
        reward_sum += reward
        following_state, following = encode(step+1, frame.skills)
        if not done:
            following = replace(following, skills=reference_next_skills(
                frame.skills, available=following.workers[:, 2] > 0, rng=rng))
        event = executed_observation(frame, actions, following.workers, following.clock)
        own_completion = np.zeros((n, 3), np.float32)
        for completed in env.completions[completion_cursor:]:
            own_completion[worker_index[completed.worker_id]] += np.asarray(
                [1., env.task(completed.task_id).value, completed.travel_cost/env.d_ref], np.float32,
            )
        recorder(step, frame, following, actions, event, own_completion, reward)
        slot = step if step < replay_capacity else int(replay_rng.integers(step+1))
        if slot < replay_capacity:
            transition = VectorTransition(frame, actions.copy(), reward.copy(), following,
                                          tuple(events), event, done)
            transition.validate()
            if step < replay_capacity:
                transitions.append(transition)
                transition_steps.append(step)
            else:
                transitions[slot], transition_steps[slot] = transition, step
        events.append(event)
        history = executor.advance_history(history, event)
        step += 1
        state, frame = following_state, following
    metrics = env.final_metrics.as_array().astype(np.float64)
    if not np.allclose(reward_sum, metrics, atol=1e-10, rtol=0):
        raise ValueError("Trajectory rewards must telescope to the final metrics")
    return CollectedTrajectory(transitions, transition_steps, recorder.sequences(), {
        "TFR": float(metrics[0]), "TCR": float(metrics[1]), "MTE": float(metrics[2]),
        "OPU": float(metrics @ objective), "steps": step, "objective": objective.tolist(),
        "skill_counts": skill_counts.tolist(), "rollout_seconds": time.perf_counter()-started,
    })
