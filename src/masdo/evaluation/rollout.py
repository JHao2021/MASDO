"""Run an operating period with skill orchestration and constrained matching."""
import time

import numpy as np
import torch

from masdo.environment.simulator import DistanceLimitedEnv
from masdo.features.encoding import StateEncoder
from masdo.features.service import service_inputs
from masdo.assignment.policy import Frame, executed_observation, maximum_value_actions
from masdo.orchestration.policy import OrganizationFrame
from masdo.orchestration.learning import SkillTransition


def rollout(scenario, environment, grid, distance, objective, seed, executor, organizer, *,
            learner=None, on_update=None):
    if learner is not None and any(p.requires_grad for p in executor.parameters()):
        raise ValueError('Assignment parameters and skills must be frozen in Stage II')
    if learner is not None and learner.online is not organizer:
        raise ValueError('Training must update the acting organizer')
    objective = np.asarray(objective, np.float64)
    rng = np.random.default_rng(seed)
    cfg = environment
    env = DistanceLimitedEnv(scenario, travel_speed=cfg['travel_speed'], d_ref=cfg['d_ref'],
                             grid=grid, max_distance=distance)
    encoder = StateEncoder(tuple(sorted(w.worker_id for w in scenario.workers)), scenario.horizon,
        int(cfg.get('encoder_max_tasks', cfg['max_tasks'])), cfg['max_task_value'], region_count=grid.num_regions)
    n = len(encoder.worker_ids)
    history = executor.initial_history(n)
    previous = np.zeros(n, np.int64)
    reward_sum = np.zeros(3, np.float64)
    step, decision_seconds = 0, 0.
    compositions, step_rewards = [], []
    started = time.perf_counter()

    def clocks():
        return np.broadcast_to([max(0., 1-step/scenario.horizon), step/scenario.horizon],
                               (n, 2)).copy().astype(np.float32)

    def encode():
        state = encoder.encode(env)
        return state, service_inputs(env, state), state.feasible[:, :len(state.task_ids)].copy()

    def organization_frame(state, pairs, mask):
        return OrganizationFrame(state.worker_features.copy(), pairs.copy(), mask.copy(),
            history.detach().cpu().numpy().copy(), previous.copy(), objective.copy(), clocks())

    def trajectory():
        nonlocal history, previous, step, decision_seconds
        state, pairs, mask = encode()
        while not env.done:
            tick = time.perf_counter()
            current = organization_frame(state, pairs, mask)
            with torch.no_grad():
                skills, order, _, _ = organizer.decide(current, rng)
            previous = skills.copy()
            compositions.append(np.bincount(skills[order], minlength=8).tolist())
            frame = Frame(state.worker_features.copy(), pairs.copy(), mask.copy(), objective.copy(), skills.copy(), clocks())
            with torch.no_grad():
                action = maximum_value_actions(executor(frame, history), objective)
            assignments = [(encoder.worker_ids[i], state.task_ids[j]) for i, j in enumerate(action) if j >= 0]
            decision_seconds += time.perf_counter()-tick
            _, reward, done, _ = env.step(assignments)
            reward = np.asarray(reward, np.float64)
            reward_sum[:] += reward
            step_rewards.append(reward.tolist())
            step += 1
            following, next_pairs, next_mask = encode()
            event = executed_observation(frame, action, following.worker_features, clocks())
            with torch.no_grad():
                history = executor.advance_history(history, event)
            next_frame = organization_frame(following, next_pairs, next_mask)
            yield SkillTransition(current, skills.copy(), order.copy(), reward.copy(), next_frame, done)
            state, pairs, mask = following, next_pairs, next_mask

    if learner is None:
        for transition in trajectory():
            transition.validate()
    else:
        receipt = learner.update(trajectory())
        if on_update is not None:
            on_update(dict(step_end=step, terminal=True, **receipt))
    metrics = env.final_metrics.as_array().astype(np.float64)
    if not np.allclose(metrics, reward_sum, atol=1e-10, rtol=0):
        raise ValueError('Vector rewards must telescope to final TFR/TCR/MTE')
    return dict(C=float(metrics[0]), G=float(metrics[1]), E=float(metrics[2]), U=float(metrics@objective),
        steps=step, compositions=compositions, skill_counts=np.sum(compositions, axis=0).tolist(),
        step_rewards=step_rewards, wall_seconds=time.perf_counter()-started,
        decision_seconds=decision_seconds, training_updates=int(learner is not None))
