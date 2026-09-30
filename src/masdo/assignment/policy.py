"""Skill-conditioned recurrent vector values and matching actions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from masdo.matching.actions import constrained_joint_action


@dataclass(frozen=True)
class Frame:
    workers: np.ndarray  # N x 7, current observable worker features; xy first.
    pairs: np.ndarray  # N x T x 21, visible worker-task service features.
    feasible: np.ndarray  # N x T, no padding/future task is a feasible edge.
    objective: np.ndarray  # fixed for an entire operating period.
    skills: np.ndarray  # N, already sampled for this actual state.
    clock: np.ndarray  # N x 2: remaining and elapsed operating-period fractions.

    def validate(self):
        n, t = self.feasible.shape
        expected = [(self.workers, (n, 7)), (self.pairs, (n, t, 21)),
                    (self.objective, (3,)), (self.skills, (n,)),
                    (self.clock, (n, 2))]
        if n < 1 or any(a.shape != shape for a, shape in expected):
            raise ValueError('Invalid frame dimensions')
        if any(not np.isfinite(a).all() for a, _ in expected):
            raise ValueError('Nonfinite frame')
        if self.feasible.dtype != np.bool_:
            raise ValueError('Feasibility must be boolean')
        if not np.issubdtype(self.skills.dtype, np.integer) or np.any((self.skills < 0) | (self.skills >= 8)):
            raise ValueError('Skills must lie in the full K=8 library')
        if np.any(self.objective < 0) or not np.isclose(self.objective.sum(), 1.):
            raise ValueError('Objective must be a simplex vector')


def executed_observation(frame: Frame, actions: np.ndarray, next_workers: np.ndarray,
                         next_clock: np.ndarray) -> np.ndarray:
    """One observed event per worker, including workers with no assignment.

    Use only the chosen CURRENT task features, observed movement, and NEXT
    worker status. No future task, service label, or next reward is an input.
    """
    validate_actions(frame, actions)
    n = len(frame.workers)
    if next_workers.shape != (n, 7) or next_clock.shape != (n, 2):
        raise ValueError('Invalid post-execution observation')
    assigned = actions >= 0
    chosen = np.zeros((n, 21), np.float32)
    ids = np.flatnonzero(assigned)
    chosen[ids] = frame.pairs[ids, actions[ids]]
    event = np.concatenate((chosen, assigned[:, None],
        next_workers[:, :2] - frame.workers[:, :2], next_workers, next_clock), -1)
    if not np.isfinite(event).all():
        raise ValueError('Nonfinite executed observation')
    return event.astype(np.float32)


def validate_actions(frame: Frame, actions: np.ndarray):
    if actions.shape != (len(frame.workers),) or not np.issubdtype(actions.dtype, np.integer):
        raise ValueError('Actions must contain one integer per worker')
    selected = actions[actions >= 0]
    if np.any(actions < -1) or np.any(selected >= frame.feasible.shape[1]):
        raise ValueError('Invalid task index; unmatched is -1')
    if len(np.unique(selected)) != len(selected):
        raise ValueError('Tasks must be exclusive')
    ids = np.flatnonzero(actions >= 0)
    if not frame.feasible[ids, actions[ids]].all():
        raise ValueError('Infeasible replay assignment')


@dataclass
class VectorOutput:
    edges: torch.Tensor  # N x T x 3; infeasible edges are zero, never selected.
    offset: torch.Tensor  # 3; action-independent state value.
    feasible: np.ndarray


class VectorSkillExecutor(nn.Module):
    def __init__(self, hidden=32, skill_dim=32, alpha=1., edge_chunk=2048):
        super().__init__()
        if hidden < 4 or skill_dim < 1 or alpha <= 0 or edge_chunk < 1:
            raise ValueError('Invalid executor configuration')
        self.config = dict(hidden=hidden, skill_dim=skill_dim, alpha=alpha, edge_chunk=edge_chunk)
        self.hidden, self.alpha, self.edge_chunk = hidden, float(alpha), edge_chunk
        self.embedding = nn.Embedding(8, skill_dim)
        self.neighbor_encoder = nn.Sequential(nn.Linear(hidden + skill_dim + 2, hidden), nn.SiLU())
        self.pair_encoder = nn.Sequential(nn.Linear(21 + 7 + hidden + 2, hidden), nn.SiLU())
        self.execution_encoder = nn.Sequential(nn.Linear(33, hidden), nn.SiLU())
        # Candidate pairs and executed events share the recurrent cell.
        self.recurrent = nn.GRUCell(hidden, hidden)
        self.base_head = nn.Linear(hidden + 3, 3)
        self.skill_head = nn.Sequential(nn.Linear(hidden + skill_dim, hidden),
                                       nn.SiLU(), nn.Linear(hidden, 3))
        self.offset_head = nn.Sequential(nn.Linear(hidden + 7 + skill_dim + 3 + 2 + 21, hidden),
                                        nn.SiLU(), nn.Linear(hidden, 3))
        self.auxiliary_offset = nn.Sequential(nn.Linear(hidden + 7 + skill_dim + 3 + 2, hidden),
                                             nn.SiLU(), nn.Linear(hidden, 1))

    @property
    def device(self):
        return self.embedding.weight.device

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def initial_history(self, workers):
        return torch.zeros((workers, self.hidden), device=self.device)

    def advance_history(self, history, observed_event):
        """Advance worker histories once after an environment step."""
        event = self.tensor(observed_event)
        if event.shape != (history.shape[0], 33):
            raise ValueError('One 33D actual event is required per worker')
        return self.recurrent(self.execution_encoder(event), history)

    def replay_history(self, observations, workers, gradient_steps=8):
        """Rebuild from period start; detach the prefix, never cache stale weights.

        All observations are replayed; only the suffix carries gradients.
        """
        if gradient_steps < 0:
            raise ValueError('gradient_steps must be nonnegative')
        history = self.initial_history(workers)
        cut = max(0, len(observations) - gradient_steps)
        with torch.no_grad():
            for event in observations[:cut]:
                history = self.advance_history(history, event)
        for event in observations[cut:]:
            history = self.advance_history(history, event)
        return history

    def _context(self, frame, history):
        w, lam, clock = self.tensor(frame.workers), self.tensor(frame.objective), self.tensor(frame.clock)
        z = torch.as_tensor(frame.skills, dtype=torch.long, device=self.device)
        emb = self.embedding(z)
        n = len(w)
        # Pool other workers; a one-worker team has zero neighbor context.
        relative = w[None, :, :2] - w[:, None, :2]
        source = torch.cat((history, emb), -1)[None].expand(n, -1, -1)
        neighbors = self.neighbor_encoder(torch.cat((source, relative), -1))
        mask = ~torch.eye(n, dtype=torch.bool, device=self.device)
        pool = (neighbors * mask[..., None]).sum(1) / max(1, n - 1)
        return w, lam, clock, emb, pool

    def forward(self, frame: Frame, history, *, skill_only=False):
        frame.validate()
        n, t = frame.feasible.shape
        if history.shape != (n, self.hidden) or history.device != self.device:
            raise ValueError('History shape/device mismatch')
        w, lam, clock, emb, pool = self._context(frame, history)
        pairs = self.tensor(frame.pairs)
        ids = np.argwhere(frame.feasible)
        chunks = []
        for start in range(0, len(ids), self.edge_chunk):
            index = torch.as_tensor(ids[start:start + self.edge_chunk], device=self.device)
            i, j = index.unbind(1)
            encoded = self.pair_encoder(torch.cat((pairs[i, j], w[i], pool[i], clock[i]), -1))
            context = self.recurrent(encoded, history[i])
            # The objective conditions the base value, not the skill adjustment.
            base = self.base_head(torch.cat((context, lam.expand(len(i), -1)), -1))
            if skill_only:
                context, base = context.detach(), base.detach()
            adjustment = self.skill_head(torch.cat((context, emb[i]), -1))
            chunks.append(base + self.alpha * adjustment)
        edges = torch.zeros((n * t, 3), device=self.device)
        if chunks:
            flat = torch.as_tensor(ids[:, 0] * t + ids[:, 1], device=self.device)
            edges = edges.index_copy(0, flat, torch.cat(chunks))
        mask = self.tensor(frame.feasible)[..., None]
        pair_pool = (pairs * mask).sum((0, 1)) / mask.sum().clamp_min(1)
        state = torch.cat((history.mean(0), w.mean(0), emb.mean(0), lam,
                           clock.mean(0), pair_pool))
        offset = self.offset_head(state)
        if skill_only:
            offset = offset.detach()
        return VectorOutput(edges.reshape(n, t, 3), offset, frame.feasible)

    def auxiliary_value(self, frame, history):
        z = torch.as_tensor(frame.skills, dtype=torch.long, device=self.device)
        state = torch.cat((history.mean(0).detach(), self.tensor(frame.workers).mean(0),
                           self.embedding(z).mean(0).detach(),
                           self.tensor(frame.objective), self.tensor(frame.clock).mean(0)))
        return self.auxiliary_offset(state).squeeze(-1)


def joint_vector(output: VectorOutput, frame: Frame, actions: np.ndarray):
    validate_actions(frame, actions)
    rows = np.flatnonzero(actions >= 0)
    return output.offset + output.edges[rows, actions[rows]].sum(0)


def maximum_value_actions(output: VectorOutput, objective: np.ndarray):
    """Maximize projected pair values while allowing unmatched workers."""
    scores = (output.edges.detach().cpu().numpy().astype(np.float64) @ objective)
    if not np.isfinite(scores).all():
        raise FloatingPointError('Nonfinite projected Q')
    # Zero denotes an unmatched solver slot, not a learned idle value.
    joint = constrained_joint_action(scores, np.zeros(len(scores)), output.feasible)
    actions = joint.action_indices.copy()
    actions[actions == scores.shape[1]] = -1
    return actions


def reference_next_skills(skills, *, available, rng):
    """Resample available workers each step; retain all other skills."""
    result = skills.copy()
    indices = np.flatnonzero(available)
    result[indices] = rng.integers(0, 8, size=len(indices), dtype=np.int64)
    return result


@dataclass(frozen=True)
class VectorTransition:
    frame: Frame
    actions: np.ndarray
    reward: np.ndarray  # Actual 3D environment metric increments.
    following: Frame  # Includes the actual sampled next skill assignment.
    history_events: tuple  # Actual events strictly BEFORE frame.
    executed_event: np.ndarray  # The one event between frame and following.
    terminal: bool  # End of the operating period.

    def validate(self):
        self.frame.validate()
        self.following.validate()
        validate_actions(self.frame, self.actions)
        if self.reward.shape != (3,) or not np.isfinite(self.reward).all():
            raise ValueError('TD requires real three-component rewards')
        if not np.array_equal(self.frame.objective, self.following.objective):
            raise ValueError('Objective changed within a period')
        fixed = self.following.workers[:, 2] == 0
        if not np.array_equal(self.frame.skills[fixed], self.following.skills[fixed]):
            raise ValueError('Unavailable workers must retain their skills')
        if len(self.frame.workers) != len(self.following.workers):
            raise ValueError('Worker identity/count must persist')
        shape = (len(self.frame.workers), 33)
        if any(x.shape != shape or not np.isfinite(x).all()
               for x in (*self.history_events, self.executed_event)):
            raise ValueError('Invalid recurrent replay event')
