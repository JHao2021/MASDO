"""Pre-assignment context and future observations for skill discovery."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Worker7, reachable-task summary21, degree1, chosen pair21, movement2,
# own completion3, platform outcome delta3, period clocks2.
RECORD_DIM = 60
TOKEN_DIM = 2*RECORD_DIM + 2


@dataclass(frozen=True)
class WorkerSequence:
    trajectory_id: str
    step: int
    worker: int
    skill: int
    objective: np.ndarray
    context: np.ndarray
    context_available: np.ndarray
    context_assigned: np.ndarray  # -1 marks the current pre-assignment observation.
    values: np.ndarray
    available: np.ndarray
    assigned: np.ndarray

    def validate(self):
        if not self.trajectory_id or self.step < 0 or self.worker < 0:
            raise ValueError('Identified assignment step required')
        for values, mask, assigned, context in (
            (self.context, self.context_available, self.context_assigned, True),
            (self.values, self.available, self.assigned, False),
        ):
            length = len(values)
            if length < 1 or values.shape != (length, RECORD_DIM):
                raise ValueError('Nonempty context and future suffix required')
            if mask.shape != values.shape or mask.dtype != np.bool_:
                raise ValueError('Invalid continuous observation mask')
            if assigned.shape != (length,) or not np.isin(assigned, [-1, 0, 1] if context else [0, 1]).all():
                raise ValueError('Invalid assignment events')
            if not np.isfinite(values[mask]).all():
                raise ValueError('Nonfinite observed sequence values')
        if self.context_assigned[-1] != -1 or self.context_available[-1, 29:58].any():
            raise ValueError('Current context must precede the assignment and its outcomes')
        if not 0 <= self.skill < 8 or self.objective.shape != (3,):
            raise ValueError('Invalid sequence condition')
        if not np.isfinite(self.objective).all() or np.any(self.objective < 0) or not np.isclose(self.objective.sum(), 1.):
            raise ValueError('Objective must be a simplex vector')


def observation_record(frame):
    n = len(frame.workers)
    count = frame.feasible.sum(1)
    opportunity = (frame.pairs * frame.feasible[..., None]).sum(1) / np.maximum(1, count[:, None])
    values = np.zeros((n, RECORD_DIM), np.float32)
    mask = np.zeros_like(values, bool)
    values[:, :7] = frame.workers
    values[:, 7:28] = opportunity
    values[:, 28] = count / max(1, frame.pairs.shape[1])
    values[:, 58:60] = frame.clock
    mask[:, :7] = True
    mask[:, 7:28] = (count > 0)[:, None]
    mask[:, 28] = True
    mask[:, 58:60] = True
    return values, mask


class TrajectoryRecorder:
    def __init__(self, trajectory_id, context_steps=32, future_steps=8):
        if context_steps < 1 or future_steps < 1:
            raise ValueError('Positive context and future lengths required')
        self.trajectory_id = trajectory_id
        self.context_steps, self.future_steps = context_steps, future_steps
        self.records = []

    def __call__(self, step, frame, following, actions, event, own_completion, reward):
        if step != len(self.records):
            raise ValueError('Missing or repeated sequence step')
        n = len(frame.workers)
        if own_completion.shape != (n, 3) or reward.shape != (3,):
            raise ValueError('Own completions and actual platform delta required')
        current, current_mask = observation_record(frame)
        values, mask = observation_record(following)
        values[:, 29:50] = event[:, :21]
        values[:, 50:52] = event[:, 22:24]
        values[:, 52:55] = own_completion
        values[:, 55:58] = reward
        mask[:, 29:50] = (actions >= 0)[:, None]
        mask[:, 50:58] = True
        self.records.append(dict(skills=frame.skills.copy(), objective=frame.objective.copy(),
            selected=frame.workers[:, 2] > 0, current=current, current_mask=current_mask,
            values=values, mask=mask, assigned=(actions >= 0).astype(np.int64)))

    def sequences(self):
        result = []
        for step, first in enumerate(self.records):
            past = self.records[max(0, step-self.context_steps+1):step]
            future = self.records[step:step+self.future_steps]
            for worker in np.flatnonzero(first['selected']):
                # The label is z_(i,t). Later skills may change; neither predictor
                # receives them. The suffix follows the actual Stage-I policy.
                seq = WorkerSequence(self.trajectory_id, step, int(worker), int(first['skills'][worker]),
                    first['objective'].copy(),
                    np.stack([r['values'][worker] for r in past] + [first['current'][worker]]),
                    np.stack([r['mask'][worker] for r in past] + [first['current_mask'][worker]]),
                    np.asarray([r['assigned'][worker] for r in past] + [-1], np.int64),
                    np.stack([r['values'][worker] for r in future]),
                    np.stack([r['mask'][worker] for r in future]),
                    np.asarray([r['assigned'][worker] for r in future], np.int64))
                seq.validate()
                result.append(seq)
        return result


class SequenceNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(RECORD_DIM))
        self.register_buffer('scale', torch.ones(RECORD_DIM))
        self.register_buffer('fitted', torch.tensor(False))

    def fit_once(self, sequences):
        if self.fitted.item() or not sequences:
            raise ValueError('Fit normalization once on nonempty training sequences')
        for seq in sequences:
            seq.validate()
        values = np.concatenate([v for s in sequences for v in (s.context, s.values)])
        mask = np.concatenate([m for s in sequences for m in (s.context_available, s.available)])
        mean, scale = np.zeros(RECORD_DIM), np.ones(RECORD_DIM)
        for d in range(RECORD_DIM):
            observed = values[mask[:, d], d]
            if len(observed):
                mean[d], scale[d] = observed.mean(), max(.05, observed.std())
        self.mean.copy_(torch.as_tensor(mean, device=self.mean.device))
        self.scale.copy_(torch.as_tensor(scale, device=self.scale.device))
        self.fitted.fill_(True)

    def _pad(self, rows):
        b, length = len(rows), max(len(row[0]) for row in rows)
        values = np.zeros((b, length, RECORD_DIM), np.float32)
        mask = np.zeros_like(values, bool)
        assigned = np.full((b, length), -1, np.int64)
        valid = np.zeros((b, length), bool)
        for i, (v, m, a) in enumerate(rows):
            size = len(v)
            values[i, :size] = np.where(m, v, 0.)
            mask[i, :size], assigned[i, :size], valid[i, :size] = m, a, True
        device = self.mean.device
        v, m = torch.as_tensor(values, device=device), torch.as_tensor(mask, device=device)
        normalized = torch.where(m, (v-self.mean)/self.scale, 0.)
        a = torch.as_tensor(assigned, device=device)
        assignment_token = F.one_hot(a.clamp_min(0), 2).float() * (a >= 0)[..., None]
        token = torch.cat((normalized, m.float(), assignment_token), -1)
        return dict(values=normalized, mask=m, assigned=a, valid=torch.as_tensor(valid, device=device), token=token)

    def batch(self, sequences):
        if not sequences or not self.fitted.item():
            raise ValueError('Fit normalization before forming a nonempty batch')
        for seq in sequences:
            seq.validate()
        batch = self._pad([(s.values, s.available, s.assigned) for s in sequences])
        context = self._pad([(s.context, s.context_available, s.context_assigned) for s in sequences])
        batch.update(context_token=context['token'], context_valid=context['valid'],
                     objective=torch.as_tensor(np.stack([s.objective for s in sequences]),
                                               dtype=torch.float32, device=self.mean.device),
                     skill=torch.tensor([s.skill for s in sequences], device=self.mean.device))
        return batch
