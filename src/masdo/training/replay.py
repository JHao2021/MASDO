"""Bounded cross-round replay with tensor-only checkpoint serialization."""
from dataclasses import fields, is_dataclass

import numpy as np
import torch

from masdo.assignment.policy import Frame, VectorTransition
from masdo.discovery.sequences import WorkerSequence


_RECORDS = {c.__name__: c for c in (Frame, VectorTransition, WorkerSequence)}


def _pack(value):
    if is_dataclass(value):
        return {'record': type(value).__name__, 'fields': {
            field.name: _pack(getattr(value, field.name)) for field in fields(value)}}
    if isinstance(value, np.ndarray):
        return {'array': torch.from_numpy(value.copy())}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {'tuple': [_pack(x) for x in value]}
    if isinstance(value, list):
        return [_pack(x) for x in value]
    return value


def _unpack(value):
    if isinstance(value, list):
        return [_unpack(x) for x in value]
    if isinstance(value, dict):
        if 'record' in value:
            return _RECORDS[value['record']](**{k: _unpack(v) for k, v in value['fields'].items()})
        if 'array' in value:
            return value['array'].cpu().numpy().copy()
        if 'tuple' in value:
            return tuple(_unpack(x) for x in value['tuple'])
    return value


class TrainingReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.fit_sequences, self.transitions, self.bonuses = [], [], []

    def add_fit(self, sequences, rng):
        candidates = self.fit_sequences + list(sequences)
        if len(candidates) > self.capacity:
            indices = rng.choice(len(candidates), self.capacity, replace=False)
            candidates = [candidates[int(i)] for i in indices]
        self.fit_sequences = candidates

    def add_policy(self, transitions, bonuses):
        if len(transitions) != len(bonuses):
            raise ValueError('One attributed discovery bonus per transition required')
        self.transitions = (self.transitions + list(transitions))[-self.capacity:]
        self.bonuses = (self.bonuses + list(bonuses))[-self.capacity:]

    def state_dict(self):
        return dict(capacity=self.capacity, fit_sequences=_pack(self.fit_sequences),
                    transitions=_pack(self.transitions), bonuses=list(self.bonuses))

    def load_state_dict(self, state):
        if state['capacity'] != self.capacity:
            raise ValueError('Replay capacity differs from checkpoint')
        self.fit_sequences = _unpack(state['fit_sequences'])
        self.transitions = _unpack(state['transitions'])
        self.bonuses = list(state['bonuses'])
