"""Per-step goal-conditioned skill selection."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .attention import LearnedAutoregressiveOrganizer


@dataclass(frozen=True)
class OrganizationFrame:
    workers: np.ndarray
    pairs: np.ndarray
    feasible: np.ndarray
    history: np.ndarray
    previous: np.ndarray
    objective: np.ndarray
    clocks: np.ndarray

    @property
    def available(self):
        return np.flatnonzero(self.workers[:, 2] > 0)


def conditional_chain(actor, features, previous, actions, order):
    """Re-evaluate new choices, conditioned on fixed and preceding skills."""
    total, count = actor.fixed_summary(features, previous, order)
    distributions = []
    for i in order:
        logits = actor.skill_logits(features[i], total / max(1, count))
        distributions.append(logits.log_softmax(-1))
        total = total + actor.pair_summary(torch.cat(
            (features[i], actor.skill_embeddings[int(actions[i])]), -1))
        count += 1
    return torch.stack(distributions) if distributions else features.new_empty((0, 8))


class SkillOrganizer(LearnedAutoregressiveOrganizer):
    def __init__(self, skill_embeddings, executor_hidden, hidden=64, *, with_critic=True):
        super().__init__(skill_embeddings, hidden)
        self.executor_hidden = executor_hidden
        self.history_projection = nn.Linear(executor_hidden, 21)
        self.objective_encoder = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, 2*hidden))
        nn.init.zeros_(self.objective_encoder[-1].weight)
        nn.init.zeros_(self.objective_encoder[-1].bias)
        if with_critic:
            self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def encode(self, frame):
        h = torch.as_tensor(frame.history, dtype=torch.float32, device=self.skill_embeddings.device).detach()
        if h.shape != (len(frame.workers), self.executor_hidden):
            raise ValueError('Expected frozen assignment-network histories')
        history = self.history_projection(h)
        features = self.context(frame.workers, frame.pairs, frame.feasible, history,
                                frame.previous, frame.objective, frame.clocks)
        objective = torch.as_tensor(frame.objective, dtype=features.dtype, device=features.device)
        gain, shift = self.objective_encoder(objective).chunk(2, -1)
        return features * (1 + gain.tanh()) + shift

    def decide(self, frame, rng, *, greedy=False):
        features = self.encode(frame)
        order = rng.permutation(frame.available)
        skills, logp, entropy = self.sample(features, frame.previous, order, rng, greedy=greedy)
        # Acting never evaluates the training-only critic.
        return skills, order, logp, entropy

    def evaluate(self, frame, skills, order):
        order, skills = np.asarray(order), np.asarray(skills)
        if (order.shape != frame.available.shape or set(order.tolist()) != set(frame.available.tolist())
                or skills.shape != frame.previous.shape or np.any((skills < 0) | (skills >= 8))):
            raise ValueError('Order must contain exactly the available workers')
        fixed = frame.workers[:, 2] == 0
        if not np.array_equal(skills[fixed], frame.previous[fixed]):
            raise ValueError('Unavailable workers must retain their skills')
        features = self.encode(frame)
        log_probs = conditional_chain(self, features, frame.previous, skills, order)
        if len(order):
            chosen = torch.as_tensor(skills[order], dtype=torch.long, device=features.device)
            logp = log_probs.gather(1, chosen[:, None]).sum()
            entropy = -(log_probs.exp()*log_probs).sum()
        else:
            logp = entropy = features.sum()*0.
        value = self.critic(features.mean(0)).squeeze(-1)
        return logp, entropy, value, log_probs
