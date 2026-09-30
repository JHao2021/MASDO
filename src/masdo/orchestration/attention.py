"""Task attention, team attention and autoregressive skill assignment."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class LearnedAutoregressiveOrganizer(nn.Module):
    """Attention-based worker context and sequential skill sampling.

    Inputs comprise 7 worker features, 21 pair features, a 21-dimensional
    history representation, and the previous skill. Skill embeddings are fixed.
    Only available workers choose new skills; fixed workers seed the ASC summary.
    """
    def __init__(self, skill_embeddings, hidden=64):
        super().__init__()
        embeddings = torch.as_tensor(skill_embeddings).detach().clone().float()
        if embeddings.ndim != 2 or embeddings.shape[0] != 8 or embeddings.shape[1] < 1 or hidden % 4:
            raise ValueError('Expected eight nonempty skill embeddings and a four-head width')
        skill_dim = embeddings.shape[1]
        self.register_buffer('skill_embeddings', embeddings)
        self.null_previous = nn.Parameter(torch.zeros(skill_dim))
        self.worker = nn.Sequential(nn.Linear(28, hidden), nn.Tanh())
        self.pair = nn.Sequential(nn.Linear(42, hidden), nn.Tanh())
        self.task_query = nn.Linear(hidden, hidden, bias=False)
        self.team_attention = nn.MultiheadAttention(hidden, 4, batch_first=True, dropout=0.)
        self.fusion = nn.Sequential(nn.Linear(4 * hidden + skill_dim + 2, hidden), nn.Tanh())
        self.key = nn.Linear(skill_dim, hidden, bias=False)
        self.partial_key = nn.Linear(hidden, hidden, bias=False)
        self.compatibility = nn.Linear(hidden, 1, bias=False)
        self.pair_summary = nn.Sequential(nn.Linear(hidden + skill_dim, hidden), nn.Tanh())
        self.hidden = hidden
        # Start with a uniform skill distribution.
        nn.init.zeros_(self.compatibility.weight)

    def context(self, workers, pairs, feasible, history, previous, objective, clocks):
        device = self.skill_embeddings.device
        def tensor(x, dtype=torch.float32):
            return torch.as_tensor(x, dtype=dtype, device=device)
        w, p, mask, h = tensor(workers), tensor(pairs), tensor(feasible, torch.bool), tensor(history)
        prev, lam, clock = tensor(previous, torch.long), tensor(objective), tensor(clocks)
        n = len(w)
        if n < 1 or w.shape != (n, 7) or p.shape != (*mask.shape, 21) or mask.shape[0] != n:
            raise ValueError('Invalid worker/pair/feasibility shapes')
        if h.shape != (n, 21) or prev.shape != (n,) or clock.shape != (n, 2) or lam.shape != (3,):
            raise ValueError('Invalid history/previous/objective/clock shapes')
        if ((prev < -1) | (prev >= 8)).any():
            raise ValueError('Previous Skill must be -1 (null) or a K8 ID')
        if not all(torch.isfinite(x).all() for x in (w, p, h, lam, clock)):
            raise ValueError('Nonfinite context')
        local = self.worker(torch.cat((w, h), -1))
        if p.shape[1]:
            pair = self.pair(torch.cat((p, h[:, None].expand(-1, p.shape[1], -1)), -1))
            score = (pair * self.task_query(local)[:, None]).sum(-1) / self.hidden ** .5
            score = score.masked_fill(~mask, -1e9)
            weight = score.softmax(-1) * mask
            weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-12)
            pooled = (pair * weight[..., None]).sum(1)
        else:
            pooled = torch.zeros_like(local)
        team, _ = self.team_attention(local[None], local[None], local[None], need_weights=False)
        u = torch.cat((pooled, team[0]), -1)
        global_context = u.mean(0).expand(n, -1)
        previous_embedding = self.skill_embeddings[prev.clamp_min(0)]
        previous_embedding = torch.where((prev == -1)[:, None], self.null_previous, previous_embedding)
        f = self.fusion(torch.cat((u, global_context, previous_embedding, clock), -1))
        return f

    def fixed_summary(self, features, previous, order):
        """Initialize the summary with fixed skills of unavailable workers."""
        fixed = np.setdiff1d(np.arange(len(features)), order)
        if not len(fixed):
            return features.new_zeros(self.hidden), 0
        indices = torch.as_tensor(fixed, device=features.device)
        skills = torch.as_tensor(np.asarray(previous)[fixed], device=features.device)
        embeddings = self.skill_embeddings[skills.clamp_min(0)]
        embeddings = torch.where((skills < 0)[:, None], self.null_previous, embeddings)
        values = self.pair_summary(torch.cat((features[indices], embeddings), -1))
        return values.sum(0), len(fixed)

    def skill_logits(self, feature, summary):
        return self.compatibility(torch.tanh(
            feature + self.key(self.skill_embeddings) + self.partial_key(summary))).squeeze(-1)

    def sample(self, features, previous, order, rng, *, greedy=False):
        chosen = np.asarray(previous).copy()
        total, count = self.fixed_summary(features, previous, order)
        logp, entropy = features.sum()*0., features.sum()*0.
        for i in order:
            logits = self.skill_logits(features[i], total / max(1, count))
            prob = logits.softmax(-1)
            if greedy:
                choice = int(logits.argmax())
            else:
                cpu_prob = prob.detach().double().cpu().numpy()
                choice = int(rng.choice(8, p=cpu_prob / cpu_prob.sum()))
            chosen[i] = choice
            logp = logp + logits.log_softmax(-1)[choice]
            entropy = entropy - (prob * logits.log_softmax(-1)).sum()
            total = total + self.pair_summary(torch.cat((features[i], self.skill_embeddings[choice]), -1))
            count += 1
        return chosen, logp, entropy
