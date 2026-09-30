"""Actor-critic learning from complete operating periods."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch

from .policy import OrganizationFrame


@dataclass(frozen=True)
class SkillTransition:
    frame: OrganizationFrame
    skills: np.ndarray
    order: np.ndarray
    reward: np.ndarray  # Actual m_(t+1) - m_t.
    following: OrganizationFrame
    terminal: bool  # operating-period end

    def validate(self):
        if self.reward.shape != (3,) or not np.isfinite(self.reward).all():
            raise ValueError('Reward must be real TFR/TCR/MTE increments')
        if not np.array_equal(self.frame.objective, self.following.objective):
            raise ValueError('Objective changed within the operating period')
        if not np.array_equal(self.following.previous, self.skills):
            raise ValueError('Next organizer context must contain the actual previous skills')


class SkillTDLearner:
    def __init__(self, organizer, learning_rate=3e-4, entropy_weight=.01, target_interval=32,
                 warmup_updates=32, warmup_learning_rate=3e-3):
        if target_interval < 1 or learning_rate <= 0 or entropy_weight < 0:
            raise ValueError('Invalid skill learning schedule')
        if warmup_updates < 0 or warmup_learning_rate <= 0:
            raise ValueError('Invalid warm-start learning schedule')
        self.online, self.target = organizer, deepcopy(organizer).requires_grad_(False)
        self.optimizer = torch.optim.Adam(organizer.parameters(), lr=learning_rate)
        self.config = dict(learning_rate=learning_rate, entropy_weight=entropy_weight,
                           target_interval=target_interval, warmup_updates=warmup_updates,
                           warmup_learning_rate=warmup_learning_rate)
        self.updates = 0

    def target_return(self, transition):
        transition.validate()
        with torch.no_grad():
            target = self.online.skill_embeddings.new_tensor(float(transition.reward @ transition.frame.objective))
            if not transition.terminal:
                target += self.target.critic(self.target.encode(transition.following).mean(0)).squeeze(-1)
        return target

    def update(self, transitions):
        learning_rate = (self.config['warmup_learning_rate'] if self.updates < self.config['warmup_updates']
                         else self.config['learning_rate'])
        for group in self.optimizer.param_groups:
            group['lr'] = learning_rate
        self.optimizer.zero_grad(set_to_none=True)
        totals = dict(actor_loss=0., critic_loss=0., advantage=0., target=0.)
        count = 0
        # Accumulate gradients with a fixed acting policy and target critic.
        # Consume transitions lazily so full-period feature tensors need not be retained.
        for transition in transitions:
            target = self.target_return(transition)
            logp, entropy, value, _ = self.online.evaluate(transition.frame, transition.skills, transition.order)
            delta = target-value
            actor_loss = -logp*delta.detach()-self.config['entropy_weight']*entropy
            critic_loss = delta.square()
            loss = actor_loss+critic_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite skill TD objective')
            loss.backward()
            totals['actor_loss'] += float(actor_loss.detach())
            totals['critic_loss'] += float(critic_loss.detach())
            totals['advantage'] += float(delta.detach())
            totals['target'] += float(target)
            count += 1
        if not count:
            raise ValueError('An operating period must contain transitions')
        component_norms = {}
        for name, parameter in self.online.named_parameters():
            if parameter.grad is not None:
                parameter.grad.div_(count)
                key = name.split('.')[0]
                component_norms[key] = component_norms.get(key,0.) + float(parameter.grad.detach().square().sum())
        norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10., error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config['target_interval'] == 0:
            self.target.load_state_dict(self.online.state_dict())
        return dict(update=self.updates, transitions=count, **{key: value/count for key, value in totals.items()},
                    gradient_norm=float(norm), learning_rate=learning_rate,
                    component_gradient_norms={k:v**.5 for k,v in component_norms.items()})

    def state_dict(self):
        return dict(config=self.config, online=self.online.state_dict(), target=self.target.state_dict(),
                    optimizer=self.optimizer.state_dict(), updates=self.updates)

    def load_state_dict(self, saved):
        if saved['config'] != self.config:
            raise ValueError('Skill learner configuration mismatch')
        for name in ('online','target','optimizer'):
            getattr(self, name).load_state_dict(saved[name])
        self.updates = saved['updates']
