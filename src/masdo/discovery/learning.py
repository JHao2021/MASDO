"""Prediction fitting, skill bonuses, and skill-directed TD updates."""
from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from masdo.assignment.policy import maximum_value_actions
from .sequences import SequenceNormalizer
from .predictors import ConditionalSequenceDensity, SkillSequenceDiscriminator


class SequenceDiscovery:
    def __init__(self, hidden=64, learning_rate=3e-4, device='cpu',
                 beta_behavior=.001, beta_future=.001):
        if learning_rate <= 0 or min(beta_behavior, beta_future) < 0:
            raise ValueError('Invalid discovery configuration')
        self.config = dict(hidden=hidden, learning_rate=learning_rate,
                           beta_behavior=beta_behavior, beta_future=beta_future)
        self.normalizer = SequenceNormalizer().to(device)
        ps = ConditionalSequenceDensity(hidden, True).to(device)
        p0 = deepcopy(ps)
        p0.conditioned = False  # equal initial weights/capacity, only Z condition differs
        self.models = nn.ModuleDict(dict(ps=ps, p0=p0, discriminator=SkillSequenceDiscriminator(hidden).to(device)))
        self.optimizer = torch.optim.Adam(self.models.parameters(), lr=learning_rate)
        self.updates = 0

    def fit(self, sequences, *, calibration_sequences=None):
        calibration = sequences if calibration_sequences is None else calibration_sequences
        if not self.normalizer.fitted.item():
            self.normalizer.fit_once(calibration)
        batch = self.normalizer.batch(sequences)
        length = batch['valid'].sum(1).clamp_min(1)
        nll_s = -(self.models['ps'].log_likelihood(batch)/length).mean()
        nll_0 = -(self.models['p0'].log_likelihood(batch)/length).mean()
        ce = F.cross_entropy(self.models['discriminator'](batch), batch['skill'])
        loss = nll_s+nll_0+ce
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite sequence fit loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.models.parameters(), 10., error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        return dict(fit_update=self.updates,
                    nll_skill=float(nll_s.detach()), nll_reference=float(nll_0.detach()),
                    discriminator_ce=float(ce.detach()), fit_gradient_norm=float(norm))

    @torch.no_grad()
    def score(self, sequences):
        if not self.normalizer.fitted.item():
            raise ValueError('Fit train-only normalizer before scoring')
        batch = self.normalizer.batch(sequences)
        ll_s = self.models['ps'].log_likelihood(batch)
        ll_0 = self.models['p0'].log_likelihood(batch)
        length = batch['valid'].sum(1)
        future = torch.where(length > 0, (ll_s-ll_0)/length.clamp_min(1), 0.)
        logits = self.models['discriminator'](batch)
        behavior = math.log(8) + logits.log_softmax(-1).gather(1, batch['skill'][:, None]).squeeze(1)
        bonus = self.config['beta_behavior']*behavior + self.config['beta_future']*future
        if not torch.isfinite(bonus).all():
            raise FloatingPointError('Nonfinite discovery bonus')
        return dict(bonus=bonus.cpu().numpy(), behavior=behavior.cpu().numpy(), future=future.cpu().numpy(),
                    suffix_length=length.cpu().numpy(), fit_update=self.updates,
                    discriminator_correct=(logits.argmax(1)==batch['skill']).cpu().numpy())

    def state_dict(self):
        return dict(config=self.config, normalizer=self.normalizer.state_dict(), models=self.models.state_dict(),
                    optimizer=self.optimizer.state_dict(), updates=self.updates)

    def load_state_dict(self, saved):
        if saved['config'] != self.config:
            raise ValueError('Discovery configuration mismatch')
        for name in ('normalizer', 'models', 'optimizer'):
            getattr(self, name).load_state_dict(saved[name])
        self.updates = saved['updates']


def auxiliary_prediction(network, frame, actions, history):
    output = network(frame, history, skill_only=True)
    rows = np.flatnonzero(actions >= 0)
    # The auxiliary scalar value uses its own offset, separate from vector TD.
    edge = output.edges[rows, actions[rows]].sum(0) @ network.tensor(frame.objective)
    return network.auxiliary_value(frame, history) + edge


class AuxiliaryTDLearner:
    def __init__(self, network, learning_rate=3e-4, target_interval=32):
        self.online, self.target = network, deepcopy(network).requires_grad_(False)
        self.parameters = list(network.embedding.parameters()) + list(network.skill_head.parameters()) + list(network.auxiliary_offset.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=learning_rate)
        self.target_interval, self.updates = target_interval, 0

    def prediction_and_target(self, transition, discovery_bonus):
        transition.validate()
        if not np.isfinite(discovery_bonus):
            raise ValueError('Invalid auxiliary bonus')
        n = len(transition.frame.workers)
        with torch.no_grad():
            history = self.online.replay_history(transition.history_events, n, 0)
        prediction = auxiliary_prediction(self.online, transition.frame, transition.actions, history)
        with torch.no_grad():
            reward = float(transition.reward @ transition.frame.objective) + discovery_bonus
            target = self.online.tensor(reward)
            if not transition.terminal:
                future = self.target.replay_history((*transition.history_events, transition.executed_event), n, 0)
                output = self.target(transition.following, future)
                actions = maximum_value_actions(output, transition.following.objective)
                target += auxiliary_prediction(self.target, transition.following, actions, future)
        return prediction, target

    def update(self, transitions, bonuses):
        if not transitions or len(transitions) != len(bonuses):
            raise ValueError('Nonempty paired auxiliary replay required')
        pairs = [self.prediction_and_target(t, b) for t, b in zip(transitions, bonuses)]
        loss = F.mse_loss(torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs]))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite auxiliary TD loss')
        # Clear stale gradients before the skill-only auxiliary update.
        self.online.zero_grad(set_to_none=True)
        loss.backward()
        norms = {}
        for name, parameter in self.online.named_parameters():
            if parameter.grad is not None:
                if name.split('.')[0] not in ('embedding', 'skill_head', 'auxiliary_offset'):
                    raise RuntimeError('Auxiliary gradient escaped skill-only ownership')
                key = name.split('.')[0]
                norms[key] = norms.get(key, 0.) + float(parameter.grad.detach().square().sum())
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, 10., error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return dict(auxiliary_update=self.updates, auxiliary_loss=float(loss.detach()),
                    auxiliary_gradient_norm=float(norm),
                    auxiliary_component_gradient_norms={k: v**.5 for k, v in norms.items()})

    def state_dict(self):
        return dict(target=self.target.state_dict(), optimizer=self.optimizer.state_dict(),
                    target_interval=self.target_interval, updates=self.updates)

    def load_state_dict(self, saved):
        self.target.load_state_dict(saved['target'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.target_interval, self.updates = saved['target_interval'], saved['updates']
