"""Multi-agent vector temporal-difference learning."""
from copy import deepcopy

import torch
from torch.nn import functional as F

from .policy import joint_vector, maximum_value_actions


class VectorTDLearner:
    def __init__(self, network, learning_rate=3e-4, target_interval=32, gradient_steps=8):
        if target_interval < 1 or gradient_steps < 0:
            raise ValueError('Invalid learning schedule')
        self.online = network
        self.target = deepcopy(network).requires_grad_(False)
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        self.target_interval, self.gradient_steps = target_interval, gradient_steps
        self.updates = 0

    def prediction_and_target(self, transition):
        transition.validate()
        n = len(transition.frame.workers)
        history = self.online.replay_history(transition.history_events, n, self.gradient_steps)
        predicted = joint_vector(self.online(transition.frame, history), transition.frame, transition.actions)
        with torch.no_grad():
            target = self.online.tensor(transition.reward)
            if not transition.terminal:
                following_history = self.target.replay_history(
                    (*transition.history_events, transition.executed_event), n, 0)
                output = self.target(transition.following, following_history)
                # Use the target network for both action selection and evaluation.
                action = maximum_value_actions(output, transition.following.objective)
                target = target + joint_vector(output, transition.following, action)
        return predicted, target

    def update(self, transitions):
        if not transitions:
            raise ValueError('Empty vector TD batch')
        values = [self.prediction_and_target(t) for t in transitions]
        prediction = torch.stack([x[0] for x in values])
        target = torch.stack([x[1] for x in values])
        loss = F.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite vector TD loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        component_norms = {}
        for name, parameter in self.online.named_parameters():
            if parameter.grad is not None:
                key = name.split('.')[0]
                component_norms[key] = component_norms.get(key, 0.) + float(parameter.grad.detach().square().sum())
        component_norms = {key: value ** .5 for key, value in component_norms.items()}
        norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10., error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return dict(update=self.updates, loss=float(loss.detach()), gradient_norm=float(norm),
                    component_gradient_norms=component_norms,
                    target_component_mean=target.mean(0).cpu().tolist(),
                    prediction_component_mean=prediction.detach().mean(0).cpu().tolist(),
                    td_component_abs=(prediction.detach() - target).abs().mean(0).cpu().tolist())

    def state_dict(self):
        return dict(config=self.online.config, online=self.online.state_dict(),
                    target=self.target.state_dict(), optimizer=self.optimizer.state_dict(),
                    updates=self.updates, target_interval=self.target_interval,
                    gradient_steps=self.gradient_steps)

    def load_state_dict(self, state):
        if state['config'] != self.online.config:
            raise ValueError('Executor configuration differs from checkpoint')
        self.online.load_state_dict(state['online'])
        self.target.load_state_dict(state['target'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.updates = int(state['updates'])
        self.target_interval = int(state['target_interval'])
        self.gradient_steps = int(state['gradient_steps'])
