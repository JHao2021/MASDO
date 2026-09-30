"""Predict the complete future suffix from an explicit pre-assignment prefix."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from .sequences import RECORD_DIM, TOKEN_DIM


def prefix_hidden(recurrent, tokens, valid):
    packed = pack_padded_sequence(tokens, valid.sum(1).cpu(), batch_first=True, enforce_sorted=False)
    return recurrent(packed)[1]


class ConditionalSequenceDensity(nn.Module):
    def __init__(self, hidden=64, conditioned=True):
        super().__init__()
        self.conditioned = conditioned
        self.recurrent = nn.GRU(TOKEN_DIM+3+8, hidden, batch_first=True)
        self.gaussian = nn.Linear(hidden, 2*RECORD_DIM)
        self.assignment = nn.Linear(hidden, 2)

    def log_likelihood(self, batch):
        z = F.one_hot(batch['skill'], 8).float()
        if not self.conditioned:
            z = torch.zeros_like(z)
        condition = torch.cat((batch['objective'], z), -1)
        prefix = batch['context_token']
        initial = prefix_hidden(self.recurrent, torch.cat(
            (prefix, condition[:, None].expand(-1, prefix.shape[1], -1)), -1), batch['context_valid'])
        token = batch['token']
        # Predict suffix[0] from C and a zero start token. At subsequent steps
        # teacher forcing exposes only earlier suffix tokens, never the target.
        shifted = torch.cat((torch.zeros_like(token[:, :1]), token[:, :-1]), 1)
        history, _ = self.recurrent(torch.cat(
            (shifted, condition[:, None].expand(-1, token.shape[1], -1)), -1), initial)
        mean, log_std = self.gaussian(history).chunk(2, -1)
        log_std = log_std.clamp(-5., 2.)
        gaussian = -.5*((batch['values']-mean)*torch.exp(-log_std)).square()
        gaussian = gaussian - log_std - .5*math.log(2*math.pi)
        continuous = (gaussian*batch['mask']).sum(-1)
        categorical = self.assignment(history).log_softmax(-1).gather(
            -1, batch['assigned'].clamp_min(0)[..., None]).squeeze(-1)
        return ((continuous+categorical)*batch['valid']).sum(1)


class SkillSequenceDiscriminator(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.recurrent = nn.GRU(TOKEN_DIM, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 8)

    def forward(self, batch):
        initial = prefix_hidden(self.recurrent, batch['context_token'], batch['context_valid'])
        packed = pack_padded_sequence(batch['token'], batch['valid'].sum(1).cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, hidden = self.recurrent(packed, initial)
        return self.head(hidden[-1])
