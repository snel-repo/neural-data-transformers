#!/usr/bin/env python3
# Author: Joel Ye
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
from torch.distributions import Poisson

from src.utils import binary_mask_to_attn_mask
from src.mask import UNMASKED_LABEL

class RatesOracle(nn.Module):

    def __init__(self, config, num_neurons, device, **kwargs):
        super().__init__()
        assert config.REQUIRES_RATES == True, "Oracle requires rates"
        if config.LOSS.TYPE == "poisson":
            self.classifier = nn.PoissonNLLLoss(reduction='none', log_input=config.LOGRATE)
        else:
            raise Exception(f"Loss type {config.LOSS_TYPE} not supported")

    def get_hidden_size(self):
        return 0

    def forward(self, src, mask_labels, rates=None, **kwargs):
        # output is t x b x neurons (rate predictions)
        loss = self.classifier(rates, mask_labels)
        # Mask out losses unmasked labels
        masked_loss = loss[mask_labels != UNMASKED_LABEL]
        masked_loss = masked_loss.mean()
        return (
            masked_loss.unsqueeze(0),
            rates,
            None,
            torch.tensor(0, device=masked_loss.device, dtype=torch.float),
            None,
            None,
        )

class RandomModel(nn.Module):
    # Guess a random rate in LOGRATE_RANGE
    # Purpose - why is our initial loss so close to our final loss
    LOGRATE_RANGE = (-2.5, 2.5)

    def __init__(self, config, num_neurons, device):
        super().__init__()
        self.device = device
        if config.LOSS.TYPE == "poisson":
            self.classifier = nn.PoissonNLLLoss(reduction='none', log_input=config.LOGRATE)
        else:
            raise Exception(f"Loss type {config.LOSS_TYPE} not supported")

    def forward(self, src, mask_labels, *args, **kwargs):
        # output is t x b x neurons (rate predictions)
        rates = torch.rand(mask_labels.size(), dtype=torch.float32).to(self.device)
        rates *= (self.LOGRATE_RANGE[1] - self.LOGRATE_RANGE[0])
        rates += self.LOGRATE_RANGE[0]
        loss = self.classifier(rates, mask_labels)
        # Mask out losses unmasked labels
        loss[mask_labels == UNMASKED_LABEL] = 0.0

        return loss.mean(), rates
