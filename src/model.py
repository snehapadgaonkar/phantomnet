"""
model.py
---------
Defines the PhantomNet deep neural network used for behavioral fingerprint classification.
"""

import torch
import torch.nn as nn


class PhantomNet(nn.Module):
    """Simple configurable feed-forward model."""

    def __init__(self, input_dim=77, hidden_dim=128, num_layers=2, dropout=0.3, num_classes=2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
