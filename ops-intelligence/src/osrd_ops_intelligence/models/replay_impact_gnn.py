"""Topology-conditioned GNN for paired OSRD replay impact estimates."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATv2Conv

from osrd_ops_intelligence.features.replay_graph import EDGE_FEATURE_NAMES


class ReplayImpactGNN(nn.Module):
    """Estimate per-train arrival delta and conflict risk in one replay episode."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 32,
        heads: int = 2,
        dropout: float = 0.1,
        edge_channels: int = len(EDGE_FEATURE_NAMES),
    ) -> None:
        super().__init__()
        if hidden_channels % heads:
            raise ValueError("hidden_channels must be divisible by heads")
        self.dropout = dropout
        self.encoder = nn.Sequential(nn.Linear(input_channels, hidden_channels), nn.ReLU())
        self.conv1 = GATv2Conv(
            hidden_channels,
            hidden_channels // heads,
            heads=heads,
            edge_dim=edge_channels,
            add_self_loops=False,
            dropout=dropout,
        )
        self.conv2 = GATv2Conv(
            hidden_channels,
            hidden_channels // heads,
            heads=heads,
            edge_dim=edge_channels,
            add_self_loops=False,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 2),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        hidden = self.encoder(x)
        message = torch.relu(self.conv1(hidden, edge_index, edge_attr))
        hidden = self.norm1(
            hidden + torch.nn.functional.dropout(message, p=self.dropout, training=self.training)
        )
        message = torch.relu(self.conv2(hidden, edge_index, edge_attr))
        hidden = self.norm2(
            hidden + torch.nn.functional.dropout(message, p=self.dropout, training=self.training)
        )
        return self.head(hidden)
