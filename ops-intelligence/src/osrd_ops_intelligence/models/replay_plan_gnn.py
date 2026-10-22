"""Graph-level scorer for action-conditioned railway timetable alternatives."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

from osrd_ops_intelligence.features.replay_graph import EDGE_FEATURE_NAMES


class ReplayPlanGNN(nn.Module):
    """Predict plan-level conflict and delay outcomes from a service interaction graph."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
        edge_channels: int = len(EDGE_FEATURE_NAMES),
        action_channels: int = 0,
    ) -> None:
        super().__init__()
        if hidden_channels % heads:
            raise ValueError("hidden_channels must be divisible by heads")
        self.dropout = dropout
        self.action_channels = action_channels
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
        readout_channels = hidden_channels * 2
        if action_channels:
            self.action_encoder = nn.Sequential(
                nn.Linear(action_channels, hidden_channels), nn.ReLU()
            )
            readout_channels += hidden_channels
        self.readout = nn.Sequential(
            nn.Linear(readout_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
        )
        self.outcome_head = nn.Linear(hidden_channels, 2)
        self.score_head = nn.Linear(hidden_channels, 1)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
        action_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        hidden = self.encoder(x)
        message = torch.relu(self.conv1(hidden, edge_index, edge_attr))
        hidden = self.norm1(
            hidden + torch.nn.functional.dropout(message, p=self.dropout, training=self.training)
        )
        message = torch.relu(self.conv2(hidden, edge_index, edge_attr))
        hidden = self.norm2(
            hidden + torch.nn.functional.dropout(message, p=self.dropout, training=self.training)
        )
        pooled_features = [global_mean_pool(hidden, batch), global_max_pool(hidden, batch)]
        if self.action_channels:
            if action_features is None or action_features.shape[-1] != self.action_channels:
                raise ValueError("This model requires matching graph-level action features")
            pooled_features.append(self.action_encoder(action_features))
        pooled = torch.cat(pooled_features, dim=-1)
        representation = self.readout(pooled)
        # Outcomes are unconstrained during fitting; clamp physical predictions at inference.
        return self.outcome_head(representation), self.score_head(representation).squeeze(-1)
