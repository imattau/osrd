"""Heterogeneous graph model for RIDE's train-event delay task."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HeteroConv

RIDE_EDGE_DIMS = {
    ("station", "to", "station"): 9,
    ("station", "to_rev", "station"): 9,
    ("train", "past", "station"): 6,
    ("station", "past_rev", "train"): 6,
    ("train", "future", "station"): 5,
    ("station", "future_rev", "train"): 5,
}


class RideHeterogeneousGNN(nn.Module):
    """Predict normalized delay change for each train-to-future-station edge.

    The input widths are inferred from the actual serialized RIDE feature tensors
    because the published train feature specification and tensor width differ by
    one column in the pinned Lite shard. This keeps the model honest about the
    provider's stored tensor while the mismatch is documented and resolved.
    """

    def __init__(
        self,
        train_features: int = 32,
        station_features: int = 17,
        hidden_channels: int = 128,
        layers: int = 6,
        heads: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be at least 1")
        if hidden_channels % heads:
            raise ValueError("hidden_channels must be divisible by heads")

        self.dropout = dropout
        self.node_encoders = nn.ModuleDict(
            {
                "train_encoder": nn.Sequential(
                    nn.Linear(train_features, hidden_channels), nn.ReLU()
                ),
                "station_encoder": nn.Sequential(
                    nn.Linear(station_features, hidden_channels), nn.ReLU()
                ),
            }
        )
        self.convolutions = nn.ModuleList(
            [
                HeteroConv(
                    {
                        edge_type: GATv2Conv(
                            (hidden_channels, hidden_channels),
                            hidden_channels // heads,
                            heads=heads,
                            concat=True,
                            edge_dim=edge_dim,
                            add_self_loops=False,
                            dropout=dropout,
                        )
                        for edge_type, edge_dim in RIDE_EDGE_DIMS.items()
                    },
                    aggr="mean",
                )
                for _ in range(layers)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "train_norm": nn.LayerNorm(hidden_channels),
                        "station_norm": nn.LayerNorm(hidden_channels),
                    }
                )
                for _ in range(layers)
            ]
        )
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 5 + 1, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, graph: HeteroData) -> Tensor:
        """Return one normalized delay-change prediction per future edge."""
        x_dict = {
            node_type: self.node_encoders[f"{node_type}_encoder"](graph[node_type].x)
            for node_type in ("train", "station")
        }
        edge_index_dict = {edge_type: graph[edge_type].edge_index for edge_type in RIDE_EDGE_DIMS}
        edge_attr_dict = {edge_type: graph[edge_type].edge_attr for edge_type in RIDE_EDGE_DIMS}

        for convolution, norms in zip(self.convolutions, self.norms, strict=True):
            messages = convolution(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {
                node_type: norms[f"{node_type}_norm"](
                    x_dict[node_type]
                    + torch.nn.functional.dropout(
                        torch.relu(messages[node_type]),
                        p=self.dropout,
                        training=self.training,
                    )
                )
                for node_type in x_dict
            }

        future = graph[("train", "future", "station")]
        train_index, station_index = future.edge_index
        rank = future.future_rank.to(x_dict["train"].dtype).unsqueeze(-1)
        edge_input = torch.cat(
            (
                x_dict["train"][train_index],
                x_dict["station"][station_index],
                future.edge_attr,
                rank,
            ),
            dim=-1,
        )
        return self.edge_decoder(edge_input).squeeze(-1)
