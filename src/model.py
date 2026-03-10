"""GNN model definitions for hybrid log anomaly detection.

Contains the GAT-based graph autoencoder architecture.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class GraphAttentionEncoder(nn.Module):
    """A simple GAT-based encoder for graph-structured log data."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        out_dim: int = 32,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # Lazy import so the module can be imported without torch-geometric
        from torch_geometric.nn import GATConv

        self.conv1 = GATConv(in_channels, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class GraphAutoEncoder(nn.Module):
    """Graph Autoencoder for anomaly detection via reconstruction error."""

    def __init__(self, encoder: GraphAttentionEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        # Simple inner-product decoder
        # Reconstruction: sigmoid(Z @ Z^T)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(z @ z.t())

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, edge_index)
        return self.decode(z)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: dict[str, Any],
    path: str = "models/gat_model.pt",
) -> None:
    """Save a full training checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str = "models/gat_model.pt") -> dict[str, Any]:
    """Load a training checkpoint."""
    return torch.load(path, map_location="cpu", weights_only=False)
