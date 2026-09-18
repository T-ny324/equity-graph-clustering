"""Two-layer GCN encoder: (X, A) -> Z."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.layers import GraphConv, normalise_adjacency


class GCNEncoder(nn.Module):
    """Stack of GraphConv layers producing node embeddings.

    Depth is capped at 2 by default. Each layer is one hop, and stacking
    applies A_hat repeatedly — in the graph-spectral domain that is a
    frequency response (1 - lambda)^L, which drives every component except
    DC toward zero. Deep stacks therefore collapse all embeddings onto one
    vector, which is fatal for a clustering task.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64,
                 embed_dim:int = 16, n_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [embed_dim]
        self.layers = nn.ModuleList(
            GraphConv(dims[i],dims[i+1],bias = False) for i in range(n_layers)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X:  torch.Tensor, A: torch.Tensor,
                normalise: bool = True) -> torch.Tensor:
        A_hat = normalise_adjacency(A) if normalise  else A
        H  = X
        for i,layer in enumerate(self.layers):
            H = layer(H,A_hat)
            if i < len(self.layers) - 1:
                H = self.dropout(torch.relu(H))
        return H
    
