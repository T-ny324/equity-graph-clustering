"""Graph convolution: H' = sigma(A_hat H W), A_hat = D^-1/2 (A+I) D^-1/2."""
from __future__ import annotations 

import math 
import torch 
import torch.nn as nn


def normalise_adjacency(A: torch.Tensor) -> torch.Tensor:
    """Symmetric normalisation with self-loops.

    Returns A_hat = D^-1/2 (A + I) D^-1/2, whose spectrum lies in (-1, 1].
    """
    A_hat = A + torch.eye(A.shape[-1], device  = A.device , dtype = A.dtype)
    d = A_hat.sum(-1).clamp(min=1e-12).pow(-0.5)
    return A_hat * d.unsqueeze(-1) * d.unsqueeze(-2)

class GraphConv(nn.Module):
    """One graph convolution layer."""

    def __init__(self, in_dim:int, out_dim: int, bias: bool ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim,out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        #  Glorot: Var(W) = 2 / ( in + out)
        a = math.sqrt(6.0 / (self.weight.shape[0] + self.weight.shape[1]))
        nn.init.uniform_(self.weight, -a ,a )
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, H: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        out = A_hat @ (H @ self.weight)
        return out if self.bias is None else out + self.bias