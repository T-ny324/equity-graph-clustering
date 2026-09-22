"""Phase 4b objectives: Deep Graph Infomax, plus an optional volatility head.

Reads  : nothing. Pure model code; the walk-forward loop that drives it is
         src/models/train.py.

Deep Graph Infomax (Velickovic et al., ICLR 2019)
--------------------------------------------------
Reference: docs/phase4_maths_reference.md, section 4.2; the volatility head
is section 4.3.

A self-supervised objective for the encoder f: (X, A_hat) -> Z. Corrupt the
features by permuting rows, X_tilde = P X, with the graph A left untouched,
and encode both:

    Z       = f(X, A_hat)            real
    Z_tilde = f(X_tilde, A_hat)      corrupted

Form the graph summary s = sigmoid(mean over nodes of Z) and score every
embedding against it with a bilinear discriminator D(z, s) = sigmoid(z^T W s).
The loss is the binary cross-entropy of a joint classification -- real
embeddings labelled 1, corrupted embeddings labelled 0:

    L = -(1/2N) [ sum_i log D(z_i, s) + sum_i log(1 - D(z_tilde_i, s)) ]

Why row-shuffling works
-----------------------
Row-shuffling preserves the marginal feature distribution exactly -- every
volatility and beta is still present, merely assigned to the wrong asset. The
encoder therefore cannot separate real from corrupted by feature values
alone; the only usable signal is whether a node's features are consistent
with its graph position. Winning the discrimination task requires learning
the feature-structure relationship, which is precisely what Z should encode.

The auxiliary volatility head
-----------------------------
A linear head y_hat = w^T z + b predicting the cross-sectionally standardised
realised volatility over (t, t+1], trained by mean squared error, so that the
total loss is L = L_dgi + lambda * L_vol.

Why volatility and not returns: volatility is persistent and predictable;
monthly returns are close to unpredictable. Using returns as the auxiliary
target would fit noise and invite an unearned alpha claim. Volatility
provides a genuine learnable signal and a defensible out-of-sample number.

`training.vol_lambda` defaults to 0.0, so the head is built and wired but
contributes nothing until enabled. DGI-alone versus DGI-plus-vol is therefore
a one-line config ablation rather than a code change.

Leakage rule
------------
The volatility target is realised AFTER t. It may appear only in the loss
term -- never in the encoder input, never in X, never in the graph. It is
built once, up front (`train.build_vol_target`), stored separately from X,
and consumed at exactly one call site in `train.train_window`.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def corrupt(X: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Row-permute the feature matrix: the DGI negative sample.

    Parameters
    ----------
    X : torch.Tensor
        Node features for one date, [N, F].
    generator : torch.Generator
        Source of the permutation, so corruption is reproducible under a seed.

    Returns
    -------
    torch.Tensor
        `X[perm]` for a uniformly random permutation `perm` of the rows. Same
        shape and the same multiset of rows as `X`; only the assignment of
        rows to nodes changes.
    """
    perm = torch.randperm(X.shape[0], generator=generator)
    return X[perm]


class Discriminator(nn.Module):
    """Bilinear scorer D(z, s) = z^T W s, returned as a logit.

    Parameters
    ----------
    embed_dim : int
        Dimension d of both the node embeddings and the summary vector.

    Notes
    -----
    `forward` returns the logit, NOT the sigmoid. `DGILoss` feeds it to
    `binary_cross_entropy_with_logits`, which is numerically stable where a
    separate `log(sigmoid(.))` is not.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Glorot: Var(W) = 2 / (d + d) = 1 / d.
        a = math.sqrt(6.0 / (self.weight.shape[0] + self.weight.shape[1]))
        nn.init.uniform_(self.weight, -a, a)

    def forward(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Logits `z @ W @ s`.

        Parameters
        ----------
        z : torch.Tensor
            Node embeddings, [N, d].
        s : torch.Tensor
            Graph summary, [d].

        Returns
        -------
        torch.Tensor
            One logit per node, [N].
        """
        return z @ (self.weight @ s)


class DGILoss(nn.Module):
    """Deep Graph Infomax loss; owns the discriminator it trains.

    Parameters
    ----------
    embed_dim : int
        Dimension of the embeddings produced by the encoder.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.discriminator = Discriminator(embed_dim)

    def forward(self, Z: torch.Tensor, Z_corrupt: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy of real-vs-corrupted against the summary of Z.

        Parameters
        ----------
        Z : torch.Tensor
            Embeddings of the real features, [N, d].
        Z_corrupt : torch.Tensor
            Embeddings of the row-shuffled features on the same graph, [N, d].

        Returns
        -------
        torch.Tensor
            Scalar loss. At initialisation the discriminator cannot tell the
            two apart, so the value sits at log(2) ~ 0.693.
        """
        # The summary is computed from the REAL embeddings only, as in the
        # paper; the corrupted set is scored against it.
        s = torch.sigmoid(Z.mean(dim=0))
        logits = torch.cat([self.discriminator(Z, s),
                            self.discriminator(Z_corrupt, s)])
        labels = torch.cat([torch.ones(Z.shape[0]),
                            torch.zeros(Z_corrupt.shape[0])]).to(logits)
        return F.binary_cross_entropy_with_logits(logits, labels)


class VolHead(nn.Module):
    """Linear head y_hat = w^T z + b: one volatility prediction per node.

    Parameters
    ----------
    embed_dim : int
        Dimension of the embeddings it reads.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Predictions of shape [N] for embeddings of shape [N, d]."""
        return self.linear(Z).squeeze(-1)


def masked_mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean squared error over the finite entries of `y` only.

    Parameters
    ----------
    y_hat : torch.Tensor
        Predictions, [N].
    y : torch.Tensor
        Targets, [N]; NaN marks "no target" and is excluded from the mean.

    Returns
    -------
    torch.Tensor
        Scalar loss. Zero -- but still attached to the graph -- when no entry
        of `y` is finite, which is the final rebalance date, whose forward
        window does not exist.
    """
    mask = torch.isfinite(y)
    if not bool(mask.any()):
        return (y_hat * 0.0).sum()
    return ((y_hat[mask] - y[mask]) ** 2).mean()
