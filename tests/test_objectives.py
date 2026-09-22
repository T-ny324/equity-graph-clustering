"""Tests for the Phase 4b objectives, the volatility target and the loop.

Everything is built from small synthetic tensors and frames. Nothing reads
data/processed/*: a test that reads real data is a description of today's
dataset, not a test of the code.

Run from repo root:  python -m pytest tests/test_objectives.py -v
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.encoder import GCNEncoder
from src.models.layers import normalise_adjacency
from src.models.objectives import (
    DGILoss, Discriminator, VolHead, corrupt, masked_mse,
)
from src.models.train import (
    build_vol_target, forward_realised_vol, precompute_A_hat, walk_forward,
)

TRADING_DAYS = 252


# ----------------------------------------------------------------------
# corrupt
# ----------------------------------------------------------------------
def test_corrupt_is_a_row_permutation() -> None:
    """Same rows, different order: sorting the rows recovers the input."""
    X = torch.arange(24, dtype=torch.float32).reshape(8, 3)   # rows sorted by col 0
    Xc = corrupt(X, torch.Generator().manual_seed(0))

    assert Xc.shape == X.shape
    assert torch.equal(Xc[Xc[:, 0].argsort()], X)
    assert not torch.equal(Xc, X), "with N=8 and a fixed seed a row must move"


def test_corrupt_is_reproducible_under_seed() -> None:
    X = torch.randn(50, 4, generator=torch.Generator().manual_seed(3))
    a = corrupt(X, torch.Generator().manual_seed(7))
    b = corrupt(X, torch.Generator().manual_seed(7))
    c = corrupt(X, torch.Generator().manual_seed(8))

    assert torch.equal(a, b)
    assert not torch.equal(a, c)


# ----------------------------------------------------------------------
# Discriminator / DGILoss
# ----------------------------------------------------------------------
def test_discriminator_output_shape() -> None:
    d = Discriminator(5)
    out = d(torch.randn(11, 5), torch.randn(5))
    assert out.shape == (11,)


def test_dgi_loss_is_log2_at_initialisation() -> None:
    """An untrained bilinear discriminator scores real and corrupted alike."""
    Z = 0.1 * torch.randn(64, 16, generator=torch.Generator().manual_seed(0))
    Zc = corrupt(Z, torch.Generator().manual_seed(1))
    torch.manual_seed(0)
    loss = DGILoss(16)(Z, Zc)

    assert loss.ndim == 0 and loss.item() > 0
    assert loss.item() == pytest.approx(math.log(2), abs=0.05)


def _two_block_graph(n_per: int = 20, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Two dense communities; two features carry the community, three are noise."""
    g = torch.Generator().manual_seed(seed)
    N = 2 * n_per
    same = torch.zeros(N, N)
    same[:n_per, :n_per] = 1.0
    same[n_per:, n_per:] = 1.0
    p = torch.where(same > 0, 0.6, 0.05)
    A = (torch.rand(N, N, generator=g) < p).float()
    A = torch.triu(A, 1)
    A = A + A.T

    labels = torch.cat([torch.zeros(n_per), torch.ones(n_per)])
    signal = torch.stack([labels, 1.0 - labels], dim=1)
    X = torch.cat([signal + 0.3 * torch.randn(N, 2, generator=g),
                   torch.randn(N, 3, generator=g)], dim=1)
    return X, A


def test_dgi_loss_decreases_on_a_learnable_toy_graph() -> None:
    """Features correlated with structure: the objective must be learnable."""
    X, A = _two_block_graph()
    A_hat = normalise_adjacency(A)
    torch.manual_seed(0)
    enc = GCNEncoder(X.shape[1], hidden_dim=16, embed_dim=8, dropout=0.0)
    dgi = DGILoss(8)
    opt = torch.optim.Adam([*enc.parameters(), *dgi.parameters()], lr=0.01)
    gen = torch.Generator().manual_seed(1)

    losses = []
    for _ in range(300):
        opt.zero_grad()
        Z = enc(X, A_hat, normalise=False)
        Zc = enc(corrupt(X, gen), A_hat, normalise=False)
        loss = dgi(Z, Zc)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[0] == pytest.approx(math.log(2), abs=0.1)
    assert np.mean(losses[-20:]) < np.mean(losses[:20]) - 0.2
    assert losses[-1] < 0.5 * losses[0]


# ----------------------------------------------------------------------
# Volatility head
# ----------------------------------------------------------------------
def test_vol_head_shape() -> None:
    assert VolHead(6)(torch.randn(9, 6)).shape == (9,)


def test_masked_mse_ignores_nan_targets() -> None:
    y_hat = torch.tensor([0.5, 1.0, -1.0])
    y = torch.tensor([0.0, 1.0, 0.0])
    base = masked_mse(y_hat, y)
    assert base.item() == pytest.approx((0.25 + 0.0 + 1.0) / 3)

    with_nan = masked_mse(torch.cat([y_hat, torch.tensor([3.0])]),
                          torch.cat([y, torch.tensor([float("nan")])]))
    assert with_nan.item() == pytest.approx(base.item())

    # The all-NaN date contributes zero and still backpropagates.
    p = torch.randn(3, requires_grad=True)
    z = masked_mse(p, torch.full((3,), float("nan")))
    assert z.item() == 0.0
    z.backward()
    assert torch.equal(p.grad, torch.zeros(3))


# ----------------------------------------------------------------------
# Volatility target
# ----------------------------------------------------------------------
def _synthetic_panel(sigmas: dict[str, float], n_sessions: int = 300,
                     every: int = 50, seed: int = 0,
                     ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Long panel whose returns have an EXACT sample std per forward window.

    Each ticker's returns are rescaled inside every (t, t+1] window so that
    the window's sample std equals that ticker's sigma exactly, which makes
    the realised-vol check exact rather than statistical.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_sessions)
    dates = idx[every - 1::every]
    frames = []
    for ticker, sig in sigmas.items():
        r = rng.normal(0.0, sig, size=n_sessions)
        for i in range(len(dates) - 1):
            m = (idx > dates[i]) & (idx <= dates[i + 1])
            seg = r[m]
            r[m] = (seg - seg.mean()) / seg.std(ddof=1) * sig
        frames.append(pd.DataFrame({"date": idx, "ticker": ticker, "log_ret": r}))
    return pd.concat(frames, ignore_index=True), dates


def _save(panel: pd.DataFrame, path: Path) -> Path:
    panel.to_parquet(path, index=False)
    return path


def test_forward_realised_vol_recovers_known_sigma() -> None:
    sigmas = {"A": 0.01, "B": 0.02, "C": 0.03}
    panel, dates = _synthetic_panel(sigmas)
    ret = panel.pivot(index="date", columns="ticker", values="log_ret")

    raw = forward_realised_vol(ret, dates)
    for t, s in sigmas.items():
        assert np.allclose(raw[t].iloc[:-1], s * np.sqrt(TRADING_DAYS))
    assert raw.iloc[-1].isna().all()


def test_build_vol_target_final_row_nan_and_standardised(tmp_path: Path) -> None:
    sigmas = {"A": 0.01, "B": 0.02, "C": 0.03, "D": 0.04, "E": 0.05}
    panel, dates = _synthetic_panel(sigmas)
    y = build_vol_target(_save(panel, tmp_path / "panel.parquet"), dates,
                         list(sigmas))

    assert y.shape == (len(dates), 5)
    assert np.isnan(y[-1]).all()
    assert np.isfinite(y[:-1]).all()
    assert (np.abs(y[:-1]) <= 3.0).all()                 # the winsor clip
    assert np.allclose(np.median(y[:-1], axis=1), 0.0)   # row-wise centring
    assert (np.diff(y[:-1], axis=1) > 0).all()           # sigma order kept


def test_vol_target_has_no_lookahead(tmp_path: Path) -> None:
    """Row t depends on returns in (t, t+1] and on nothing else.

    The most important test in this file: corrupting every return strictly
    before t must leave y[t] untouched, and corrupting returns inside
    (t, t+1] must change y[t] and only y[t].
    """
    sigmas = {"A": 0.01, "B": 0.02, "C": 0.03, "D": 0.04}
    panel, dates = _synthetic_panel(sigmas)
    tickers = list(sigmas)
    k = 2                                                # a middle date
    base = build_vol_target(_save(panel, tmp_path / "base.parquet"), dates, tickers)

    # Everything strictly before t, every ticker, replaced by loud noise.
    before = panel.copy()
    m = before["date"] < dates[k]
    before.loc[m, "log_ret"] = np.random.default_rng(1).normal(0.0, 0.5, size=int(m.sum()))
    y_before = build_vol_target(_save(before, tmp_path / "before.parquet"), dates, tickers)
    assert np.allclose(y_before[k], base[k])
    assert not np.allclose(y_before[k - 1], base[k - 1])  # the corruption bit

    # One ticker's returns inside (t, t+1] scaled up tenfold.
    inside = panel.copy()
    m = ((inside["date"] > dates[k]) & (inside["date"] <= dates[k + 1])
         & (inside["ticker"] == "A"))
    inside.loc[m, "log_ret"] *= 10.0
    y_inside = build_vol_target(_save(inside, tmp_path / "inside.parquet"), dates, tickers)
    assert not np.allclose(y_inside[k], base[k])
    others = [i for i in range(len(dates) - 1) if i != k]
    assert np.allclose(y_inside[others], base[others])


# ----------------------------------------------------------------------
# Walk-forward schedule
# ----------------------------------------------------------------------
def test_walk_forward_schedule_and_nan_lead() -> None:
    """Leading dates are NaN, embedded dates are finite, refit count is right."""
    T, N, F_in = 12, 10, 3
    g = torch.Generator().manual_seed(0)
    X = torch.randn(T, N, F_in, generator=g).numpy()
    A = (torch.rand(T, N, N, generator=g) < 0.3).float()
    A = torch.triu(A, 1)
    A = A + A.transpose(1, 2)
    y = np.random.default_rng(0).normal(size=(T, N))
    y[-1] = np.nan
    dates = pd.date_range("2020-01-31", periods=T, freq="ME")
    cfg = {
        "model": {"hidden_dim": 8, "embed_dim": 4, "n_layers": 2,
                  "dropout": 0.0, "init": "glorot"},
        "training": {"epochs": 3, "patience": 2, "lr": 0.01,
                     "weight_decay": 0.0, "seed": 0, "vol_lambda": 0.5,
                     "train_window": 4, "embargo": 1, "refit_freq": 2},
    }

    Z, log = walk_forward(X, precompute_A_hat(A.numpy()), y, dates, cfg)

    lead = cfg["training"]["train_window"] + cfg["training"]["embargo"]
    assert Z.shape == (T, N, 4)
    assert np.isnan(Z[:lead]).all()
    assert np.isfinite(Z[lead:]).all()
    # t_end = 3, 5, 7, 9 embed [5,6], [7,8], [9,10], [11]; t_end = 11 has
    # nothing left to embed.
    assert len(log) == 4
    assert log["n_embed"].tolist() == [2, 2, 2, 1]
    # lambda > 0, so the skill number exists wherever the embed window holds a
    # date with a defined target. The last refit embeds only the final date,
    # whose forward window does not exist, so its Spearman is NaN by design.
    assert np.isfinite(log["vol_spearman"].iloc[:-1]).all()
    assert np.isnan(log["vol_spearman"].iloc[-1])
    assert (log["erank_mean"] > 1.0).all()
