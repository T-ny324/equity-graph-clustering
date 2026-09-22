"""Phase 4b training: walk-forward Deep Graph Infomax -> embeddings.npz.

Reads  : data/processed/features.npz    (X [T, N, F], tickers, dates)
         data/processed/graphs.npz      (A [T, N, N]; graphs_mi.npz when
                                         training.graph_source is
                                         mutual_information)
         data/processed/panel.parquet   (date, ticker, log_ret -- used for
                                         the volatility TARGET only)
Writes : data/processed/embeddings.npz  (Z [T, N, embed_dim], dates, tickers,
                                         refit_log, config_json)

Protocol: walk-forward
----------------------
Reference: docs/phase4_maths_reference.md, section 5.1; the diagnostics
logged per refit are section 3.3.

For each refit: train on the `train_window` rebalance dates ending at t, skip
`embargo` dates, then embed the next `refit_freq` dates with frozen weights.
Advance by `refit_freq` and repeat until the date range is exhausted.

    [ ........ train_window ........ t ][ embargo ][ .... refit_freq .... ]
                                        held out    embedded with the encoder
                                                    fitted at t, then frozen

Why walk-forward
----------------
A single fit pooled over all 128 dates leaks -- the encoder would have seen
2026 while embedding 2017 -- and unlike a supervised task nothing warns you,
because the loss looks fine and the clusters look coherent.

The embargo exists because features use trailing windows up to 252 days and
graphs use 252-day correlation windows, so two adjacent rebalance dates share
about 92% of their underlying data; without a gap the train and embed windows
overlap through the features even when the dates do not.

Dates before the first embed window have no trained encoder. They are left
NaN in Z -- never filled with an untrained embedding -- and their count is
reported.

Leakage rule for the volatility target
--------------------------------------
`build_vol_target` is realised AFTER each date t. It is built once, up front,
stored separately from X, and enters the computation at exactly one point:
the loss term inside `train_window`. It is never an encoder input, never a
feature, never part of the graph.

Run from repo root:  python -m src.models.train
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

from src.features.node_features import ANNUALISE, standardise
from src.models.encoder import GCNEncoder
from src.models.layers import normalise_adjacency
from src.models.objectives import DGILoss, VolHead, corrupt, masked_mse

# Fraction of each training window's dates held out for early stopping.
VAL_FRAC = 0.2


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------
def effective_rank(Z_t: np.ndarray) -> float:
    """exp of the spectral entropy of the singular values of one date's Z.

    Parameters
    ----------
    Z_t : np.ndarray
        Embeddings for one date, [N, d].

    Returns
    -------
    float
        A continuous count of the directions in use: d for a flat spectrum,
        1 for a rank-one matrix (maths reference section 3.3). A drift
        toward 1 under training is oversmoothing and is a finding, not a
        tolerance problem.
    """
    s = np.linalg.svd(np.asarray(Z_t, dtype=float), compute_uv=False)
    if not np.isfinite(s).all() or s.sum() <= 0:
        return float("nan")
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def embedding_variance(Z_t: np.ndarray) -> float:
    """(1/Nd) sum_{i,f} (Z_if - Zbar_f)^2: per-dimension variance, averaged.

    Parameters
    ----------
    Z_t : np.ndarray
        Embeddings for one date, [N, d].

    Returns
    -------
    float
        Each embedding dimension centred on its own mean across nodes
        (maths reference section 3.3). Decay toward zero is oversmoothing.
    """
    return float(np.asarray(Z_t, dtype=float).var(axis=0, ddof=0).mean())


# ----------------------------------------------------------------------
# The volatility target
# ----------------------------------------------------------------------
def forward_realised_vol(ret: pd.DataFrame,
                         dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Annualised realised volatility over (t, t+1], one row per rebalance date.

    Parameters
    ----------
    ret : pd.DataFrame
        Daily log returns, date x ticker.
    dates : pd.DatetimeIndex
        Rebalance dates, ascending, every one a member of `ret.index`.

    Returns
    -------
    pd.DataFrame
        `sqrt(252) * std` of the returns strictly after date t up to and
        including date t+1, indexed by `dates`, columns as `ret`. The final
        row is NaN: it has no forward window.

    Notes
    -----
    This is the raw, pre-standardisation quantity. `build_vol_target` applies
    the cross-sectional standardisation on top.
    """
    ret = ret.sort_index()
    dates = pd.DatetimeIndex(dates)
    out = pd.DataFrame(np.nan, index=dates, columns=ret.columns)
    for i in range(len(dates) - 1):
        # Left-open, right-closed. Returns ON date t belong to the trailing
        # features, not to the target; the window is (t, t+1].
        window = ret.loc[(ret.index > dates[i]) & (ret.index <= dates[i + 1])]
        out.iloc[i] = window.std(ddof=1).to_numpy() * ANNUALISE
    return out


def build_vol_target(panel_path: str | Path, dates: pd.DatetimeIndex,
                     tickers: list[str], winsor: float = 3.0) -> np.ndarray:
    """The [T, N] standardised forward-volatility target; final row NaN.

    Parameters
    ----------
    panel_path : str or Path
        data/processed/panel.parquet, long with date, ticker, log_ret.
    dates : pd.DatetimeIndex
        Rebalance dates in the artefacts' order.
    tickers : list[str]
        Canonical ticker order, taken from features.npz.
    winsor : float, default 3.0
        Clip bound passed to `standardise`; matches `features.winsor`.

    Returns
    -------
    np.ndarray
        [T, N] float. Each row is `forward_realised_vol` standardised across
        tickers at that date with the same robust median / MAD / clip
        procedure as the features -- imported, not reimplemented, so the two
        cannot drift apart. The last row is NaN and must be masked out of any
        loss.

    Notes
    -----
    LEAKAGE RULE. Everything this returns is realised after its date. It is a
    target for the loss term and nothing else.
    """
    panel = pd.read_parquet(panel_path, columns=["date", "ticker", "log_ret"])
    panel["date"] = pd.to_datetime(panel["date"])
    ret = panel.pivot(index="date", columns="ticker", values="log_ret").sort_index()

    missing = [t for t in tickers if t not in ret.columns]
    assert not missing, f"{len(missing)} ticker(s) absent from the panel: {missing}"
    ret = ret[list(tickers)]

    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    off = dates.difference(ret.index)
    assert not len(off), f"{len(off)} rebalance dates are not trading sessions: {list(off[:5])}"

    raw = forward_realised_vol(ret, dates)
    # axis=1 only: across tickers at one date. Standardising across time would
    # leak every later month into every earlier one.
    y = standardise(raw, winsor=winsor).to_numpy(dtype=float)
    assert np.isnan(y[-1]).all(), "the final date has no forward window and must be NaN"
    return y


# ----------------------------------------------------------------------
# Training pieces
# ----------------------------------------------------------------------
def precompute_A_hat(A: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Normalise every adjacency once, up front.

    Parameters
    ----------
    A : np.ndarray or torch.Tensor
        Adjacency stack, [T, N, N].

    Returns
    -------
    torch.Tensor
        `D^-1/2 (A + I) D^-1/2` per date, float32, [T, N, N].

    Notes
    -----
    A_hat is identical across epochs for a given date, so recomputing it
    inside the loop would repeat an O(N^2) operation `epochs` times per date.
    Pass `normalise=False` to the encoder from here on.
    """
    A_t = torch.as_tensor(np.asarray(A), dtype=torch.float32)
    return normalise_adjacency(A_t)


def train_window(X_win: torch.Tensor, A_hat_win: torch.Tensor,
                 y_win: torch.Tensor, cfg: dict,
                 seed: int) -> tuple[GCNEncoder, dict]:
    """Fit one refit on a window of dates with early stopping.

    Parameters
    ----------
    X_win : torch.Tensor
        Features for the window's dates, [n, N, F].
    A_hat_win : torch.Tensor
        Pre-normalised adjacencies for the same dates, [n, N, N].
    y_win : torch.Tensor
        Forward-volatility targets, [n, N], NaN where undefined. Used in the
        loss term only.
    cfg : dict
        The full config; reads `cfg["model"]` and `cfg["training"]`.
    seed : int
        Seeds the weight draw and the corruption generator.

    Returns
    -------
    tuple[GCNEncoder, dict]
        The encoder with its best-validation weights restored, and a history
        dict: per-epoch `train_loss` and `val_loss`, `epochs_run`,
        `best_epoch`, `best_val_loss`, `early_stopped`, and the trained
        `vol_head` (needed for the out-of-sample volatility diagnostic).

    Notes
    -----
    Each epoch iterates over the training dates, accumulates the mean loss
    across them, then steps the optimiser once. The last `VAL_FRAC` of the
    window's dates are held out for early stopping on validation loss with
    `training.patience`; the best weights are restored before returning.
    The validation corruption is drawn from a fresh generator each epoch so
    successive validation losses are comparable.
    """
    mcfg, tcfg = cfg["model"], cfg["training"]
    lam = float(tcfg.get("vol_lambda", 0.0))
    epochs, patience = int(tcfg["epochs"]), int(tcfg["patience"])

    n_dates, _, in_dim = X_win.shape
    n_val = max(1, int(VAL_FRAC * n_dates))
    n_train = n_dates - n_val
    assert n_train >= 1, f"window of {n_dates} dates leaves no training dates"
    train_idx, val_idx = range(n_train), range(n_train, n_dates)

    torch.manual_seed(seed)                      # the weight draw
    encoder = GCNEncoder(in_dim, hidden_dim=int(mcfg["hidden_dim"]),
                         embed_dim=int(mcfg["embed_dim"]),
                         n_layers=int(mcfg["n_layers"]),
                         dropout=float(mcfg["dropout"]))
    dgi = DGILoss(int(mcfg["embed_dim"]))
    vol_head = VolHead(int(mcfg["embed_dim"]))
    opt = torch.optim.Adam(
        [*encoder.parameters(), *dgi.parameters(), *vol_head.parameters()],
        lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    gen = torch.Generator().manual_seed(seed)    # the corruption draw

    def window_loss(idx: range, generator: torch.Generator) -> torch.Tensor:
        total = torch.zeros(())
        for i in idx:
            X_t, A_t = X_win[i], A_hat_win[i]
            Z = encoder(X_t, A_t, normalise=False)
            Z_c = encoder(corrupt(X_t, generator), A_t, normalise=False)
            loss = dgi(Z, Z_c)
            if lam > 0.0:
                # LEAKAGE RULE: y_win[i] is realised AFTER date i. This is the
                # only place it is read, and it is read as a target for the
                # loss -- it never enters the encoder, X, or the graph.
                loss = loss + lam * masked_mse(vol_head(Z), y_win[i])
            total = total + loss
        return total / len(idx)

    hist_tr: list[float] = []
    hist_va: list[float] = []
    best_val, best_epoch, best_state, since_best = math.inf, 0, None, 0
    early_stopped = False
    for epoch in range(1, epochs + 1):
        encoder.train()
        opt.zero_grad()
        loss = window_loss(train_idx, gen)
        loss.backward()
        opt.step()

        encoder.eval()
        with torch.no_grad():
            val = window_loss(val_idx, torch.Generator().manual_seed(seed + 1))
        hist_tr.append(loss.item())
        hist_va.append(val.item())

        if val.item() < best_val:
            best_val, best_epoch, since_best = val.item(), epoch, 0
            best_state = (copy.deepcopy(encoder.state_dict()),
                          copy.deepcopy(dgi.state_dict()),
                          copy.deepcopy(vol_head.state_dict()))
        else:
            since_best += 1
            if since_best >= patience:
                early_stopped = True
                break

    encoder.load_state_dict(best_state[0])
    dgi.load_state_dict(best_state[1])
    vol_head.load_state_dict(best_state[2])
    encoder.eval()
    return encoder, {
        "train_loss": hist_tr, "val_loss": hist_va,
        "epochs_run": len(hist_tr), "best_epoch": best_epoch,
        "best_val_loss": best_val, "early_stopped": early_stopped,
        "n_train_dates": n_train, "n_val_dates": n_val,
        "vol_head": vol_head,
    }


def embed(encoder: GCNEncoder, X_slice: torch.Tensor,
          A_hat_slice: torch.Tensor) -> np.ndarray:
    """Deterministic embeddings for a run of dates from a frozen encoder.

    Parameters
    ----------
    encoder : GCNEncoder
        A trained encoder.
    X_slice : torch.Tensor
        Features, [n_dates, N, F].
    A_hat_slice : torch.Tensor
        Pre-normalised adjacencies, [n_dates, N, N].

    Returns
    -------
    np.ndarray
        [n_dates, N, embed_dim] float32. `eval()` inside `no_grad()`, so
        dropout is off and the result is a deterministic function of the
        weights.
    """
    encoder.eval()
    with torch.no_grad():
        Z = encoder(X_slice, A_hat_slice, normalise=False)
    return Z.cpu().numpy().astype(np.float32)


def walk_forward(X: np.ndarray, A_hat: torch.Tensor, y: np.ndarray,
                 dates: pd.DatetimeIndex,
                 cfg: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Run every refit and assemble Z over the full date range.

    Parameters
    ----------
    X : np.ndarray
        Feature tensor, [T, N, F].
    A_hat : torch.Tensor
        Output of `precompute_A_hat`, [T, N, N].
    y : np.ndarray
        Output of `build_vol_target`, [T, N].
    dates : pd.DatetimeIndex
        Length-T rebalance dates.
    cfg : dict
        The full config.

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame]
        `Z` of shape [T, N, embed_dim], float32, NaN on every date before the
        first embed window; and one diagnostics row per refit. The per-epoch
        loss curves ride along as JSON strings in the `train_curve` and
        `val_curve` columns so the artefact carries them.
    """
    mcfg, tcfg = cfg["model"], cfg["training"]
    W, E, R = int(tcfg["train_window"]), int(tcfg["embargo"]), int(tcfg["refit_freq"])
    lam, seed = float(tcfg.get("vol_lambda", 0.0)), int(tcfg["seed"])

    # np.array (a copy) rather than as_tensor over a possibly read-only view.
    X_t = torch.as_tensor(np.array(X, dtype=np.float32))
    y_t = torch.as_tensor(np.array(y, dtype=np.float32))
    T, N, _ = X_t.shape
    Z = np.full((T, N, int(mcfg["embed_dim"])), np.nan, dtype=np.float32)

    rows = []
    t_end, refit = W - 1, 0
    while t_end + E + 1 < T:                     # at least one date to embed
        tr0, tr1 = t_end - W + 1, t_end + 1      # train: [tr0, t_end]
        em0 = t_end + E + 1                      # embed: [em0, em1)
        em1 = min(em0 + R, T)
        print(f"  refit {refit}: train {dates[tr0].date()}..{dates[t_end].date()} "
              f"({W} dates) | embargo {E} | embed {dates[em0].date()}.."
              f"{dates[em1 - 1].date()} ({em1 - em0} dates)")

        encoder, hist = train_window(X_t[tr0:tr1], A_hat[tr0:tr1],
                                     y_t[tr0:tr1], cfg, seed + refit)
        Z_em = embed(encoder, X_t[em0:em1], A_hat[em0:em1])
        Z[em0:em1] = Z_em

        eranks = [effective_rank(Z_em[i]) for i in range(len(Z_em))]
        variances = [embedding_variance(Z_em[i]) for i in range(len(Z_em))]

        # Out-of-sample volatility skill: the one hard supervised number in
        # the project. Only meaningful when the head was actually trained.
        vol_rho = float("nan")
        if lam > 0.0:
            with torch.no_grad():
                y_hat = hist["vol_head"](torch.as_tensor(Z_em)).numpy()
            rhos = []
            for i in range(len(Z_em)):
                m = np.isfinite(y[em0 + i])
                if m.sum() >= 3:
                    rhos.append(float(spearmanr(y_hat[i][m], y[em0 + i][m])[0]))
            vol_rho = float(np.mean(rhos)) if rhos else float("nan")

        rows.append({
            "refit": refit,
            "train_start": dates[tr0].date(), "train_end": dates[t_end].date(),
            "embed_start": dates[em0].date(), "embed_end": dates[em1 - 1].date(),
            "n_embed": em1 - em0,
            "epochs": hist["epochs_run"], "best_epoch": hist["best_epoch"],
            "early_stop": hist["early_stopped"],
            "train_first": hist["train_loss"][0],
            "train_final": hist["train_loss"][hist["best_epoch"] - 1],
            "val_best": hist["best_val_loss"],
            "erank_mean": float(np.mean(eranks)), "erank_min": float(np.min(eranks)),
            "var_mean": float(np.mean(variances)),
            "vol_spearman": vol_rho,
            "train_curve": json.dumps([round(v, 6) for v in hist["train_loss"]]),
            "val_curve": json.dumps([round(v, 6) for v in hist["val_loss"]]),
        })
        r = rows[-1]
        print(f"    epochs {r['epochs']:>3} (best {r['best_epoch']:>3}"
              f"{', early stop' if r['early_stop'] else ''}) | "
              f"loss {r['train_first']:.4f} -> {r['train_final']:.4f} "
              f"(val {r['val_best']:.4f}) | erank {r['erank_mean']:.2f} "
              f"(min {r['erank_min']:.2f}) | var {r['var_mean']:.4f}"
              + ((f" | OOS vol spearman {vol_rho:+.3f}" if np.isfinite(vol_rho)
                  else " | OOS vol spearman n/a (no target in window)")
                 if lam > 0.0 else ""))

        t_end += R
        refit += 1

    return Z, pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def main() -> None:
    cfg = yaml.safe_load(open("config/base.yaml"))
    mcfg, tcfg = cfg["model"], cfg["training"]
    if str(mcfg.get("init", "glorot")) != "glorot":
        raise NotImplementedError(
            f"model.init={mcfg['init']!r}: GraphConv initialises with Glorot "
            f"only; the config records the choice, it does not switch it")
    if str(tcfg.get("objective", "dgi")) != "dgi":
        raise NotImplementedError(f"training.objective={tcfg['objective']!r}: only dgi is implemented")

    seed = int(tcfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    lam = float(tcfg.get("vol_lambda", 0.0))

    proc = Path(cfg["data"]["processed_dir"])
    source = str(tcfg.get("graph_source", "correlation"))
    graph_file = {"correlation": "graphs.npz",
                  "mutual_information": "graphs_mi.npz"}[source]

    print("[1] loading artefacts")
    feat = np.load(proc / "features.npz", allow_pickle=True)
    g = np.load(proc / graph_file, allow_pickle=True)
    tickers = [str(t) for t in feat["tickers"]]
    date_strs = [str(s) for s in feat["dates"]]
    # Node and date order must be identical or every refit silently pairs
    # one asset's features with another's neighbourhood.
    assert tickers == [str(t) for t in g["tickers"]], \
        f"ticker order differs between features.npz and {graph_file}"
    assert date_strs == [str(s) for s in g["dates"]], \
        f"date list differs between features.npz and {graph_file}"
    dates = pd.DatetimeIndex(pd.to_datetime(date_strs))
    X, A = feat["X"], g["A"]
    T, N, F_in = X.shape
    print(f"  X {X.shape}  A {A.shape}  ({dates[0].date()} to {dates[-1].date()}) "
          f"graph_source={source}")

    print("[2] forward volatility target  (loss term only -- never an input)")
    y = build_vol_target(proc / "panel.parquet", dates, tickers,
                         winsor=float(cfg.get("features", {}).get("winsor", 3.0)))
    finite = np.isfinite(y[:-1]).mean()
    print(f"  y {y.shape}: {finite:.1%} of entries defined before the final "
          f"(NaN) row; vol_lambda={lam}"
          + ("" if lam > 0 else "  -> head is wired but inactive"))

    print("[3] normalising every adjacency once")
    A_hat = precompute_A_hat(A)

    print(f"[4] walk-forward: train_window={tcfg['train_window']} "
          f"embargo={tcfg['embargo']} refit_freq={tcfg['refit_freq']} "
          f"epochs<={tcfg['epochs']} patience={tcfg['patience']} seed={seed}")
    Z, log = walk_forward(X, A_hat, y, dates, cfg)

    nan_dates = int(np.isnan(Z).all(axis=(1, 2)).sum())
    assert np.isfinite(Z[nan_dates:]).all(), "NaN inside an embedded date"

    out = proc / "embeddings.npz"
    np.savez_compressed(
        out,
        Z=Z.astype(np.float32),
        dates=np.array(date_strs, dtype=object),
        tickers=np.array(tickers, dtype=object),
        refit_log=np.array(log.to_csv(index=False)),
        config_json=np.array(json.dumps({"model": mcfg, "training": tcfg},
                                        sort_keys=True, default=str)),
    )
    print(f"\nwrote {out}  Z.shape={Z.shape}")

    show = log.drop(columns=["train_curve", "val_curve"])
    print("\nper-refit diagnostics:")
    print(show.round(4).to_string(index=False))

    print(f"\nrefits                        : {len(log)}")
    print(f"dates with no encoder (NaN)   : {nan_dates} of {T} "
          f"({dates[0].date()} to {dates[nan_dates - 1].date()})")
    print(f"mean effective rank of Z      : {log['erank_mean'].mean():.3f} "
          f"(per-refit min {log['erank_min'].min():.3f}) of {mcfg['embed_dim']}")
    print(f"mean embedding variance       : {log['var_mean'].mean():.4f}")
    for which in (0, len(log) - 1):
        curve = json.loads(log.loc[which, "train_curve"])
        vcurve = json.loads(log.loc[which, "val_curve"])
        marks = sorted({0, 4, 9, 19, 49, 99, len(curve) - 1} & set(range(len(curve))))
        traj = "  ".join(f"e{m + 1}:{curve[m]:.3f}/{vcurve[m]:.3f}" for m in marks)
        print(f"loss trajectory, refit {which:<2} (train/val): {traj}")
    if lam > 0.0:
        print("\n==================================================================")
        print(f"OUT-OF-SAMPLE VOLATILITY SKILL (Spearman, predicted vs realised)")
        print(f"  per refit : {' '.join(f'{v:+.3f}' for v in log['vol_spearman'])}")
        print(f"  mean      : {log['vol_spearman'].mean():+.4f}")
        print("==================================================================")


if __name__ == "__main__":
    main()
