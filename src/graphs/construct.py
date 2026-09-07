"""Phase 3 Part A: a temporal sequence of correlation kNN graphs.

Reads  : data/processed/panel.parquet   (long: date, ticker, log_ret)
         data/processed/features.npz    (ONLY 'dates' and 'tickers')
Writes : data/processed/graphs.npz      (A, A_binary, C, dates, tickers,
                                         config_json)

Pipeline
--------
For each rebalance date t:

    trailing corr_window sessions ending at t (inclusive)
      -> residualise (remove the top n_factors principal components)
      -> Ledoit-Wolf shrunk correlation
      -> kNN adjacency
      -> store

Design boundary
---------------
Edges are PAIRWISE quantities. This module never reads the node-feature
tensor X -- only the `dates` and `tickers` arrays from features.npz, and those
purely to guarantee the graph sequence shares the feature tensor's rebalance
clock and ticker ordering. Keeping node and edge information disjoint is what
makes the later "does the graph add anything over the features alone"
comparison meaningful.

Storage is DENSE float32 [T, N, N]. At N=105 the whole sequence is ~5.6 MB,
so scipy.sparse would add indexing complexity for no benefit.

Leakage rules
-------------
Highest priority. A leaked graph produces coherent-looking clusters that are
worthless, and there is no accuracy metric that will reveal it.

1. The correlation window for rebalance date t is the `corr_window` sessions
   ending at t INCLUSIVE. Data after t must never enter.
2. Windows are sliced by INTEGER POSITION in the session index, never by date
   arithmetic, so holidays cannot introduce an off-by-one.
3. Residualisation betas (here, principal components) are estimated inside the
   same trailing window, never globally over the full sample.
4. Ledoit-Wolf shrinkage intensity is estimated per window, never fitted once
   on all the data.
5. Nothing in this module shifts, centres or fills forward-looking.

Run from repo root:  python -m src.graphs.construct
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.covariance import LedoitWolf


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_returns(panel_path: str, tickers: list[str]) -> pd.DataFrame:
    """Wide date x ticker log-return frame in the exact given ticker order.

    Parameters
    ----------
    panel_path : str
        Path to data/processed/panel.parquet.
    tickers : list[str]
        Ticker order taken from features.npz. The returned columns match this
        exactly; a mismatch between the graph's node order and the feature
        tensor's would be a silent and catastrophic bug.

    Returns
    -------
    pd.DataFrame
        Dense log returns, no NaNs.

    Raises
    ------
    ValueError
        If a requested ticker is absent from the panel.
    """
    panel = pd.read_parquet(panel_path, columns=["date", "ticker", "log_ret"])
    panel["date"] = pd.to_datetime(panel["date"])
    ret = panel.pivot(index="date", columns="ticker", values="log_ret").sort_index()

    missing = [t for t in tickers if t not in ret.columns]
    if missing:
        raise ValueError(f"{len(missing)} tickers absent from the panel: {missing}")
    ret = ret[list(tickers)]

    # The first session has no predecessor, so its log difference is all-NaN.
    # That is mechanical, not missing data.
    if ret.iloc[0].isna().all():
        ret = ret.iloc[1:]

    # clean.py masks the return on a forward-filled day, because a return
    # spanning a stale price is mechanically zero rather than observed. Those
    # holes cannot reach LedoitWolf, so restore the zero the price path
    # implies -- but say so, loudly, rather than repairing in silence.
    holes = int(ret.isna().to_numpy().sum())
    if holes:
        where = ret.isna().stack()
        where = where[where]
        print(f"  {holes} masked (stale-price) returns restored to 0.0:")
        for (d, t) in list(where.index)[:10]:
            print(f"    {t} on {d.date()}")
        ret = ret.fillna(0.0)

    assert not ret.isna().to_numpy().any(), "NaNs survived in the return frame"
    assert list(ret.columns) == list(tickers), "ticker order does not match features.npz"
    print(f"  returns: {ret.shape[1]} tickers x {len(ret)} sessions "
          f"({ret.index[0].date()} to {ret.index[-1].date()})")
    return ret


# ----------------------------------------------------------------------
# Market-mode removal
# ----------------------------------------------------------------------
def residualise(R: np.ndarray, n_factors: int = 1) -> np.ndarray:
    """Project out the leading `n_factors` principal components of a window.

    Parameters
    ----------
    R : np.ndarray
        Return window of shape [T_win, N].
    n_factors : int, default 1
        Number of leading components to remove. 1 removes the market mode.

    Returns
    -------
    np.ndarray
        Residual matrix, same shape as `R`.

    Notes
    -----
    The market mode is the common component that otherwise links every asset
    to every other and yields a hairball in which no cluster is separable.
    Components are estimated on this window only -- never globally.
    """
    if n_factors <= 0:
        return np.array(R, dtype=float, copy=True)

    X = np.asarray(R, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)

    # Right singular vectors are the PCA loadings; project the window onto the
    # leading n_factors and subtract that projection.
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:n_factors].T                       # [N, n_factors]
    return Xc - (Xc @ V) @ V.T


# ----------------------------------------------------------------------
# Correlation
# ----------------------------------------------------------------------
def shrunk_correlation(R: np.ndarray,
                       method: str = "ledoit_wolf") -> tuple[np.ndarray, float]:
    """Shrunk correlation matrix from a return window.

    Parameters
    ----------
    R : np.ndarray
        Return window of shape [T_win, N].
    method : {'ledoit_wolf', 'sample'}, default 'ledoit_wolf'
        'sample' returns the plain sample correlation with intensity 0.0.

    Returns
    -------
    tuple[np.ndarray, float]
        `(C, shrinkage_intensity)` with C an N x N correlation matrix.

    Notes
    -----
    At q = N / T_win = 105 / 252 = 0.42 the sample correlation eigenvalues are
    badly biased. Marchenko-Pastur places the pure-noise upper edge at
    (1 + sqrt(q))^2 = 2.72, so eigenvalues below that carry no information yet
    are dispersed far from 1 by sampling noise alone. Shrinkage deflates those
    noise eigenvalues while leaving the large structural ones roughly intact,
    which is what makes the resulting neighbour ranking stable.

    Shrinkage intensity is estimated on this window only.
    """
    X = np.asarray(R, dtype=float)

    if method == "ledoit_wolf":
        lw = LedoitWolf().fit(X)
        cov = lw.covariance_
        intensity = float(lw.shrinkage_)
    elif method == "sample":
        cov = np.cov(X, rowvar=False)
        intensity = 0.0
    else:
        raise ValueError(f"unknown shrinkage method: {method!r}")

    d = np.sqrt(np.diag(cov))
    if not np.all(d > 0):
        raise ValueError("zero-variance asset in the window; correlation undefined")

    C = cov / np.outer(d, d)
    C = 0.5 * (C + C.T)                        # kill fp asymmetry
    np.fill_diagonal(C, 1.0)

    assert np.allclose(C, C.T, atol=1e-12), "correlation is not symmetric"
    assert np.allclose(np.diag(C), 1.0, atol=1e-12), "correlation diagonal is not 1"
    return C, intensity


def marchenko_pastur_edge(n: int, t: int) -> float:
    """Upper edge of the Marchenko-Pastur bulk, `(1 + sqrt(n / t))^2`.

    Eigenvalues below this are consistent with pure noise for a correlation
    matrix of `n` series estimated on `t` observations.
    """
    q = n / t
    return float((1.0 + np.sqrt(q)) ** 2)


# ----------------------------------------------------------------------
# kNN adjacency
# ----------------------------------------------------------------------
def knn_adjacency(S: np.ndarray, k: int, symmetrise: str = "union",
                  weighted: bool = True) -> np.ndarray:
    """k-nearest-neighbour adjacency from an N x N similarity matrix.

    Parameters
    ----------
    S : np.ndarray
        Similarity matrix, N x N. Larger means more similar.
    k : int
        Neighbours per node, before symmetrisation.
    symmetrise : {'union', 'intersection'}, default 'union'
        'union' takes the elementwise maximum of the directed indicator and
        its transpose; 'intersection' takes the elementwise minimum.
    weighted : bool, default True
        Multiply the symmetric mask by `S`; otherwise return the mask.

    Returns
    -------
    np.ndarray
        N x N adjacency, symmetric, with an exactly zero diagonal.

    Notes
    -----
    The directed kNN relation is NOT symmetric: a hub asset appears in many
    neighbourhoods while a peripheral one appears in few, so `j in kNN(i)`
    does not imply `i in kNN(j)`.

    Union is the default because intersection routinely disconnects peripheral
    nodes, and an isolated node receives no messages during propagation, which
    makes its GNN embedding meaningless. Union guarantees every node keeps at
    least its own k neighbours.
    """
    A = np.asarray(S, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"S must be square, got {A.shape}")
    n = A.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"k must satisfy 1 <= k < N={n}, got {k}")

    # -inf on the diagonal so no node can select itself.
    masked = A.copy()
    np.fill_diagonal(masked, -np.inf)

    # k largest per row -> directed indicator.
    idx = np.argpartition(-masked, kth=k - 1, axis=1)[:, :k]
    directed = np.zeros((n, n), dtype=float)
    directed[np.arange(n)[:, None], idx] = 1.0

    if symmetrise == "union":
        mask = np.maximum(directed, directed.T)
    elif symmetrise == "intersection":
        mask = np.minimum(directed, directed.T)
    else:
        raise ValueError(f"unknown symmetrise: {symmetrise!r}")
    np.fill_diagonal(mask, 0.0)

    out = mask * A if weighted else mask
    np.fill_diagonal(out, 0.0)
    return out


# ----------------------------------------------------------------------
# Sequence construction
# ----------------------------------------------------------------------
def _correlation_stack(ret: pd.DataFrame, dates: pd.DatetimeIndex,
                       cfg: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Windows -> residuals -> correlations, plus per-date meta.

    Split out from `build_graph_sequence` because it is the expensive half and
    is independent of k, so `sweep_k` can compute it once for every k.
    """
    window = int(cfg["corr_window"])
    n_factors = int(cfg["n_factors"])
    method = str(cfg["shrinkage"])

    values = ret.to_numpy(dtype=float)
    index = ret.index
    n = ret.shape[1]
    mp_edge = marchenko_pastur_edge(n, window)
    iu = np.triu_indices(n, k=1)

    C_stack = np.zeros((len(dates), n, n), dtype=np.float32)
    rows = []
    for i, d in enumerate(dates):
        # Integer position, never date arithmetic: holidays make date maths
        # off-by-one and the error would be invisible downstream.
        pos = index.get_loc(d)
        if not isinstance(pos, (int, np.integer)):
            raise ValueError(f"rebalance date {d} is not a unique trading session")
        start = pos + 1 - window
        if start < 0:
            raise ValueError(
                f"{d.date()} is at position {pos}; a {window}-session window "
                f"needs {window - 1 - pos} more sessions of history"
            )

        R = values[start:pos + 1]              # trailing window, t inclusive
        C, intensity = shrunk_correlation(residualise(R, n_factors), method)
        C_stack[i] = C.astype(np.float32)

        eig = np.linalg.eigvalsh(C)
        rows.append({
            "shrinkage": intensity,
            "mean_corr": float(C[iu].mean()),
            "top_eigval_frac": float(eig[-1] / eig.sum()),
            "n_eigs_above_mp": int((eig > mp_edge).sum()),
        })

    meta = pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="date"))
    return C_stack, meta


def _adjacency_stack(C_stack: np.ndarray, cfg: dict, k: int,
                     weighted: bool | None = None) -> np.ndarray:
    """Apply the kNN step to a stack of correlation matrices."""
    similarity = str(cfg.get("similarity", "signed"))
    symmetrise = str(cfg.get("symmetrise", "union"))
    if weighted is None:
        weighted = bool(cfg.get("weighted", True))

    out = np.zeros_like(C_stack, dtype=np.float32)
    for i in range(C_stack.shape[0]):
        C = C_stack[i].astype(float)
        if similarity == "signed":
            S = C
        elif similarity == "absolute":
            S = np.abs(C)
        else:
            raise ValueError(f"unknown similarity: {similarity!r}")
        out[i] = knn_adjacency(S, k, symmetrise, weighted).astype(np.float32)
    return out


def build_graph_sequence(ret: pd.DataFrame, dates: pd.DatetimeIndex, cfg: dict,
                         k: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build the full graph sequence on the rebalance clock.

    Parameters
    ----------
    ret : pd.DataFrame
        Wide date x ticker log returns.
    dates : pd.DatetimeIndex
        Rebalance dates; each must be a session in `ret.index`.
    cfg : dict
        The `graph` config block.
    k : int
        Neighbours per node.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, pd.DataFrame]
        `(A, C_stack, meta)` with A the weighted adjacency [T, N, N] float32,
        C_stack the correlation matrices [T, N, N] float32, and meta indexed
        by date with columns shrinkage, mean_corr, top_eigval_frac and
        n_eigs_above_mp.

    Notes
    -----
    C_stack is returned and stored because Part B needs the correlations for
    the Gaussian-benchmark comparison, and recomputing them would be wasteful.
    """
    C_stack, meta = _correlation_stack(ret, dates, cfg)
    return _adjacency_stack(C_stack, cfg, k), C_stack, meta


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------
def edge_persistence(A1: np.ndarray, A2: np.ndarray) -> float:
    """Jaccard overlap of two adjacency matrices' edge sets.

    Computed on the binarised upper triangle: `|E1 & E2| / |E1 | E2|`.
    Returns 0.0 when the union is empty.

    Notes
    -----
    Interpretation. Near 0 means the graph is noise and any clustering built
    on it will be unstable. Near 1 means the graph is not dynamic at all, and
    the premise of a *temporal* study collapses. Roughly 0.6 to 0.85 indicates
    structure that genuinely evolves.

    Honest caveat: consecutive 252-session windows one month apart share about
    92% of their observations, so a high score is partly mechanical and should
    not be read as evidence of real stability on its own.
    """
    iu = np.triu_indices(A1.shape[0], k=1)
    e1 = A1[iu] != 0
    e2 = A2[iu] != 0
    union = int((e1 | e2).sum())
    if union == 0:
        return 0.0
    return float((e1 & e2).sum() / union)


def _gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector; 0 is uniform."""
    v = np.sort(np.asarray(x, dtype=float))
    n = v.size
    total = v.sum()
    if n == 0 or total <= 0:
        return 0.0
    # 2*sum(i*v_i)/(n*sum(v)) - (n+1)/n, with i one-based.
    i = np.arange(1, n + 1)
    return float((2.0 * (i * v).sum()) / (n * total) - (n + 1) / n)


def graph_diagnostics(A: np.ndarray) -> pd.DataFrame:
    """Per-date structural diagnostics for a [T, N, N] adjacency stack.

    Returns
    -------
    pd.DataFrame
        One row per date with n_edges, density, mean_degree, min_degree,
        max_degree, degree_gini, n_components and is_connected.
    """
    rows = []
    n = A.shape[1]
    max_edges = n * (n - 1) / 2
    for i in range(A.shape[0]):
        B = (A[i] != 0)
        deg = B.sum(axis=1)
        n_comp, _ = connected_components(csr_matrix(B), directed=False)
        rows.append({
            "n_edges": int(B.sum() // 2),
            "density": float(B.sum() / 2 / max_edges),
            "mean_degree": float(deg.mean()),
            "min_degree": int(deg.min()),
            "max_degree": int(deg.max()),
            "degree_gini": _gini(deg),
            "n_components": int(n_comp),
            "is_connected": bool(n_comp == 1),
        })
    return pd.DataFrame(rows)


def sweep_k(ret: pd.DataFrame, dates: pd.DatetimeIndex, cfg: dict,
            k_values: list[int]) -> pd.DataFrame:
    """Summarise the graph sequence at each candidate k.

    The correlation stack does not depend on k, so it is built once and reused
    across every candidate. Results are identical to rebuilding per k.

    Returns
    -------
    pd.DataFrame
        One row per k: k, mean_degree, density, frac_dates_connected,
        mean_persistence, std_persistence.
    """
    C_stack, _ = _correlation_stack(ret, dates, cfg)

    rows = []
    for k in k_values:
        A = _adjacency_stack(C_stack, cfg, k)
        diag = graph_diagnostics(A)
        pers = np.array([edge_persistence(A[i], A[i + 1])
                         for i in range(len(A) - 1)])
        rows.append({
            "k": k,
            "mean_degree": diag["mean_degree"].mean(),
            "density": diag["density"].mean(),
            "frac_dates_connected": diag["is_connected"].mean(),
            "mean_persistence": float(pers.mean()) if pers.size else np.nan,
            "std_persistence": float(pers.std()) if pers.size else np.nan,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def main() -> None:
    cfg_all = yaml.safe_load(open("config/base.yaml"))
    gcfg = cfg_all["graph"]
    proc = Path(cfg_all["data"]["processed_dir"])

    print("[1] aligning to the feature tensor's clock and ticker order")
    feat = np.load(proc / "features.npz", allow_pickle=True)
    tickers = [str(t) for t in feat["tickers"]]
    dates = pd.DatetimeIndex(pd.to_datetime([str(s) for s in feat["dates"]]))
    print(f"  {len(dates)} rebalance dates, {len(tickers)} tickers "
          f"({dates[0].date()} to {dates[-1].date()})")

    print("[2] loading returns")
    ret = load_returns(proc / "panel.parquet", tickers)

    print(f"[3] k sweep over {gcfg['k_sweep']}")
    sweep = sweep_k(ret, dates, gcfg, list(gcfg["k_sweep"]))
    print(sweep.round(4).to_string(index=False))

    k = int(gcfg["k"])
    print(f"\n[4] building the final sequence at k={k}")
    A, C_stack, meta = build_graph_sequence(ret, dates, gcfg, k)
    A_bin = _adjacency_stack(C_stack, gcfg, k, weighted=False).astype(np.uint8)

    diag = graph_diagnostics(A).set_index(pd.DatetimeIndex(dates, name="date"))
    pers = np.array([edge_persistence(A[i], A[i + 1]) for i in range(len(A) - 1)])

    out = proc / "graphs.npz"
    np.savez_compressed(
        out,
        A=A.astype(np.float32),
        A_binary=A_bin,
        C=C_stack.astype(np.float32),
        dates=np.array([d.strftime("%Y-%m-%d") for d in dates], dtype=object),
        tickers=np.array(tickers, dtype=object),
        config_json=np.array(json.dumps(gcfg, sort_keys=True, default=str)),
    )
    print(f"  wrote {out}  A.shape={A.shape}")

    print(f"\nchosen k                     : {k}")
    print(f"mean degree                  : {diag['mean_degree'].mean():.2f} "
          f"(range {diag['min_degree'].min()} to {diag['max_degree'].max()})")
    print(f"fraction of dates connected  : {diag['is_connected'].mean():.1%}")
    print(f"mean edge persistence        : {pers.mean():.4f} "
          f"(sd {pers.std():.4f})")
    print(f"mean shrinkage intensity     : {meta['shrinkage'].mean():.4f} "
          f"(min {meta['shrinkage'].min():.4f})")
    print(f"mean eigenvalues above MP    : {meta['n_eigs_above_mp'].mean():.2f} "
          f"(edge {marchenko_pastur_edge(len(tickers), int(gcfg['corr_window'])):.3f})")


if __name__ == "__main__":
    main()
