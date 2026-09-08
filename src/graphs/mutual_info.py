"""Phase 3 Part B: mutual-information kNN graphs, as a control against Part A.

Reads  : data/processed/panel.parquet   (long: date, ticker, log_ret)
         data/processed/graphs.npz      (C, A, dates, tickers from Part A)
Writes : data/processed/graphs_mi.npz   (A, A_binary, M, Delta, dates,
                                         tickers, config_json)

Experimental design
-------------------
This is a CONTROLLED comparison. The graph topology is held fixed -- same kNN,
same k, same union symmetrisation -- and only the similarity measure changes:
Pearson correlation in Part A, mutual information here. Any difference in the
results is then attributable to the similarity measure alone.

To guarantee that, `residualise` and `knn_adjacency` are IMPORTED from
src.graphs.construct rather than reimplemented. Reimplementing either would
silently break the control, which is the one thing this module exists to
protect.

The shrunk correlation matrices from Part A are reused, never recomputed: they
are needed both for the Gaussian benchmark and for the nonlinear-excess
measurement.

Three facts that shape everything below
---------------------------------------
1. **The Gaussian identity.** For a bivariate Gaussian pair with correlation
   rho, MI = -0.5 * ln(1 - rho^2) exactly, in nats. This is simultaneously the
   estimator's unit test and the definition of what MI adds beyond
   correlation. If an estimator cannot reproduce this curve, no downstream
   number in this module is interpretable.

2. **MI is unsigned.** It depends on rho^2, so rho = +0.9 and rho = -0.9 give
   an identical 0.830 nats. An unsigned MI-kNN graph therefore connects an
   asset to its strongest HEDGES as readily as to its closest peers, which is
   a substantive defect for equity clustering. Both variants are built so the
   effect can be measured rather than assumed.

3. **MI noise is one-sided.** MI is non-negative, so sampling error has
   nowhere to cancel and independent series still yield MI_hat > 0.
   Correlation needs no null model because its errors are signed and average
   out. MI does. This is the biggest practical difference between the two
   branches, and it is why `mi_null_floor` exists.

Deliberately excluded
---------------------
Conditional mutual information, partial mutual information and the
path-consistency algorithm are NOT implemented. Equation (17) needs a
three-dimensional joint histogram: b^3 cells against T = 252 observations is
roughly two samples per cell at b = 5, so the estimate would be noise wearing
the costume of a result. The exclusion is a judgement about estimability, not
an oversight.

Leakage rules
-------------
1. The window for rebalance date t is the corr_window sessions ending at t
   INCLUSIVE, sliced by integer position and never by date arithmetic.
2. Residualisation happens within each window, never globally.
3. Bin edges are computed from the window only.
4. No negative shift, no bfill, no center=True anywhere in this module.

Run from repo root:  python -m src.graphs.mutual_info
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_regression

from src.graphs.construct import (
    edge_persistence,
    knn_adjacency,
    load_returns,
    residualise,
)

# Guards |rho| away from 1 so the Gaussian identity cannot overflow.
RHO_CLIP = 1.0 - 1e-12

# sklearn's KSG breaks ties with a tiny noise draw; fix it for reproducibility.
KSG_RANDOM_STATE = 0


# ----------------------------------------------------------------------
# The Gaussian benchmark
# ----------------------------------------------------------------------
def gaussian_mi(rho: np.ndarray | float) -> np.ndarray | float:
    """Exact mutual information of a bivariate Gaussian pair, in nats.

    `MI = -0.5 * ln(1 - rho^2)`, elementwise, with |rho| clipped just inside 1
    so the endpoints cannot overflow.

    Parameters
    ----------
    rho : np.ndarray or float
        Correlation(s). Scalars and full matrices both work.

    Returns
    -------
    np.ndarray or float
        Mutual information in nats, same shape as the input.
    """
    r = np.clip(np.asarray(rho, dtype=float), -RHO_CLIP, RHO_CLIP)
    out = -0.5 * np.log1p(-(r ** 2))
    return float(out) if np.isscalar(rho) or np.ndim(rho) == 0 else out


# ----------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------
def _bin_index(v: np.ndarray, bins: int, scheme: str) -> tuple[np.ndarray, int]:
    """Discretise one series; returns (bin index per sample, n_bins used)."""
    if scheme == "quantile":
        edges = np.unique(np.quantile(v, np.linspace(0.0, 1.0, bins + 1)))
    elif scheme == "uniform":
        edges = np.linspace(float(v.min()), float(v.max()), bins + 1)
    else:
        raise ValueError(f"unknown bin scheme: {scheme!r}")

    if edges.size < 2:                       # a constant series
        return np.zeros(v.size, dtype=np.int64), 1
    n_bins = edges.size - 1
    idx = np.clip(np.searchsorted(edges, v, side="left") - 1, 0, n_bins - 1)
    return idx.astype(np.int64), n_bins


def mi_binned(x: np.ndarray, y: np.ndarray, bins: int = 6,
              scheme: str = "quantile", correct_bias: bool = True) -> float:
    """Plug-in (histogram) mutual information estimator, in nats.

    Parameters
    ----------
    x, y : np.ndarray
        Equal-length samples.
    bins : int, default 6
        Bins per marginal. The joint histogram has bins^2 cells.
    scheme : {'quantile', 'uniform'}, default 'quantile'
        'quantile' gives equal-frequency bins, 'uniform' equal-width.
    correct_bias : bool, default True
        Subtract the Miller-Madow term `(bx - 1) * (by - 1) / (2T)`, where bx
        and by count NON-EMPTY marginal bins. The plug-in estimator's bias is
        systematic and upward, so this matters.

    Returns
    -------
    float
        MI in nats, clipped at 0 from below.

    Notes
    -----
    Quantile binning is the default because returns are heavy-tailed. With
    equal-width bins, nearly all the mass lands in the central bins and the
    tails sit almost empty -- precisely the opposite of what is wanted, since
    the tails are where the interesting joint behaviour lives. Equal-frequency
    bins put the resolution where the data actually is.

    At T = 252 the sample size is the binding constraint: bins^2 cells against
    252 points is ~10 samples per cell at b=5 and under 1 at b=16. Below about
    5 per cell the estimate is dominated by noise, which is why the KSG
    estimator is preferred for production runs.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(f"length mismatch: {x.size} vs {y.size}")
    n = x.size

    ix, nx = _bin_index(x, bins, scheme)
    iy, ny = _bin_index(y, bins, scheme)

    joint = np.bincount(ix * ny + iy, minlength=nx * ny).reshape(nx, ny)
    pxy = joint / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)

    nz = pxy > 0                              # empty cells contribute nothing
    mi = float((pxy[nz] * np.log(pxy[nz] / (px * py)[nz])).sum())

    if correct_bias:
        bx = int((px > 0).sum())
        by = int((py > 0).sum())
        mi -= (bx - 1) * (by - 1) / (2.0 * n)

    return max(mi, 0.0)


def mi_ksg(x: np.ndarray, y: np.ndarray, n_neighbors: int = 4) -> float:
    """Kraskov-Stogbauer-Grassberger mutual information, in nats.

    Parameters
    ----------
    x, y : np.ndarray
        Equal-length samples.
    n_neighbors : int, default 4
        The KSG kappa. Small values reduce bias and raise variance.

    Returns
    -------
    float
        MI in nats, clipped at 0 from below.

    Notes
    -----
    KSG avoids binning entirely. It works from k-nearest-neighbour distances
    in the joint space, so the effective resolution adapts to local density --
    fine where points are dense, coarse where they are sparse. That makes it
    far less biased than a histogram at T = 252, where any fixed binning is
    either too coarse to see structure or too fine to have samples in its
    cells.

    Implemented via sklearn's `mutual_info_regression`, which breaks ties with
    a tiny noise draw; `random_state` is pinned so runs are reproducible. That
    noise also makes the estimator only *approximately* symmetric in its two
    arguments (to ~1e-9), which is why `mi_matrix` computes each unordered
    pair once and mirrors it rather than computing both directions.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(f"length mismatch: {x.size} vs {y.size}")

    val = mutual_info_regression(
        x.reshape(-1, 1), y,
        discrete_features=False,
        n_neighbors=n_neighbors,
        random_state=KSG_RANDOM_STATE,
    )[0]
    return max(float(val), 0.0)


_ESTIMATORS = {"binned": mi_binned, "ksg": mi_ksg}


def _estimator_fn(estimator: str):
    try:
        return _ESTIMATORS[estimator]
    except KeyError:
        raise ValueError(
            f"unknown estimator {estimator!r}; expected one of {sorted(_ESTIMATORS)}"
        ) from None


# ----------------------------------------------------------------------
# Matrices
# ----------------------------------------------------------------------
def _pair_chunk(R: np.ndarray, pairs: np.ndarray, estimator: str,
                kwargs: dict) -> list[float]:
    """Estimate MI for a chunk of (i, j) pairs. Runs inside a joblib worker."""
    fn = _estimator_fn(estimator)
    return [fn(R[:, i], R[:, j], **kwargs) for i, j in pairs]


def mi_matrix(R: np.ndarray, estimator: str = "ksg", n_jobs: int = -1,
              **kwargs) -> np.ndarray:
    """Full N x N mutual information matrix for a return window.

    Parameters
    ----------
    R : np.ndarray
        Return window of shape [T_win, N].
    estimator : {'ksg', 'binned'}, default 'ksg'
    n_jobs : int, default -1
        Passed to joblib.Parallel; -1 uses every core.
    **kwargs
        Forwarded to the estimator (`n_neighbors`, or `bins`/`scheme`/
        `correct_bias`).

    Returns
    -------
    np.ndarray
        Symmetric N x N matrix in nats, with a ZERO diagonal.

    Notes
    -----
    The diagonal is set to 0.0 rather than to the marginal entropy. Either is
    acceptable since `knn_adjacency` masks the diagonal to -inf before
    selecting neighbours, but zero is the choice made here and the module is
    consistent about it -- a diagonal carrying entropy would be the single
    largest entry in every row and would silently become each node's own
    nearest neighbour in any code that forgot to mask it.

    Only the N*(N-1)/2 upper-triangle pairs are computed; the result is
    mirrored, since MI is symmetric.
    """
    R = np.asarray(R, dtype=float)
    n = R.shape[1]
    iu = np.triu_indices(n, k=1)
    pairs = np.column_stack(iu)

    n_chunks = max(1, min(len(pairs), 64))
    chunks = np.array_split(pairs, n_chunks)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_pair_chunk)(R, c, estimator, kwargs) for c in chunks
    )
    values = np.concatenate([np.asarray(r, dtype=float) for r in results])

    M = np.zeros((n, n), dtype=float)
    M[iu] = values
    M = M + M.T                               # mirror; diagonal stays 0
    return M


def marginal_entropies(R: np.ndarray, bins: int = 6) -> np.ndarray:
    """Per-asset Shannon entropy of the binned return series, in nats.

    Parameters
    ----------
    R : np.ndarray
        Return window of shape [T_win, N].
    bins : int, default 6
        Bins per series, equal-frequency (quantile), matching `mi_binned`'s
        default so the two are on the same footing.

    Returns
    -------
    np.ndarray
        Length-N vector of entropies.

    Notes
    -----
    With equal-frequency bins every series is spread evenly by construction,
    so H sits close to log(bins) for every asset and varies only through ties
    (exact-zero returns on stale-price days) and bin collapse. The
    normalisation in `normalise_mi` is therefore close to a uniform rescaling
    on this data. That is worth knowing rather than assuming: the run reports
    the actual spread of H.
    """
    R = np.asarray(R, dtype=float)
    out = np.zeros(R.shape[1], dtype=float)
    for j in range(R.shape[1]):
        idx, nb = _bin_index(R[:, j], bins, "quantile")
        p = np.bincount(idx, minlength=nb) / R.shape[0]
        p = p[p > 0]
        out[j] = float(-(p * np.log(p)).sum())
    return out


def normalise_mi(M: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Normalised MI, `NMI_ij = M_ij / sqrt(H_i * H_j)`, clipped to [0, 1].

    Parameters
    ----------
    M : np.ndarray
        Raw MI matrix, N x N.
    H : np.ndarray
        Marginal entropies, length N.

    Returns
    -------
    np.ndarray
        N x N normalised MI in [0, 1].

    Notes
    -----
    Raw MI scales with marginal entropy, so a high-entropy (volatile) asset
    shows large MI against everything. Left unnormalised, volatility would
    leak into the neighbour ranking -- and volatility is already a node
    feature (`vol_21`, `vol_63`). The graph would then be re-encoding
    information the model already has, instead of contributing the pairwise
    structure it is there to contribute.
    """
    H = np.asarray(H, dtype=float)
    denom = np.sqrt(np.outer(H, H))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom > 0, np.asarray(M, dtype=float) / denom, 0.0)
    return np.clip(out, 0.0, 1.0)


def mi_similarity(M: np.ndarray, C: np.ndarray, H: np.ndarray,
                  signed: bool) -> np.ndarray:
    """The similarity matrix fed to `knn_adjacency`.

    Parameters
    ----------
    M : np.ndarray
        Raw MI matrix.
    C : np.ndarray
        Correlation matrix over the same window, used only for its sign.
    H : np.ndarray
        Marginal entropies, for the normalisation.
    signed : bool
        If True, zero every negative-correlation entry so only same-direction
        dependence can form an edge. MI depends on rho^2, so without this the
        graph wires an asset to its strongest hedge as readily as to its
        closest peer.

    Returns
    -------
    np.ndarray
        N x N similarity in [0, 1].
    """
    NMI = normalise_mi(M, H)
    if not signed:
        return NMI
    return np.where(np.asarray(C, dtype=float) > 0, NMI, 0.0)


def nonlinear_excess(M: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Dependence not explained by linear correlation: `M - gaussian_mi(C)`.

    Parameters
    ----------
    M : np.ndarray
        Estimated MI matrix, in nats.
    C : np.ndarray
        Correlation matrix over the same window.

    Returns
    -------
    np.ndarray
        Delta, elementwise, in nats.

    Notes
    -----
    Delta is the dependence that linear correlation misses, measured in nats.
    Read it plainly: if Delta is approximately zero across the universe, then
    mutual information is reproducing the correlation graph at far greater
    computational cost. That is a legitimate and reportable finding, not a
    failure of the method -- it is the answer to the question this module was
    built to ask.

    Delta can be negative where the estimator undershoots the Gaussian value;
    that is estimation error, not negative nonlinear dependence.
    """
    return np.asarray(M, dtype=float) - gaussian_mi(np.asarray(C, dtype=float))


# ----------------------------------------------------------------------
# Validation and cost
# ----------------------------------------------------------------------
def validate_estimator(rhos: list[float], T: int = 252, n_trials: int = 20,
                       estimators: tuple = ("binned", "ksg"),
                       seed: int = 0, **kwargs) -> pd.DataFrame:
    """Recover the Gaussian MI curve; the estimator's unit test as a report.

    For each rho, draws `n_trials` bivariate Gaussian samples of length `T`
    with that correlation, estimates MI with each estimator, and compares
    against the exact `gaussian_mi(rho)`.

    Returns
    -------
    pd.DataFrame
        Columns: rho, mi_true, and per estimator `<est>_mean`, `<est>_std`,
        `<est>_bias` (mean estimate minus truth).

    Notes
    -----
    Run as a printed diagnostic, not only as a pytest case. If an estimator
    cannot reproduce this curve at the sample size actually in use, then no
    downstream MI number in this module is interpretable, and the right
    response is to fix the estimator rather than to proceed.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for rho in rhos:
        cov = np.array([[1.0, rho], [rho, 1.0]])
        est: dict[str, list[float]] = {e: [] for e in estimators}
        for _ in range(n_trials):
            s = rng.multivariate_normal([0.0, 0.0], cov, size=T)
            for e in estimators:
                fn = _estimator_fn(e)
                est[e].append(fn(s[:, 0], s[:, 1], **kwargs.get(e, {})))

        row = {"rho": rho, "mi_true": gaussian_mi(rho)}
        for e in estimators:
            v = np.asarray(est[e])
            row[f"{e}_mean"] = float(v.mean())
            row[f"{e}_std"] = float(v.std())
            row[f"{e}_bias"] = float(v.mean() - row["mi_true"])
        rows.append(row)
    return pd.DataFrame(rows)


def mi_null_floor(R: np.ndarray, estimator: str = "ksg", n_perm: int = 100,
                  n_pairs: int = 200, seed: int = 0, n_jobs: int = -1,
                  **kwargs) -> dict:
    """Permutation null distribution for MI on this window.

    Samples `n_pairs` random asset pairs, and for each draws `n_perm`
    permutations of one series -- destroying any dependence while preserving
    the marginal distribution exactly -- then estimates MI.

    Returns
    -------
    dict
        `mean`, `std`, `p95`, `p99` and `n_draws` of the null distribution.

    Notes
    -----
    This exists because MI is non-negative. Sampling error has nowhere to
    cancel, so two genuinely independent series still produce MI_hat > 0.
    Correlation needs no such null model: its errors are signed and average
    out around zero. The 95th percentile is the threshold below which an
    observed MI is indistinguishable from noise, and on a short window that
    threshold is emphatically not zero.
    """
    R = np.asarray(R, dtype=float)
    t, n = R.shape
    rng = np.random.default_rng(seed)

    i = rng.integers(0, n, size=n_pairs)
    j = (i + rng.integers(1, n, size=n_pairs)) % n        # never i == j

    def _one(a: int, b: int, s: int) -> list[float]:
        fn = _estimator_fn(estimator)
        r = np.random.default_rng(s)
        x = R[:, a]
        return [fn(x, r.permutation(R[:, b]), **kwargs) for _ in range(n_perm)]

    seeds = rng.integers(0, 2**31 - 1, size=n_pairs)
    out = Parallel(n_jobs=n_jobs)(
        delayed(_one)(int(a), int(b), int(s)) for a, b, s in zip(i, j, seeds)
    )
    draws = np.concatenate([np.asarray(o, dtype=float) for o in out])

    return {
        "mean": float(draws.mean()),
        "std": float(draws.std()),
        "p95": float(np.percentile(draws, 95)),
        "p99": float(np.percentile(draws, 99)),
        "n_draws": int(draws.size),
    }


def timing_probe(R: np.ndarray, estimator: str, n_pairs: int = 200,
                 n_dates: int = 128, **kwargs) -> dict:
    """Time `n_pairs` MI estimates and extrapolate to the full run.

    Returns
    -------
    dict
        `seconds_per_pair`, `seconds_per_matrix`, `seconds_full_run`,
        `n_pairs_per_matrix`.

    Notes
    -----
    Printed BEFORE any full run, so the compute cost is known in advance
    rather than discovered halfway through one. Correlation is a single matrix
    multiply; this is 5,460 estimates per date.
    """
    R = np.asarray(R, dtype=float)
    n = R.shape[1]
    rng = np.random.default_rng(0)
    fn = _estimator_fn(estimator)

    a = rng.integers(0, n, size=n_pairs)
    b = (a + rng.integers(1, n, size=n_pairs)) % n

    t0 = time.perf_counter()
    for p, q in zip(a, b):
        fn(R[:, p], R[:, q], **kwargs)
    elapsed = time.perf_counter() - t0

    per_pair = elapsed / n_pairs
    pairs_per_matrix = n * (n - 1) / 2
    return {
        "seconds_per_pair": per_pair,
        "n_pairs_per_matrix": int(pairs_per_matrix),
        "seconds_per_matrix": per_pair * pairs_per_matrix,
        "seconds_full_run": per_pair * pairs_per_matrix * n_dates,
    }


# ----------------------------------------------------------------------
# Sequence construction
# ----------------------------------------------------------------------
def build_mi_graph_sequence(ret: pd.DataFrame, dates: pd.DatetimeIndex,
                            C_stack: np.ndarray, cfg: dict, k: int,
                            estimator: str, signed: bool,
                            cache_dir: Path | None = None,
                            ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build the MI-kNN graph sequence on the rebalance clock.

    Parameters
    ----------
    ret : pd.DataFrame
        Wide date x ticker log returns.
    dates : pd.DatetimeIndex
        Rebalance dates; each must be a session in `ret.index`.
    C_stack : np.ndarray
        Part A's correlation matrices for the SAME dates, [T, N, N].
    cfg : dict
        Full config; `cfg['graph']` and `cfg['mi']` are both read.
    k : int
        Neighbours per node, matching Part A.
    estimator : {'ksg', 'binned'}
    signed : bool
        If True, `S = sign(C) * NMI` with negatives zeroed, so only
        same-direction dependence can form an edge. If False, `S = NMI`, and
        hedges become neighbours.
    cache_dir : Path or None
        If given, each date's raw MI matrix is written to
        `mi_{estimator}_{date}.npy` and reused on a rerun.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, pd.DataFrame]
        `(A, M_stack, meta)`; meta has mean_mi, mean_nmi, mean_delta and
        frac_above_null per date. The kNN selection mask is carried on
        `meta.attrs['A_binary']` as a [T, N, N] uint8 stack.

    Notes
    -----
    Under sign-masking the union guarantee of degree >= k holds for the
    SELECTION, not for the non-zero weights: an asset with fewer than k
    positive-correlation partners has the remainder of its k slots filled by
    entries whose similarity is exactly 0. Those are real selections carrying
    no signal, so `A` (weighted) can show a degree below k even though
    `A_binary` does not. The run reports how often this happens.
    """
    gcfg = cfg["graph"]
    micfg = cfg.get("mi", {})
    window = int(gcfg["corr_window"])
    n_factors = int(gcfg["n_factors"])
    symmetrise = str(gcfg.get("symmetrise", "union"))
    weighted = bool(gcfg.get("weighted", True))
    bins = int(micfg.get("bins", 6))

    est_kwargs = _estimator_kwargs(micfg, estimator)
    null_p95 = float(micfg.get("_null_p95", 0.0))

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    values = ret.to_numpy(dtype=float)
    index = ret.index
    n = ret.shape[1]
    iu = np.triu_indices(n, k=1)

    A = np.zeros((len(dates), n, n), dtype=np.float32)
    A_bin = np.zeros((len(dates), n, n), dtype=np.uint8)
    M_stack = np.zeros((len(dates), n, n), dtype=np.float32)
    rows = []

    for i, d in enumerate(dates):
        pos = index.get_loc(d)
        if not isinstance(pos, (int, np.integer)):
            raise ValueError(f"rebalance date {d} is not a unique trading session")
        start = pos + 1 - window
        if start < 0:
            raise ValueError(
                f"{d.date()} is at position {pos}; a {window}-session window "
                f"needs {window - 1 - pos} more sessions of history"
            )

        cache = (cache_dir / f"mi_{estimator}_{pd.Timestamp(d).date()}.npy"
                 if cache_dir is not None else None)
        Rw = residualise(values[start:pos + 1], n_factors)

        if cache is not None and cache.exists():
            M = np.load(cache)
        else:
            M = mi_matrix(Rw, estimator=estimator, **est_kwargs)
            if cache is not None:
                np.save(cache, M)

        H = marginal_entropies(Rw, bins=bins)
        NMI = normalise_mi(M, H)
        C = np.asarray(C_stack[i], dtype=float)

        S = mi_similarity(M, C, H, signed)

        A[i] = knn_adjacency(S, k, symmetrise, weighted).astype(np.float32)
        # Recorded separately, never as (A != 0): sign-masking sets some
        # similarities to exactly 0, so a selected edge can carry zero weight
        # and would otherwise vanish from the binary graph. Part A derives its
        # A_binary from the same selection mask, and the two files have to
        # mean the same thing for the comparison to be valid.
        A_bin[i] = (knn_adjacency(S, k, symmetrise, weighted=False) != 0)
        M_stack[i] = M.astype(np.float32)

        delta = nonlinear_excess(M, C)
        rows.append({
            "mean_mi": float(M[iu].mean()),
            "mean_nmi": float(NMI[iu].mean()),
            "mean_delta": float(delta[iu].mean()),
            "frac_above_null": float((M[iu] > null_p95).mean()),
        })
        print(f"    [{i + 1}/{len(dates)}] {pd.Timestamp(d).date()}  "
              f"mean MI {rows[-1]['mean_mi']:.4f}  "
              f"mean Delta {rows[-1]['mean_delta']:+.4f}")

    meta = pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="date"))
    # The kNN selection mask travels on meta.attrs so the documented 3-tuple
    # return stays intact; main() saves it as A_binary.
    meta.attrs["A_binary"] = A_bin
    return A, M_stack, meta


def _estimator_kwargs(micfg: dict, estimator: str) -> dict:
    """Pull only the keys the chosen estimator actually accepts."""
    if estimator == "ksg":
        return {"n_neighbors": int(micfg.get("n_neighbors", 4))}
    if estimator == "binned":
        return {
            "bins": int(micfg.get("bins", 6)),
            "scheme": str(micfg.get("bin_scheme", "quantile")),
        }
    raise ValueError(f"unknown estimator: {estimator!r}")


def compare_graphs(A_corr: np.ndarray, A_mi: np.ndarray) -> pd.DataFrame:
    """Per-date comparison of the correlation and MI graphs.

    Returns
    -------
    pd.DataFrame
        Columns edge_jaccard, persistence_corr, persistence_mi. Persistence is
        each graph's Jaccard overlap with its own previous date, so the first
        row is NaN.

    Notes
    -----
    A high `edge_jaccard` means MI is recovering the correlation graph, and
    the considerable extra cost buys nothing -- which is a finding, not a
    failure.

    Because MI noise is one-sided, its persistence may well be LOWER than
    correlation's: a spurious high MI cannot be cancelled by a spurious low
    one, so the neighbour ranking is noisier date to date even when the
    underlying dependence is identical.
    """
    if A_corr.shape != A_mi.shape:
        raise ValueError(f"shape mismatch: {A_corr.shape} vs {A_mi.shape}")

    rows = []
    for i in range(len(A_corr)):
        rows.append({
            "edge_jaccard": edge_persistence(A_corr[i], A_mi[i]),
            "persistence_corr": (edge_persistence(A_corr[i - 1], A_corr[i])
                                 if i else np.nan),
            "persistence_mi": (edge_persistence(A_mi[i - 1], A_mi[i])
                               if i else np.nan),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def main() -> None:
    cfg = yaml.safe_load(open("config/base.yaml"))
    gcfg, micfg = cfg["graph"], cfg["mi"]
    proc = Path(cfg["data"]["processed_dir"])
    estimator = str(micfg["estimator"])
    k = int(gcfg["k"])

    print("[1] loading Part A output and aligning returns")
    g = np.load(proc / "graphs.npz", allow_pickle=True)
    tickers = [str(t) for t in g["tickers"]]
    all_dates = pd.DatetimeIndex(pd.to_datetime([str(s) for s in g["dates"]]))
    ret = load_returns(proc / "panel.parquet", tickers)
    assert list(ret.columns) == tickers, "ticker order does not match graphs.npz"
    print(f"  {len(all_dates)} dates, {len(tickers)} tickers, estimator={estimator}")

    print("\n[2] estimator validation against the Gaussian identity")
    table = validate_estimator([0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
                               T=int(gcfg["corr_window"]), n_trials=20)
    print(table.round(4).to_string(index=False))

    # A mid-sample window, used for the null floor and the timing probe.
    mid = len(all_dates) // 2
    pos = ret.index.get_loc(all_dates[mid])
    window = int(gcfg["corr_window"])
    R_mid = residualise(ret.to_numpy(dtype=float)[pos + 1 - window:pos + 1],
                        int(gcfg["n_factors"]))
    est_kwargs = _estimator_kwargs(micfg, estimator)

    print(f"\n[3] permutation null floor on {all_dates[mid].date()}")
    null = mi_null_floor(R_mid, estimator=estimator,
                         n_perm=int(micfg["n_perm"]), **est_kwargs)
    print(f"  draws {null['n_draws']:,}  mean {null['mean']:.4f}  "
          f"sd {null['std']:.4f}  p95 {null['p95']:.4f}  p99 {null['p99']:.4f}")
    micfg["_null_p95"] = null["p95"]

    sub = micfg.get("subsample_dates")
    n_run = len(all_dates) if not sub else int(sub)
    print(f"\n[4] timing probe ({estimator})")
    probe = timing_probe(R_mid, estimator, n_dates=n_run, **est_kwargs)
    print(f"  {probe['seconds_per_pair'] * 1e3:.3f} ms/pair x "
          f"{probe['n_pairs_per_matrix']:,} pairs = "
          f"{probe['seconds_per_matrix']:.1f} s/matrix (serial)")
    print(f"  projected for {n_run} dates: "
          f"{probe['seconds_full_run'] / 60:.1f} min serial "
          f"(parallelised across cores in practice)")

    if sub:
        pick = np.linspace(0, len(all_dates) - 1, int(sub)).round().astype(int)
        pick = np.unique(pick)
    else:
        pick = np.arange(len(all_dates))
    dates = all_dates[pick]
    C_stack = g["C"][pick]
    A_corr = g["A"][pick]
    print(f"\n[5] building MI graphs on {len(dates)} dates "
          f"({dates[0].date()} to {dates[-1].date()})")

    cache_dir = micfg.get("cache_dir")
    A, M_stack, meta = build_mi_graph_sequence(
        ret, dates, C_stack, cfg, k, estimator,
        signed=bool(micfg.get("signed", True)),
        cache_dir=Path(cache_dir) if cache_dir else None,
    )

    print("\n[6] comparing against the correlation graph")
    comp = compare_graphs(A_corr, A).set_index(pd.DatetimeIndex(dates, name="date"))
    print(comp.round(4).to_string())

    Delta = np.stack([nonlinear_excess(M_stack[i].astype(float),
                                       C_stack[i].astype(float))
                      for i in range(len(dates))]).astype(np.float32)

    out = proc / "graphs_mi.npz"
    np.savez_compressed(
        out,
        A=A,
        A_binary=meta.attrs["A_binary"],
        M=M_stack,
        Delta=Delta,
        dates=np.array([d.strftime("%Y-%m-%d") for d in dates], dtype=object),
        tickers=np.array(tickers, dtype=object),
        config_json=np.array(json.dumps(
            {"graph": gcfg, "mi": {k_: v for k_, v in micfg.items()
                                   if not k_.startswith("_")}},
            sort_keys=True, default=str)),
    )
    print(f"\nwrote {out}  A.shape={A.shape}")

    iu = np.triu_indices(A.shape[1], k=1)
    print(f"\nestimator                    : {estimator} "
          f"(signed={bool(micfg.get('signed', True))})")
    print(f"mean NMI                     : {meta['mean_nmi'].mean():.4f}")
    print(f"mean Delta (nonlinear excess): {meta['mean_delta'].mean():+.4f} nats")
    print(f"frac pairs above null p95    : {meta['frac_above_null'].mean():.1%}")
    deg_w = (A != 0).sum(axis=2)
    deg_b = (meta.attrs["A_binary"] != 0).sum(axis=2)
    starved = int((deg_w < k).sum())
    print(f"degree (selected)            : min {deg_b.min()}, mean "
          f"{deg_b.mean():.2f}, max {deg_b.max()}  (k={k})")
    print(f"node-dates with < k weighted : {starved} of {deg_w.size} "
          f"({starved / deg_w.size:.2%}) -- sign-masking left them short of "
          f"k positive partners")
    print(f"mean edge Jaccard vs corr    : {comp['edge_jaccard'].mean():.4f}")
    print(f"mean persistence corr / MI   : "
          f"{comp['persistence_corr'].mean():.4f} / "
          f"{comp['persistence_mi'].mean():.4f}")


if __name__ == "__main__":
    main()
