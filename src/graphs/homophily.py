"""Phase 3c: homophily and feature-smoothness diagnostics for the graphs.

Reads  : data/processed/graphs.npz      (Part A, correlation kNN)
         data/processed/graphs_mi.npz   (Part B, mutual information)
         data/processed/features.npz    (the [T, N, F] node-feature tensor)
         data/universe_sp100.csv        (GICS sector per ticker)
Writes : data/processed/homophily.csv   (per-date metrics for both graphs)

Why this exists
---------------
These metrics decide the Phase 4 architecture. Standard message passing (GCN,
GraphSAGE) averages over each node's neighbourhood, and averaging only
concentrates signal when neighbours are actually similar. On a heterophilous
graph the same operation destroys signal instead, and a heterophily-aware
architecture is needed.

So the property is measured here, before the architecture is chosen, rather
than inferred afterwards from a model that underperforms for reasons nobody
can pin down.

Two families of metric, and they answer different questions:

- **Label homophily** (`edge_homophily`, `assortativity`) asks whether edges
  join assets in the same GICS sector. Sector plays no part in construction,
  so this is an external check that the edges mean something economically.
- **Feature smoothness** (`feature_smoothness`, `per_feature_smoothness`) asks
  whether connected assets are close in the feature space the GNN will
  actually consume. This is the one that governs the architecture choice: the
  model reads X, not sector labels.

This module measures graph properties only. No GNN, no clustering, no Phase 4.

Run from repo root:  python -m src.graphs.homophily
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import yaml


# ----------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------
def load_sector_labels(universe_csv: str | Path,
                       tickers: list[str]) -> np.ndarray:
    """GICS sector per ticker, in the exact order of `tickers`.

    Parameters
    ----------
    universe_csv : str or Path
        data/universe_sp100.csv, with `yahoo_ticker` and `sector` columns.
    tickers : list[str]
        Canonical ticker order, taken from the artefact being analysed.

    Returns
    -------
    np.ndarray
        Length-N array of sector strings.

    Raises
    ------
    AssertionError
        If any ticker has no sector, naming every offender.

    Notes
    -----
    This fails loudly on purpose. A missing sector label crashes nothing
    downstream -- it simply counts as "different sector" for every edge
    touching that node, which biases homophily downward by a quiet, plausible
    amount that no one would question. Silence is the danger here, not the
    crash.

    That failure has already happened in this project, via the renamed tickers
    BNY and FISV. It is also why the join key is normalised below: the CSV
    stores Yahoo symbols with a dot (BRK.B) while src/data/ingest.py converts
    them to a dash (BRK-B) before download, so every artefact downstream
    carries the dash form. Joining on the raw column leaves BRK-B unmatched.
    """
    uni = pd.read_csv(universe_csv)
    key = uni["yahoo_ticker"].astype(str).str.replace(".", "-", regex=False)
    sectors = uni.assign(_key=key).set_index("_key")["sector"].reindex(tickers)

    missing = [t for t, s in zip(tickers, sectors) if pd.isna(s)]
    assert not missing, (
        f"no GICS sector for {len(missing)} ticker(s): {missing}. "
        f"An unlabelled node counts as 'different sector' on every edge it "
        f"touches and biases homophily downward -- fix the universe CSV "
        f"rather than dropping the node."
    )
    return sectors.to_numpy()


# ----------------------------------------------------------------------
# Label homophily
# ----------------------------------------------------------------------
def edge_homophily(A: np.ndarray, labels: np.ndarray) -> dict:
    """Fraction of edges joining same-label nodes, raw and adjusted.

    Parameters
    ----------
    A : np.ndarray
        N x N adjacency; binarised internally, upper triangle only.
    labels : np.ndarray
        Length-N node labels.

    Returns
    -------
    dict
        `raw` -- fraction of edges whose endpoints share a label.
        `expected` -- sum of squared class proportions, the value under
        random mixing.
        `adjusted` -- `(raw - expected) / (1 - expected)`.

    Notes
    -----
    **`expected` is not 1/n_classes** unless the classes are equal in size,
    which is why the raw fraction alone is uninterpretable. With 11 GICS
    sectors of unequal size, random wiring already produces a raw homophily of
    about 0.117, not 0.091, and a raw figure of 0.30 means something very
    different depending on which baseline it is measured against.

    Quote `adjusted`: 0 is random mixing, 1 is perfect homophily, and negative
    is heterophily -- edges joining different sectors more often than chance.
    """
    labels = np.asarray(labels)
    n = len(labels)
    iu = np.triu_indices(n, k=1)

    edges = (np.asarray(A) != 0)[iu]
    same = (labels[iu[0]] == labels[iu[1]])

    n_edges = int(edges.sum())
    raw = float(same[edges].mean()) if n_edges else float("nan")

    _, counts = np.unique(labels, return_counts=True)
    expected = float(((counts / n) ** 2).sum())

    adjusted = (raw - expected) / (1.0 - expected) if expected < 1.0 else float("nan")
    return {"raw": raw, "expected": expected, "adjusted": adjusted}


def assortativity(A: np.ndarray, labels: np.ndarray) -> float:
    """Newman's attribute assortativity coefficient for a categorical label.

    Parameters
    ----------
    A : np.ndarray
        N x N adjacency, binarised internally.
    labels : np.ndarray
        Length-N node labels.

    Returns
    -------
    float
        Assortativity in [-1, 1]; NaN if the graph has no edges.

    Notes
    -----
    Reported alongside `edge_homophily` rather than instead of it. The two
    handle class imbalance differently: assortativity normalises against the
    mixing matrix's own marginals, so it is not simply a rescaling of the
    adjusted edge homophily, and the pair together is more informative than
    either alone.
    """
    B = (np.asarray(A) != 0).astype(int)
    np.fill_diagonal(B, 0)
    if B.sum() == 0:
        return float("nan")

    G = nx.from_numpy_array(B)
    nx.set_node_attributes(G, {i: str(l) for i, l in enumerate(labels)}, "label")
    return float(nx.attribute_assortativity_coefficient(G, "label"))


# ----------------------------------------------------------------------
# Feature smoothness
# ----------------------------------------------------------------------
def _pairwise_sq_dists(X_t: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance between every pair of rows."""
    X_t = np.asarray(X_t, dtype=float)
    sq = (X_t ** 2).sum(axis=1)
    d = sq[:, None] + sq[None, :] - 2.0 * (X_t @ X_t.T)
    return np.maximum(d, 0.0)              # clip floating-point negatives


def feature_smoothness(A: np.ndarray, X_t: np.ndarray) -> float:
    """Normalised Dirichlet energy: connected-pair distance over all-pair distance.

    Parameters
    ----------
    A : np.ndarray
        N x N adjacency, binarised internally.
    X_t : np.ndarray
        N x F node features for the same date.

    Returns
    -------
    float
        Mean squared feature distance across connected pairs, divided by the
        mean across all pairs. NaN if the graph has no edges.

    Notes
    -----
    **This is the metric that matters most for Phase 4**, because the GNN
    consumes X, not sector labels.

    Well below 1: connected nodes are closer in feature space than random
    pairs, so neighbourhood averaging concentrates signal and a standard GCN
    or GraphSAGE is appropriate.

    Near or above 1: averaging a neighbourhood blurs rather than sharpens, and
    a vanilla GCN is the wrong architecture -- a heterophily-aware design is
    needed instead.
    """
    n = np.asarray(A).shape[0]
    iu = np.triu_indices(n, k=1)

    d = _pairwise_sq_dists(X_t)[iu]
    e = (np.asarray(A) != 0)[iu]
    if not e.any() or d.mean() == 0:
        return float("nan")
    return float(d[e].mean() / d.mean())


def per_feature_smoothness(A: np.ndarray, X_t: np.ndarray,
                           feature_names: list[str]) -> pd.Series:
    """`feature_smoothness` computed one feature at a time.

    Returns
    -------
    pd.Series
        Smoothness ratio per feature, indexed by feature name.

    Notes
    -----
    Sharper than the aggregate. Averaging over all F features can hide the
    fact that some are smooth over the graph while others are not, and knowing
    *which* features the graph smooths is a far more specific claim about what
    information the edges actually carry than any single pooled number.
    """
    X_t = np.asarray(X_t, dtype=float)
    n = X_t.shape[0]
    iu = np.triu_indices(n, k=1)
    e = (np.asarray(A) != 0)[iu]

    out = {}
    for j, name in enumerate(feature_names):
        v = X_t[:, j]
        d = ((v[:, None] - v[None, :]) ** 2)[iu]
        out[name] = (float(d[e].mean() / d.mean())
                     if e.any() and d.mean() > 0 else float("nan"))
    return pd.Series(out, name="smoothness")


# ----------------------------------------------------------------------
# Sequence analysis
# ----------------------------------------------------------------------
def analyse(A_stack: np.ndarray, X: np.ndarray, labels: np.ndarray,
            dates: pd.DatetimeIndex, feature_names: list[str],
            name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Run every metric over a whole graph sequence.

    Parameters
    ----------
    A_stack : np.ndarray
        [T, N, N] adjacency stack.
    X : np.ndarray
        [T, N, F] feature tensor, aligned to `A_stack` and `dates`.
    labels : np.ndarray
        Length-N node labels.
    dates : pd.DatetimeIndex
        Length-T index.
    feature_names : list[str]
        Length-F feature names.
    name : str
        Graph name, written into the `graph` column.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Per-date metrics, and per-feature smoothness averaged over dates.
    """
    rows, per_feat = [], []
    for i in range(len(A_stack)):
        A, X_t = A_stack[i], X[i]
        h = edge_homophily(A, labels)
        rows.append({
            "graph": name,
            "homophily_raw": h["raw"],
            "homophily_expected": h["expected"],
            "homophily_adjusted": h["adjusted"],
            "assortativity": assortativity(A, labels),
            "smoothness": feature_smoothness(A, X_t),
        })
        per_feat.append(per_feature_smoothness(A, X_t, feature_names))

    per_date = pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="date"))
    return per_date, pd.concat(per_feat, axis=1).mean(axis=1)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def main() -> None:
    cfg = yaml.safe_load(open("config/base.yaml"))
    proc = Path(cfg["data"]["processed_dir"])

    print("[1] loading artefacts")
    g = np.load(proc / "graphs.npz", allow_pickle=True)
    gm = np.load(proc / "graphs_mi.npz", allow_pickle=True)
    f = np.load(proc / "features.npz", allow_pickle=True)

    tickers = [str(t) for t in g["tickers"]]
    # Node order must be identical across all three or every metric silently
    # compares the wrong assets.
    assert tickers == [str(t) for t in gm["tickers"]], \
        "ticker order differs between graphs.npz and graphs_mi.npz"
    assert tickers == [str(t) for t in f["tickers"]], \
        "ticker order differs between graphs.npz and features.npz"

    feature_names = [str(x) for x in f["feature_names"]]
    labels = load_sector_labels("data/universe_sp100.csv", tickers)
    vals, counts = np.unique(labels, return_counts=True)
    print(f"  {len(tickers)} tickers, {len(vals)} sectors, "
          f"{len(feature_names)} features")

    # The MI run may have used subsample_dates, so intersect rather than
    # assume the two stacks cover the same dates.
    d_corr = [str(s) for s in g["dates"]]
    d_mi = [str(s) for s in gm["dates"]]
    d_feat = [str(s) for s in f["dates"]]
    common = sorted(set(d_corr) & set(d_mi) & set(d_feat))
    print(f"[2] aligning on {len(common)} common dates "
          f"(corr {len(d_corr)}, mi {len(d_mi)}, features {len(d_feat)})")

    i_corr = [d_corr.index(d) for d in common]
    i_mi = [d_mi.index(d) for d in common]
    i_feat = [d_feat.index(d) for d in common]
    dates = pd.DatetimeIndex(pd.to_datetime(common))
    X = f["X"][i_feat]

    print("[3] correlation graph")
    corr_per_date, corr_per_feat = analyse(
        g["A"][i_corr], X, labels, dates, feature_names, "correlation")
    print("[4] mutual-information graph")
    mi_per_date, mi_per_feat = analyse(
        gm["A"][i_mi], X, labels, dates, feature_names, "mutual_information")

    out = proc / "homophily.csv"
    pd.concat([corr_per_date, mi_per_date]).to_csv(out)
    print(f"\nwrote {out}")

    cols = ["homophily_raw", "homophily_expected", "homophily_adjusted",
            "assortativity", "smoothness"]
    summary = pd.DataFrame({
        "correlation": corr_per_date[cols].mean(),
        "mutual_information": mi_per_date[cols].mean(),
    })
    summary["difference"] = summary["correlation"] - summary["mutual_information"]
    print("\nmeans across dates:")
    print(summary.round(4).to_string())

    print("\nper-feature smoothness, correlation graph (ascending = smoothest):")
    ranked = corr_per_feat.sort_values()
    for k, v in ranked.items():
        print(f"  {k:<18} {v:.4f}")

    agg = corr_per_date["smoothness"].mean()
    print(f"\naggregate smoothness (correlation): {agg:.4f}")
    print("  < 1 -> neighbours are closer in feature space than random pairs,")
    print("         so neighbourhood averaging concentrates signal (GCN is fine)")
    print("  ~ 1 -> averaging would blur rather than sharpen (heterophily-aware)")


if __name__ == "__main__":
    main()
