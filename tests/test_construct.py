"""Tests for the Phase 3a correlation kNN graph construction.

Everything is built from small synthetic frames. Nothing reads
data/processed/*: a test that reads real data is a description of today's
dataset, not a test of the code.

Run from repo root:  python -m pytest tests/test_construct.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.graphs.construct as gc

CFG = {
    "corr_window": 120,
    "n_factors": 1,
    "shrinkage": "ledoit_wolf",
    "similarity": "signed",
    "symmetrise": "union",
    "weighted": True,
}


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
def _factor_returns(n: int = 400, n_assets: int = 12, seed: int = 0,
                    noise: float = 0.15) -> pd.DataFrame:
    """A single common factor plus small idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    factor = rng.normal(0.0, 1.0, size=n)
    loadings = rng.uniform(0.7, 1.3, size=n_assets)
    X = np.outer(factor, loadings) + rng.normal(0.0, noise, size=(n, n_assets))
    return pd.DataFrame(X, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def _mean_abs_offdiag_corr(X: np.ndarray) -> float:
    C = np.corrcoef(X, rowvar=False)
    iu = np.triu_indices(C.shape[0], k=1)
    return float(np.abs(C[iu]).mean())


# ----------------------------------------------------------------------
# knn_adjacency
# ----------------------------------------------------------------------
def test_knn_selects_the_known_nearest_neighbours() -> None:
    """Hand-built similarity with an unambiguous ordering."""
    S = np.array([
        [1.0, 0.9, 0.8, 0.1],
        [0.9, 1.0, 0.7, 0.2],
        [0.8, 0.7, 1.0, 0.3],
        [0.1, 0.2, 0.3, 1.0],
    ])
    # k=1, intersection: only mutual top choices survive. 0 and 1 pick each
    # other; 2 picks 0 but 0 picks 1; 3 picks 2 but 2 picks 0.
    M = gc.knn_adjacency(S, k=1, symmetrise="intersection", weighted=False)
    assert M[0, 1] == 1 and M[1, 0] == 1
    assert M[2, 3] == 0 and M[0, 3] == 0
    assert M.sum() == 2

    # k=1, union: every node keeps its own top pick.
    U = gc.knn_adjacency(S, k=1, symmetrise="union", weighted=False)
    assert U[0, 1] == 1 and U[2, 0] == 1 and U[3, 2] == 1

    # Weighted output carries the similarity of the surviving edges.
    W = gc.knn_adjacency(S, k=1, symmetrise="union", weighted=True)
    assert W[0, 1] == pytest.approx(0.9)
    assert W[2, 0] == pytest.approx(0.8)


def test_knn_output_is_symmetric_with_zero_diagonal() -> None:
    rng = np.random.default_rng(1)
    S = rng.normal(size=(30, 30))
    S = (S + S.T) / 2
    np.fill_diagonal(S, 1.0)

    for sym in ("union", "intersection"):
        for weighted in (True, False):
            A = gc.knn_adjacency(S, k=4, symmetrise=sym, weighted=weighted)
            assert np.array_equal(A, A.T), f"{sym}/{weighted} not symmetric"
            assert np.all(np.diag(A) == 0.0), f"{sym}/{weighted} diagonal non-zero"


def test_union_gives_every_node_degree_at_least_k() -> None:
    rng = np.random.default_rng(2)
    S = rng.normal(size=(40, 40))
    S = (S + S.T) / 2
    k = 5
    A = gc.knn_adjacency(S, k=k, symmetrise="union", weighted=False)
    assert (A.sum(axis=1) >= k).all()


def test_intersection_gives_every_node_degree_at_most_k() -> None:
    rng = np.random.default_rng(3)
    S = rng.normal(size=(40, 40))
    S = (S + S.T) / 2
    k = 5
    A = gc.knn_adjacency(S, k=k, symmetrise="intersection", weighted=False)
    assert (A.sum(axis=1) <= k).all()


def test_knn_rejects_k_outside_the_valid_range() -> None:
    S = np.eye(5)
    with pytest.raises(ValueError, match="k must satisfy"):
        gc.knn_adjacency(S, k=5)
    with pytest.raises(ValueError, match="k must satisfy"):
        gc.knn_adjacency(S, k=0)


# ----------------------------------------------------------------------
# residualise
# ----------------------------------------------------------------------
def test_residualise_removes_the_market_mode() -> None:
    """Off-diagonal correlation must collapse once the common factor goes."""
    R = _factor_returns().to_numpy()

    before = _mean_abs_offdiag_corr(R)
    after = _mean_abs_offdiag_corr(gc.residualise(R, n_factors=1))

    assert before > 0.9, f"fixture is not factor-dominated: {before:.3f}"
    assert after < before / 3, f"market mode survived: {before:.3f} -> {after:.3f}"


def test_residualise_preserves_shape_and_is_a_noop_for_zero_factors() -> None:
    R = _factor_returns(n=50, n_assets=6).to_numpy()
    assert gc.residualise(R, n_factors=1).shape == R.shape
    assert np.allclose(gc.residualise(R, n_factors=0), R)


# ----------------------------------------------------------------------
# shrunk_correlation
# ----------------------------------------------------------------------
@pytest.mark.parametrize("method", ["ledoit_wolf", "sample"])
def test_shrunk_correlation_is_a_valid_correlation_matrix(method) -> None:
    R = _factor_returns(n=150, n_assets=20, seed=5).to_numpy()
    C, intensity = gc.shrunk_correlation(R, method=method)

    assert np.allclose(C, C.T, atol=1e-12)
    assert np.allclose(np.diag(C), 1.0, atol=1e-12)
    assert np.linalg.eigvalsh(C).min() >= -1e-8, "not positive semi-definite"
    assert 0.0 <= intensity <= 1.0


def test_ledoit_wolf_shrinkage_is_strictly_positive_when_n_rivals_t() -> None:
    """The regime this module operates in: q = N/T is large."""
    R = _factor_returns(n=60, n_assets=40, seed=6).to_numpy()
    _, intensity = gc.shrunk_correlation(R, method="ledoit_wolf")
    assert intensity > 0.0


def test_shrunk_correlation_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown shrinkage method"):
        gc.shrunk_correlation(_factor_returns(n=50, n_assets=5).to_numpy(), "oas")


# ----------------------------------------------------------------------
# edge_persistence
# ----------------------------------------------------------------------
def test_edge_persistence_identical_and_disjoint() -> None:
    A = np.zeros((6, 6))
    for i, j in [(0, 1), (1, 2), (2, 3), (4, 5)]:
        A[i, j] = A[j, i] = 1.0
    assert gc.edge_persistence(A, A) == 1.0

    B = np.zeros((6, 6))
    for i, j in [(0, 2), (1, 3), (0, 4), (3, 5)]:
        B[i, j] = B[j, i] = 1.0
    assert gc.edge_persistence(A, B) == 0.0


def test_edge_persistence_half_shared_is_one_third() -> None:
    """Jaccard, not overlap fraction: 2 shared of 6 distinct is 1/3, not 0.5."""
    A = np.zeros((8, 8))
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        A[i, j] = A[j, i] = 1.0
    B = np.zeros((8, 8))
    for i, j in [(0, 1), (1, 2), (5, 6), (6, 7)]:      # 2 of 4 shared
        B[i, j] = B[j, i] = 1.0

    # |A & B| = 2, |A | B| = 6  ->  1/3
    assert gc.edge_persistence(A, B) == pytest.approx(1.0 / 3.0)


def test_edge_persistence_empty_union_is_zero() -> None:
    Z = np.zeros((5, 5))
    assert gc.edge_persistence(Z, Z) == 0.0


# ----------------------------------------------------------------------
# marchenko_pastur_edge
# ----------------------------------------------------------------------
def test_marchenko_pastur_edge_known_values() -> None:
    assert gc.marchenko_pastur_edge(100, 100) == pytest.approx(4.0)

    exact = (1.0 + np.sqrt(105 / 252)) ** 2
    assert gc.marchenko_pastur_edge(105, 252) == pytest.approx(exact)
    # The spec quotes ~2.75 for q = 0.42; the exact value is 2.716.
    assert gc.marchenko_pastur_edge(105, 252) == pytest.approx(2.75, abs=0.05)


# ----------------------------------------------------------------------
# Sequence and diagnostics
# ----------------------------------------------------------------------
def test_build_graph_sequence_shapes_and_symmetry() -> None:
    ret = _factor_returns(n=300, n_assets=15, seed=8)
    dates = ret.index[[150, 200, 250, 299]]

    A, C, meta = gc.build_graph_sequence(ret, dates, CFG, k=3)

    assert A.shape == (4, 15, 15) and C.shape == (4, 15, 15)
    assert list(meta.index) == list(dates)
    assert set(meta.columns) == {"shrinkage", "mean_corr", "top_eigval_frac",
                                 "n_eigs_above_mp"}
    for i in range(len(dates)):
        assert np.array_equal(A[i], A[i].T)
        assert np.all(np.diag(A[i]) == 0.0)
        assert np.allclose(np.diag(C[i]), 1.0, atol=1e-6)


def test_build_graph_sequence_rejects_insufficient_history() -> None:
    ret = _factor_returns(n=300, n_assets=10, seed=9)
    with pytest.raises(ValueError, match="more sessions of history"):
        gc.build_graph_sequence(ret, ret.index[[10]], CFG, k=3)


def test_graph_diagnostics_columns_and_degrees() -> None:
    ret = _factor_returns(n=300, n_assets=15, seed=10)
    dates = ret.index[[150, 200, 250]]
    A, _, _ = gc.build_graph_sequence(ret, dates, CFG, k=4)

    diag = gc.graph_diagnostics(A)
    assert len(diag) == 3
    assert set(diag.columns) == {"n_edges", "density", "mean_degree",
                                 "min_degree", "max_degree", "degree_gini",
                                 "n_components", "is_connected"}
    assert (diag["min_degree"] >= 4).all()          # union floor
    assert (diag["degree_gini"] >= 0).all()


def test_gini_is_zero_for_a_uniform_degree_distribution() -> None:
    assert gc._gini(np.full(10, 7.0)) == pytest.approx(0.0, abs=1e-12)
    assert gc._gini(np.array([0.0, 0.0, 0.0, 10.0])) > 0.5


def test_sweep_k_is_monotone_in_degree() -> None:
    ret = _factor_returns(n=300, n_assets=20, seed=11)
    dates = ret.index[[150, 180, 210, 240, 270]]

    sweep = gc.sweep_k(ret, dates, CFG, [2, 4, 6])
    assert list(sweep["k"]) == [2, 4, 6]
    assert sweep["mean_degree"].is_monotonic_increasing
    assert sweep["density"].is_monotonic_increasing
    assert (sweep["mean_degree"] >= sweep["k"]).all()


# ----------------------------------------------------------------------
# The canary
# ----------------------------------------------------------------------
def test_no_lookahead() -> None:
    """THE MOST IMPORTANT TEST.

    Compute the graph at a rebalance date partway through the sample, then
    corrupt every return strictly AFTER that date and recompute. The graph at
    that date must be bitwise identical.
    """
    ret = _factor_returns(n=400, n_assets=15, seed=12)
    t = ret.index[250]
    dates = pd.DatetimeIndex([t])

    A, C, meta = gc.build_graph_sequence(ret, dates, CFG, k=4)

    corrupted = ret.copy()
    corrupted.iloc[251:] = 99.0
    A2, C2, meta2 = gc.build_graph_sequence(corrupted, dates, CFG, k=4)

    assert np.array_equal(A, A2), "adjacency at t changed when the future changed"
    assert np.array_equal(C, C2), "correlation at t changed when the future changed"
    assert meta["shrinkage"].iloc[0] == meta2["shrinkage"].iloc[0]

    # Non-vacuous: corrupting INSIDE the window must change the graph.
    inside = ret.copy()
    inside.iloc[249] = 99.0
    A3, _, _ = gc.build_graph_sequence(inside, dates, CFG, k=4)
    assert not np.array_equal(A, A3), "canary is vacuous: window never used"


def test_window_is_right_aligned_and_inclusive_of_t() -> None:
    """The window is exactly the corr_window sessions ending at t."""
    ret = _factor_returns(n=400, n_assets=12, seed=13)
    pos = 300
    t = ret.index[pos]
    w = CFG["corr_window"]

    full, _, _ = gc.build_graph_sequence(ret, pd.DatetimeIndex([t]), CFG, k=3)
    # Truncating everything after t must leave the graph at t unchanged...
    trimmed, _, _ = gc.build_graph_sequence(
        ret.iloc[:pos + 1], pd.DatetimeIndex([t]), CFG, k=3)
    assert np.array_equal(full, trimmed)

    # ...and so must dropping the sessions before the window starts.
    exact, _, _ = gc.build_graph_sequence(
        ret.iloc[pos + 1 - w:pos + 1], pd.DatetimeIndex([t]), CFG, k=3)
    assert np.array_equal(full, exact)
