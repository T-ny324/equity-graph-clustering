"""Tests for the Phase 3b mutual-information graph module.

Synthetic data built inline. Nothing reads data/processed/*.

Run from repo root:  python -m pytest tests/test_mutual_info.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.graphs.mutual_info as mi

CFG = {
    "graph": {"corr_window": 120, "n_factors": 1, "symmetrise": "union",
              "weighted": True},
    "mi": {"bins": 6, "bin_scheme": "quantile", "n_neighbors": 4},
}


def _gaussian_pair(rho: float, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n)
    return s[:, 0], s[:, 1]


def _returns(n: int = 300, n_assets: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    factor = rng.normal(0.0, 1.0, size=n)
    load = rng.uniform(0.6, 1.4, size=n_assets)
    X = np.outer(factor, load) + rng.normal(0.0, 0.6, size=(n, n_assets))
    return pd.DataFrame(X, index=idx, columns=[f"A{i}" for i in range(n_assets)])


# ----------------------------------------------------------------------
# The Gaussian identity
# ----------------------------------------------------------------------
def test_gaussian_mi_known_values() -> None:
    assert mi.gaussian_mi(0.0) == 0.0
    # -0.5 * ln(1 - 0.81) = 0.830366. The spec rounds this to 0.8305.
    assert mi.gaussian_mi(0.9) == pytest.approx(0.830366, abs=1e-6)
    assert mi.gaussian_mi(0.9) == pytest.approx(0.8305, abs=2e-4)
    assert mi.gaussian_mi(-0.9) == pytest.approx(mi.gaussian_mi(0.9))


def test_gaussian_mi_is_finite_at_the_endpoints() -> None:
    assert np.isfinite(mi.gaussian_mi(1.0))
    assert np.isfinite(mi.gaussian_mi(-1.0))


def test_gaussian_mi_is_elementwise_on_a_matrix() -> None:
    C = np.array([[1.0, 0.9], [0.9, 1.0]])
    out = mi.gaussian_mi(C)
    assert out.shape == (2, 2)
    assert out[0, 1] == pytest.approx(0.830366, abs=1e-6)
    assert np.allclose(np.diag(out), mi.gaussian_mi(1.0))


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.9])
def test_ksg_recovers_the_gaussian_curve(rho) -> None:
    """THE MOST IMPORTANT TEST, for the production estimator.

    T=2000 rather than 252, so this isolates estimator correctness from
    small-sample bias -- which validate_estimator measures separately.
    Measured max |error| across seeds is 0.035, so 0.05 is snug, not generous.
    """
    x, y = _gaussian_pair(rho, n=2000, seed=int(rho * 100))
    truth = mi.gaussian_mi(rho)
    est = mi.mi_ksg(x, y)
    assert est == pytest.approx(truth, abs=0.05), (
        f"ksg at rho={rho}: got {est:.4f}, expected {truth:.4f}")


@pytest.mark.parametrize("rho,tol", [(0.0, 0.03), (0.3, 0.04),
                                     (0.6, 0.04), (0.9, 0.10)])
def test_binned_recovers_the_gaussian_curve(rho, tol) -> None:
    """Same curve for the histogram estimator, with bins matched to T.

    bins must scale with sample size: the config default of 6 is chosen for
    T=252 and is far too coarse at T=2000. Even at bins=20 the tolerance has
    to widen at rho=0.9, because discretisation is lossy and the loss is
    worst where the joint density is most concentrated. That ceiling is
    pinned by test_binned_estimator_is_capped_by_its_bin_count below, and it
    is the reason KSG is the production estimator.
    """
    x, y = _gaussian_pair(rho, n=2000, seed=int(rho * 100))
    truth = mi.gaussian_mi(rho)
    est = mi.mi_binned(x, y, bins=20)
    assert est == pytest.approx(truth, abs=tol), (
        f"binned at rho={rho}: got {est:.4f}, expected {truth:.4f}")


def test_binned_estimator_is_capped_by_its_bin_count() -> None:
    """A b-bin discretisation can carry at most log(b) nats.

    MI of a series with itself is its own entropy, which equal-frequency
    binning pins at log(b). This is the structural ceiling that makes the
    histogram estimator undershoot at high rho.
    """
    rng = np.random.default_rng(3)
    z = rng.normal(size=2000)
    for b in (6, 16):
        assert mi.mi_binned(z, z, bins=b) == pytest.approx(np.log(b), abs=0.06)


def test_binned_estimator_improves_with_more_bins_at_high_rho() -> None:
    """At rho=0.9 a coarse grid cannot resolve the mass near the diagonal."""
    x, y = _gaussian_pair(0.9, n=2000, seed=90)
    vals = [mi.mi_binned(x, y, bins=b) for b in (4, 6, 8, 12, 20)]
    assert vals == sorted(vals), f"not monotone in bin count: {vals}"
    # ...and still falls short of the truth even at the best bin count.
    assert vals[-1] < mi.gaussian_mi(0.9)


# ----------------------------------------------------------------------
# Estimator properties
# ----------------------------------------------------------------------
@pytest.mark.parametrize("estimator", ["binned", "ksg"])
def test_estimators_are_non_negative_and_symmetric(estimator) -> None:
    x, y = _gaussian_pair(0.7, n=500, seed=3)
    fn = mi._estimator_fn(estimator)
    a, b = fn(x, y), fn(y, x)
    assert a >= 0.0 and b >= 0.0
    # KSG breaks ties with a tiny noise draw, so symmetry is exact for the
    # histogram estimator and approximate for KSG.
    assert a == pytest.approx(b, abs=1e-6)


@pytest.mark.parametrize("estimator", ["binned", "ksg"])
def test_independent_series_have_a_small_but_non_zero_floor(estimator) -> None:
    """Documents that the floor is NOT zero at realistic sample sizes."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=252)
    y = rng.normal(size=252)
    assert mi._estimator_fn(estimator)(x, y) < 0.15


def test_quantile_binning_is_invariant_to_a_monotone_transform() -> None:
    """Quantile bins depend only on ranks, so exp(y) must not change MI."""
    x, y = _gaussian_pair(0.6, n=1000, seed=11)
    a = mi.mi_binned(x, y, bins=6, scheme="quantile")
    b = mi.mi_binned(x, np.exp(y), bins=6, scheme="quantile")
    assert a == pytest.approx(b, abs=1e-9)


def test_estimators_reject_length_mismatch_and_unknown_names() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        mi.mi_binned(np.zeros(10), np.zeros(11))
    with pytest.raises(ValueError, match="unknown estimator"):
        mi._estimator_fn("kraskov")
    with pytest.raises(ValueError, match="unknown bin scheme"):
        mi.mi_binned(np.zeros(10), np.zeros(10), scheme="log")


# ----------------------------------------------------------------------
# Normalisation
# ----------------------------------------------------------------------
def test_normalise_mi_is_bounded_and_symmetric() -> None:
    R = _returns(n=200, n_assets=6, seed=5).to_numpy()
    M = mi.mi_matrix(R, estimator="binned", n_jobs=1, bins=5)
    H = mi.marginal_entropies(R, bins=5)

    NMI = mi.normalise_mi(M, H)
    assert NMI.min() >= 0.0 and NMI.max() <= 1.0
    assert np.allclose(NMI, NMI.T)


def test_marginal_entropies_are_positive_and_bounded_by_log_bins() -> None:
    R = _returns(n=200, n_assets=6, seed=6).to_numpy()
    H = mi.marginal_entropies(R, bins=6)
    assert H.shape == (6,)
    assert (H > 0).all()
    assert (H <= np.log(6) + 1e-12).all()


# ----------------------------------------------------------------------
# Nonlinear excess
# ----------------------------------------------------------------------
def test_nonlinear_excess_is_near_zero_for_a_gaussian_pair() -> None:
    x, y = _gaussian_pair(0.7, n=4000, seed=13)
    M = np.array([[0.0, mi.mi_ksg(x, y)], [mi.mi_ksg(x, y), 0.0]])
    C = np.array([[1.0, np.corrcoef(x, y)[0, 1]],
                  [np.corrcoef(x, y)[0, 1], 1.0]])
    assert mi.nonlinear_excess(M, C)[0, 1] == pytest.approx(0.0, abs=0.05)


def test_nonlinear_excess_is_clearly_positive_for_a_quadratic_link() -> None:
    """y = x**2 with symmetric x: rho ~ 0, MI clearly > 0.

    This is the test that demonstrates MI captures something correlation
    structurally cannot.
    """
    rng = np.random.default_rng(17)
    x = rng.normal(size=4000)
    y = x ** 2

    rho = float(np.corrcoef(x, y)[0, 1])
    m = mi.mi_ksg(x, y)
    assert abs(rho) < 0.1, f"fixture is not rho-free: {rho:.3f}"
    assert m > 0.3, f"MI failed to see the quadratic link: {m:.3f}"

    M = np.array([[0.0, m], [m, 0.0]])
    C = np.array([[1.0, rho], [rho, 1.0]])
    assert mi.nonlinear_excess(M, C)[0, 1] > 0.3


# ----------------------------------------------------------------------
# Matrices
# ----------------------------------------------------------------------
def test_mi_matrix_is_square_symmetric_with_zero_diagonal() -> None:
    R = _returns(n=150, n_assets=7, seed=19).to_numpy()
    M = mi.mi_matrix(R, estimator="binned", n_jobs=1, bins=5)

    assert M.shape == (7, 7)
    assert np.allclose(M, M.T)
    assert np.all(np.diag(M) == 0.0)
    assert (M >= 0).all()


def test_mi_null_floor_is_strictly_positive() -> None:
    """MI is non-negative, so the null cannot sit at zero."""
    R = _returns(n=252, n_assets=6, seed=23).to_numpy()
    null = mi.mi_null_floor(R, estimator="binned", n_perm=5, n_pairs=10,
                            n_jobs=1, bins=5)
    assert null["n_draws"] == 50
    assert null["mean"] > 0.0
    assert null["p95"] >= null["mean"]
    assert null["p99"] >= null["p95"]


def test_timing_probe_reports_a_projection() -> None:
    R = _returns(n=120, n_assets=10, seed=29).to_numpy()
    p = mi.timing_probe(R, "binned", n_pairs=20, n_dates=5, bins=5)
    assert p["seconds_per_pair"] > 0
    assert p["n_pairs_per_matrix"] == 45
    assert p["seconds_full_run"] == pytest.approx(
        p["seconds_per_pair"] * 45 * 5)


# ----------------------------------------------------------------------
# Sequence and comparison
# ----------------------------------------------------------------------
def test_build_mi_graph_sequence_shapes_and_symmetry(tmp_path) -> None:
    ret = _returns(n=300, n_assets=8, seed=31)
    dates = ret.index[[200, 250]]
    C = np.stack([np.corrcoef(ret.to_numpy()[80:201], rowvar=False),
                  np.corrcoef(ret.to_numpy()[130:251], rowvar=False)])

    A, M, meta = mi.build_mi_graph_sequence(
        ret, dates, C, CFG, k=3, estimator="binned", signed=True,
        cache_dir=tmp_path)

    assert A.shape == (2, 8, 8) and M.shape == (2, 8, 8)
    assert set(meta.columns) == {"mean_mi", "mean_nmi", "mean_delta",
                                 "frac_above_null"}
    for i in range(2):
        assert np.array_equal(A[i], A[i].T)
        assert np.all(np.diag(A[i]) == 0.0)

    # The cache is written and reused.
    assert len(list(tmp_path.glob("mi_binned_*.npy"))) == 2
    A2, M2, _ = mi.build_mi_graph_sequence(
        ret, dates, C, CFG, k=3, estimator="binned", signed=True,
        cache_dir=tmp_path)
    assert np.array_equal(M, M2) and np.array_equal(A, A2)


def test_signed_variant_never_links_a_negative_correlation_pair() -> None:
    ret = _returns(n=300, n_assets=8, seed=37)
    dates = ret.index[[250]]
    C = np.stack([np.corrcoef(ret.to_numpy()[130:251], rowvar=False)])

    A_signed, _, _ = mi.build_mi_graph_sequence(
        ret, dates, C, CFG, k=2, estimator="binned", signed=True)

    neg = (C[0] < 0) & ~np.eye(8, dtype=bool)
    # Union symmetrisation can pull a negative pair back in via the other
    # endpoint, so check the directed selection instead: no negative pair may
    # carry a positive weight.
    assert not np.any(A_signed[0][neg] > 0)


def test_a_binary_records_the_selection_not_the_non_zero_weights() -> None:
    """Sign-masking makes some selected edges carry weight exactly 0.

    A_binary must record what kNN chose, matching Part A's convention, so the
    two artefacts mean the same thing. Deriving it as (A != 0) would silently
    drop those edges and break the controlled comparison.
    """
    ret = _returns(n=300, n_assets=8, seed=53)
    dates = ret.index[[250]]
    C = np.stack([np.corrcoef(ret.to_numpy()[130:251], rowvar=False)])

    A, _, meta = mi.build_mi_graph_sequence(
        ret, dates, C, CFG, k=4, estimator="binned", signed=True)
    A_bin = meta.attrs["A_binary"]

    assert A_bin.shape == A.shape and A_bin.dtype == np.uint8
    # The selection always honours the union floor...
    assert ((A_bin != 0).sum(axis=2) >= 4).all()
    # ...and is a superset of the non-zero weights.
    assert np.all((A != 0) <= (A_bin != 0))


def test_mi_similarity_zeroes_negative_correlation_pairs_when_signed() -> None:
    M = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    C = np.array([[1.0, 0.8, -0.8], [0.8, 1.0, 0.1], [-0.8, 0.1, 1.0]])
    H = np.full(3, np.log(6))

    signed = mi.mi_similarity(M, C, H, signed=True)
    unsigned = mi.mi_similarity(M, C, H, signed=False)

    assert signed[0, 2] == 0.0 and signed[2, 0] == 0.0   # the hedge is cut
    assert signed[0, 1] > 0.0
    assert unsigned[0, 2] > 0.0                          # unsigned keeps it
    assert np.allclose(unsigned, unsigned.T)


def test_compare_graphs_columns_and_self_identity() -> None:
    ret = _returns(n=300, n_assets=8, seed=41)
    dates = ret.index[[200, 250]]
    C = np.stack([np.corrcoef(ret.to_numpy()[80:201], rowvar=False),
                  np.corrcoef(ret.to_numpy()[130:251], rowvar=False)])
    A, _, _ = mi.build_mi_graph_sequence(ret, dates, C, CFG, k=3,
                                         estimator="binned", signed=True)

    comp = mi.compare_graphs(A, A)
    assert set(comp.columns) == {"edge_jaccard", "persistence_corr",
                                 "persistence_mi"}
    assert (comp["edge_jaccard"] == 1.0).all()          # identical to itself
    assert np.isnan(comp["persistence_corr"].iloc[0])

    with pytest.raises(ValueError, match="shape mismatch"):
        mi.compare_graphs(A, A[:1])


# ----------------------------------------------------------------------
# The canary
# ----------------------------------------------------------------------
def test_no_lookahead() -> None:
    """Corrupting the future must not move the MI matrix at t."""
    ret = _returns(n=300, n_assets=8, seed=43)
    pos = 200
    window = CFG["graph"]["corr_window"]
    values = ret.to_numpy(dtype=float)

    Rw = mi.residualise(values[pos + 1 - window:pos + 1], 1)
    M = mi.mi_matrix(Rw, estimator="binned", n_jobs=1, bins=5)

    corrupted = values.copy()
    corrupted[pos + 1:] = 999.0
    Rw2 = mi.residualise(corrupted[pos + 1 - window:pos + 1], 1)
    M2 = mi.mi_matrix(Rw2, estimator="binned", n_jobs=1, bins=5)

    assert np.array_equal(M, M2), "MI at t changed when the future changed"

    # Non-vacuous: corrupting INSIDE the window must move it.
    inside = values.copy()
    inside[pos - 5] = 999.0
    Rw3 = mi.residualise(inside[pos + 1 - window:pos + 1], 1)
    assert not np.array_equal(
        M, mi.mi_matrix(Rw3, estimator="binned", n_jobs=1, bins=5))


def test_no_lookahead_through_the_full_sequence() -> None:
    """Same canary, but through build_mi_graph_sequence end to end."""
    ret = _returns(n=300, n_assets=8, seed=47)
    t = ret.index[200]
    dates = pd.DatetimeIndex([t])
    C = np.stack([np.corrcoef(ret.to_numpy()[80:201], rowvar=False)])

    A, M, _ = mi.build_mi_graph_sequence(ret, dates, C, CFG, k=3,
                                         estimator="binned", signed=True)
    future = ret.copy()
    future.iloc[201:] = 999.0
    A2, M2, _ = mi.build_mi_graph_sequence(future, dates, C, CFG, k=3,
                                           estimator="binned", signed=True)

    assert np.array_equal(M, M2) and np.array_equal(A, A2)
