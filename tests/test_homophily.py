"""Tests for the Phase 3c homophily and smoothness diagnostics.

Synthetic data built inline. Nothing reads data/processed/*.

Run from repo root:  python -m pytest tests/test_homophily.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.graphs.homophily as hp


def _blocks(sizes: list[int]) -> np.ndarray:
    """Labels for consecutive equal-label blocks, e.g. [2,3] -> AABBB."""
    return np.concatenate([[chr(65 + i)] * n for i, n in enumerate(sizes)])


def _same_label_graph(labels: np.ndarray) -> np.ndarray:
    """Every within-label pair connected, nothing across labels."""
    same = labels[:, None] == labels[None, :]
    A = same.astype(float)
    np.fill_diagonal(A, 0.0)
    return A


def _cross_label_graph(labels: np.ndarray) -> np.ndarray:
    """Every across-label pair connected, nothing within."""
    A = (labels[:, None] != labels[None, :]).astype(float)
    np.fill_diagonal(A, 0.0)
    return A


# ----------------------------------------------------------------------
# edge_homophily
# ----------------------------------------------------------------------
def test_perfect_homophily_gives_adjusted_one() -> None:
    labels = _blocks([5, 5])
    h = hp.edge_homophily(_same_label_graph(labels), labels)
    assert h["raw"] == pytest.approx(1.0)
    assert h["adjusted"] == pytest.approx(1.0)


def test_perfect_heterophily_gives_negative_adjusted() -> None:
    labels = _blocks([5, 5])
    h = hp.edge_homophily(_cross_label_graph(labels), labels)
    assert h["raw"] == pytest.approx(0.0)
    assert h["adjusted"] < 0


def test_random_wiring_two_equal_classes_is_near_zero() -> None:
    rng = np.random.default_rng(0)
    labels = _blocks([100, 100])
    A = (rng.random((200, 200)) < 0.1).astype(float)
    A = np.triu(A, 1); A = A + A.T

    h = hp.edge_homophily(A, labels)
    assert h["expected"] == pytest.approx(0.5)
    assert abs(h["adjusted"]) < 0.05


def test_expected_is_sum_of_squared_proportions_not_one_over_k() -> None:
    """The whole reason the raw fraction alone is uninterpretable."""
    labels = _blocks([80, 15, 5])                # deliberately lopsided
    h = hp.edge_homophily(_same_label_graph(labels), labels)

    expected = (0.8 ** 2) + (0.15 ** 2) + (0.05 ** 2)
    assert h["expected"] == pytest.approx(expected)
    assert h["expected"] == pytest.approx(0.665)
    assert h["expected"] != pytest.approx(1 / 3)   # 1/n_classes would be 0.333


def test_homophily_is_nan_on_an_empty_graph() -> None:
    labels = _blocks([3, 3])
    h = hp.edge_homophily(np.zeros((6, 6)), labels)
    assert np.isnan(h["raw"]) and np.isnan(h["adjusted"])


# ----------------------------------------------------------------------
# assortativity
# ----------------------------------------------------------------------
def test_assortativity_is_one_on_a_perfectly_homophilous_graph() -> None:
    labels = _blocks([6, 6])
    assert hp.assortativity(_same_label_graph(labels), labels) == pytest.approx(1.0)


def test_assortativity_is_negative_on_a_heterophilous_graph() -> None:
    labels = _blocks([6, 6])
    assert hp.assortativity(_cross_label_graph(labels), labels) < 0


def test_assortativity_is_nan_without_edges() -> None:
    assert np.isnan(hp.assortativity(np.zeros((5, 5)), _blocks([2, 3])))


# ----------------------------------------------------------------------
# feature_smoothness
# ----------------------------------------------------------------------
def _two_cluster_features(n_per: int = 10, sep: float = 20.0,
                          seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Two tight, well-separated clusters in feature space."""
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0.0, 0.1, size=(n_per, 3)),
                   rng.normal(sep, 0.1, size=(n_per, 3))])
    labels = _blocks([n_per, n_per])
    return X, labels


def test_smoothness_well_below_one_when_edges_join_similar_nodes() -> None:
    X, labels = _two_cluster_features()
    A = _same_label_graph(labels)            # edges only within a cluster
    assert hp.feature_smoothness(A, X) < 0.1


def test_smoothness_above_one_when_edges_join_dissimilar_nodes() -> None:
    X, labels = _two_cluster_features()
    A = _cross_label_graph(labels)           # edges only across clusters
    assert hp.feature_smoothness(A, X) > 1.0


def test_smoothness_is_about_one_for_random_edges() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 4))
    A = (rng.random((60, 60)) < 0.15).astype(float)
    A = np.triu(A, 1); A = A + A.T
    assert hp.feature_smoothness(A, X) == pytest.approx(1.0, abs=0.25)


def test_smoothness_is_nan_on_an_empty_graph() -> None:
    X, _ = _two_cluster_features()
    assert np.isnan(hp.feature_smoothness(np.zeros((20, 20)), X))


# ----------------------------------------------------------------------
# per_feature_smoothness
# ----------------------------------------------------------------------
def test_per_feature_separates_a_smooth_feature_from_a_noisy_one() -> None:
    """Column 0 respects the clusters; column 1 is pure noise."""
    rng = np.random.default_rng(5)
    n = 12
    labels = _blocks([n, n])
    smooth = np.concatenate([np.zeros(n), np.full(n, 10.0)])
    noise = rng.normal(size=2 * n)
    X = np.column_stack([smooth, noise])

    s = hp.per_feature_smoothness(_same_label_graph(labels), X,
                                  ["smooth", "noise"])
    assert list(s.index) == ["smooth", "noise"]
    assert s["smooth"] < 0.05                    # constant within a cluster
    assert s["noise"] == pytest.approx(1.0, abs=0.4)
    assert s["smooth"] < s["noise"]


# ----------------------------------------------------------------------
# load_sector_labels
# ----------------------------------------------------------------------
def test_load_sector_labels_orders_by_the_requested_tickers(tmp_path) -> None:
    csv = tmp_path / "u.csv"
    pd.DataFrame({"yahoo_ticker": ["AAA", "BBB", "CCC"],
                  "sector": ["Tech", "Energy", "Health"]}).to_csv(csv, index=False)

    out = hp.load_sector_labels(csv, ["CCC", "AAA"])
    assert list(out) == ["Health", "Tech"]


def test_load_sector_labels_names_the_missing_ticker(tmp_path) -> None:
    csv = tmp_path / "u.csv"
    pd.DataFrame({"yahoo_ticker": ["AAA", "BBB"],
                  "sector": ["Tech", "Energy"]}).to_csv(csv, index=False)

    with pytest.raises(AssertionError, match="ZZZ"):
        hp.load_sector_labels(csv, ["AAA", "ZZZ"])


def test_load_sector_labels_normalises_the_dot_to_dash_symbol(tmp_path) -> None:
    """The CSV stores BRK.B; every artefact downstream carries BRK-B."""
    csv = tmp_path / "u.csv"
    pd.DataFrame({"yahoo_ticker": ["AAA", "BRK.B"],
                  "sector": ["Tech", "Financials"]}).to_csv(csv, index=False)

    assert list(hp.load_sector_labels(csv, ["AAA", "BRK-B"])) == ["Tech", "Financials"]


# ----------------------------------------------------------------------
# analyse
# ----------------------------------------------------------------------
def test_analyse_returns_a_row_per_date_and_a_value_per_feature() -> None:
    X, labels = _two_cluster_features()
    A = _same_label_graph(labels)
    stack = np.stack([A, A, A])
    Xs = np.stack([X, X, X])
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")

    per_date, per_feat = hp.analyse(stack, Xs, labels, dates,
                                    ["f0", "f1", "f2"], "test")
    assert len(per_date) == 3
    assert (per_date["graph"] == "test").all()
    assert set(per_date.columns) == {"graph", "homophily_raw",
                                     "homophily_expected", "homophily_adjusted",
                                     "assortativity", "smoothness"}
    assert list(per_feat.index) == ["f0", "f1", "f2"]
    assert per_date["homophily_adjusted"].iloc[0] == pytest.approx(1.0)
