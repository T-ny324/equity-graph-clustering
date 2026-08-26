"""Regression tests for the Phase 1 universe fixes.

Run from repo root:  python -m pytest tests/test_universe.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.clean import filter_full_history
from src.data.ingest import load_universe

RENAMED = ["BNY", "FISV"]           # resolved via the CSV's yahoo_ticker
GONE = ["BK", "FI", "DFS", "HES", "IPG"]   # renamed-away or delisted


def test_load_universe_resolves_renames_and_drops_delistings(tmp_path) -> None:
    """The CSV pins symbol resolution; nothing is delegated to Yahoo."""
    csv = tmp_path / "universe.csv"
    pd.DataFrame({
        "ticker": ["AAPL", "BK", "FI", "DFS", "HES", "IPG", "BRK.B"],
        "name": ["Apple", "BNY Mellon", "Fiserv", "Discover", "Hess",
                 "Interpublic", "Berkshire"],
        "sector": ["Information Technology", "Financials", "Financials",
                   "Financials", "Energy", "Communication Services",
                   "Financials"],
        "yahoo_ticker": ["AAPL", "BNY", "FISV", "DFS", "HES", "IPG", "BRK.B"],
        "status": ["active", "renamed", "renamed", "delisted", "delisted",
                   "delisted", "active"],
        "notes": ["", "BK->BNY", "FI->FISV", "merged", "acquired", "merged", ""],
    }).to_csv(csv, index=False)

    tickers = load_universe(csv)

    assert isinstance(tickers, list)
    for t in RENAMED:
        assert t in tickers
    for t in GONE:
        assert t not in tickers
    assert "BRK-B" in tickers          # dot-to-dash normalisation survives


def _synthetic_close() -> pd.DataFrame:
    """Three names on one 200-session index, starting on days 0, 3 and 60."""
    idx = pd.date_range("2015-01-01", periods=200, freq="D")
    close = pd.DataFrame(100.0, index=idx, columns=["DAY0", "DAY3", "DAY60"])
    close.loc[idx[:3], "DAY3"] = np.nan
    close.loc[idx[:60], "DAY60"] = np.nan
    return close


def test_filter_full_history_drops_late_listings() -> None:
    keep = filter_full_history(_synthetic_close(), tolerance_days=7)
    assert keep == ["DAY0", "DAY3"]


@pytest.mark.parametrize("tolerance,expected", [
    (0, ["DAY0"]),
    (3, ["DAY0", "DAY3"]),
    (60, ["DAY0", "DAY3", "DAY60"]),
])
def test_filter_full_history_respects_tolerance(tolerance, expected) -> None:
    assert filter_full_history(_synthetic_close(), tolerance) == expected


def test_filter_full_history_drops_all_nan_column() -> None:
    close = _synthetic_close()
    close["EMPTY"] = np.nan
    assert "EMPTY" not in filter_full_history(close, tolerance_days=7)
