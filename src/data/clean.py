"""Clean raw OHLCV into a canonical, calendar-aligned panel.

Reads  : data/raw/prices_raw.parquet   (long: date, ticker, o/h/l/c, volume)
Writes : data/processed/panel.parquet  (long, aligned, with returns + flags)
         data/processed/coverage.csv   (per-ticker diagnostics)

Run from repo root:  python -m src.data.clean
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PRICE_FIELDS = ["open", "high", "low", "close"]
ALL_FIELDS = PRICE_FIELDS + ["volume"]


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw long panel and enforce basic invariants."""
    raw = pd.read_parquet(path)
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()

    # Duplicate (date, ticker) rows can appear when download batches overlap.
    # Keep the last occurrence; pivot() will raise otherwise.
    n_before = len(raw)
    raw = raw.drop_duplicates(subset=["date", "ticker"], keep="last")
    if len(raw) < n_before:
        print(f"  dropped {n_before - len(raw):,} duplicate (date, ticker) rows")

    return raw.sort_values(["date", "ticker"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# Reshaping
# ----------------------------------------------------------------------
def to_wide(long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Long -> {field: DataFrame indexed by date, columns by ticker}."""
    return {f: long.pivot(index="date", columns="ticker", values=f)
            for f in ALL_FIELDS}


def to_long(wide: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """{field: wide frame} -> long frame, dropping all-NaN observations."""
    out = pd.concat(
        {f: df.stack(future_stack=True) for f, df in wide.items()},
        axis=1,
    )
    out.index.names = ["date", "ticker"]
    return out.reset_index()


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------
def coverage_stats(close: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker coverage, computed on a wide close frame."""
    valid = close.notna()
    n_sessions = len(close)

    rep = pd.DataFrame({
        "n_obs": valid.sum(),
        "first_date": close.apply(lambda s: s.first_valid_index()),
        "last_date": close.apply(lambda s: s.last_valid_index()),
    })
    rep["frac_sessions"] = rep["n_obs"] / n_sessions

    # Internal holes only: NaNs between first and last valid observation.
    # Distinguishes "listed late" (not a hole) from "gappy" (a hole).
    def _internal_missing(s: pd.Series) -> float:
        first, last = s.first_valid_index(), s.last_valid_index()
        if first is None:
            return 1.0
        span = s.loc[first:last]
        return float(span.isna().mean())

    rep["frac_internal_nan"] = close.apply(_internal_missing)
    return rep.sort_values("frac_sessions")


# ----------------------------------------------------------------------
# Calendar
# ----------------------------------------------------------------------
def build_calendar(close: pd.DataFrame, min_coverage: float) -> pd.DatetimeIndex:
    """Sessions on which at least `min_coverage` of tickers have a price.

    A pure union of dates admits spurious sessions created by a single
    bad series; requiring broad participation removes them.
    """
    participation = close.notna().mean(axis=1)
    keep = participation >= min_coverage
    dropped = (~keep).sum()
    if dropped:
        print(f"  dropped {dropped} low-participation sessions "
              f"(<{min_coverage:.0%} of tickers)")
    return close.index[keep]


# ----------------------------------------------------------------------
# Filtering
# ----------------------------------------------------------------------
def filter_universe(close: pd.DataFrame, min_history: int,
                    max_internal_nan: float) -> list[str]:
    """Return the tickers that pass the history and completeness filters."""
    rep = coverage_stats(close)

    too_short = rep.index[rep["n_obs"] < min_history]
    too_gappy = rep.index[rep["frac_internal_nan"] > max_internal_nan]

    if len(too_short):
        print(f"  dropped {len(too_short)} tickers: history < {min_history} "
              f"sessions -> {sorted(too_short)}")
    if len(too_gappy):
        print(f"  dropped {len(too_gappy)} tickers: internal NaN > "
              f"{max_internal_nan:.1%} -> {sorted(too_gappy)}")

    return sorted(set(close.columns) - set(too_short) - set(too_gappy))


def filter_full_history(close: pd.DataFrame, tolerance_days: int) -> list[str]:
    """Return the tickers already trading at the start of the sample.

    A name that lists mid-sample passes the history-length filter yet leaves
    a NaN block at the front of the panel, which makes N time-varying and
    yields structurally NaN correlation matrices downstream.
    """
    cutoff = close.index[0] + pd.Timedelta(days=tolerance_days)
    first = close.apply(lambda s: s.first_valid_index())
    keep = first.notna() & (first <= cutoff)
    return sorted(close.columns[keep])


# ----------------------------------------------------------------------
# Filling
# ----------------------------------------------------------------------
def fill_prices(wide: dict[str, pd.DataFrame],
                max_ffill: int) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Forward-fill price fields with a hard limit; flag what was filled.

    Volume is deliberately NOT filled: a stale price implies no trade, and
    a fabricated volume would corrupt liquidity features downstream.
    """
    close = wide["close"]

    # A cell is "filled" if it was NaN and ffill supplied a value.
    was_nan = close.isna()
    filled_mask = was_nan & close.ffill(limit=max_ffill).notna()

    out = {}
    for f in PRICE_FIELDS:
        out[f] = wide[f].ffill(limit=max_ffill)

    # Volume: NaN wherever the price was carried forward.
    vol = wide["volume"].copy()
    vol = vol.mask(filled_mask)
    # Zero volume is a data artefact for large caps and breaks Amihud
    # (division by zero); treat it as missing rather than as a real zero.
    vol = vol.mask(vol <= 0)
    out["volume"] = vol

    n_filled = int(filled_mask.to_numpy().sum())
    n_cells = int(close.size)
    print(f"  forward-filled {n_filled:,} price cells "
          f"({n_filled / n_cells:.3%} of panel, limit {max_ffill}d)")

    return out, filled_mask


# ----------------------------------------------------------------------
# Derived series
# ----------------------------------------------------------------------
def add_derived(wide: dict[str, pd.DataFrame],
                filled_mask: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Log returns, dollar volume, and the is_filled flag."""
    close = wide["close"]

    log_ret = np.log(close).diff()
    # A return spanning a filled gap is mechanically zero and is not a
    # real observation. Flag it; do not silently keep it as information.
    log_ret = log_ret.mask(filled_mask)

    wide = dict(wide)
    wide["log_ret"] = log_ret
    wide["dollar_volume"] = close * wide["volume"]
    wide["is_filled"] = filled_mask.astype(float)
    return wide


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def build_panel(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw long frame -> (canonical panel, coverage report)."""
    ucfg, dcfg = cfg["universe"], cfg["data"]

    print("[1] reshaping to wide")
    wide = to_wide(raw)

    # --- Pass 1: drop clearly broken tickers BEFORE building the calendar,
    #     so that a single bad series cannot define trading sessions.
    print("[2] first-pass universe filter (pre-calendar)")
    keep = filter_universe(
        wide["close"],
        min_history=ucfg["min_history_days"],
        max_internal_nan=ucfg["max_missing_frac"],
    )
    wide = {f: df[keep] for f, df in wide.items()}

    print("[3] building trading calendar")
    calendar = build_calendar(wide["close"], ucfg["min_session_coverage"])
    wide = {f: df.reindex(calendar) for f, df in wide.items()}

    # --- Pass 2: re-filter on the fixed calendar. Reindexing can expose
    #     holes that were invisible on a ticker's own sparse index.
    print("[4] second-pass universe filter (post-calendar)")
    keep = filter_universe(
        wide["close"],
        min_history=ucfg["min_history_days"],
        max_internal_nan=ucfg["max_missing_frac"],
    )
    wide = {f: df[keep] for f, df in wide.items()}

    # --- Fixed-N: a name that lists mid-sample passes the filters above but
    #     still leaves a NaN block at the front of the panel.
    if ucfg.get("require_full_history", False):
        tol = ucfg["start_tolerance_days"]
        keep = filter_full_history(wide["close"], tolerance_days=tol)
        dropped = sorted(set(wide["close"].columns) - set(keep))
        if dropped:
            first = wide["close"][dropped].apply(lambda s: s.first_valid_index())
            print(f"  dropped {len(dropped)} tickers: not trading within "
                  f"{tol}d of {wide['close'].index[0].date()} -> {dropped}")
            for t in dropped:
                d = first[t]
                print(f"    {t:<6} first valid {d.date() if d is not None else 'never'}")
        wide = {f: df[keep] for f, df in wide.items()}

    print("[5] filling")
    wide, filled_mask = fill_prices(wide, max_ffill=ucfg["max_ffill_days"])

    # The fixed-N guarantee Phase 3 depends on: a complete close rectangle,
    # so no correlation window can be structurally NaN.
    if ucfg.get("require_full_history", False):
        holes = wide["close"].columns[wide["close"].isna().any()]
        assert not len(holes), (
            f"close still has NaNs after filtering and filling for "
            f"{len(holes)} tickers: {sorted(holes)}"
        )

    print("[6] derived series")
    wide = add_derived(wide, filled_mask)

    print("[7] back to long")
    panel = to_long(wide)
    panel = panel.dropna(subset=["close"])
    panel["is_filled"] = panel["is_filled"].astype(bool)

    report = coverage_stats(wide["close"])
    report["max_abs_ret"] = wide["log_ret"].abs().max()
    report["n_filled"] = filled_mask.sum()

    print(f"\nfinal panel: {len(panel):,} rows, "
          f"{panel['ticker'].nunique()} tickers, "
          f"{panel['date'].nunique()} sessions, "
          f"{panel['date'].min().date()} to {panel['date'].max().date()}")

    return panel, report


def main() -> None:
    cfg = yaml.safe_load(open("config/base.yaml"))
    raw_path = Path(cfg["data"]["raw_dir"]) / "prices_raw.parquet"
    out_dir = Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {raw_path}")
    raw = load_raw(raw_path)

    panel, report = build_panel(raw, cfg)

    panel.to_parquet(out_dir / "panel.parquet", index=False)
    report.to_csv(out_dir / "coverage.csv")
    print(f"\nwrote {out_dir/'panel.parquet'} and {out_dir/'coverage.csv'}")

    print("\nworst coverage:")
    print(report.head(8).to_string())
    print("\nlargest absolute daily log returns:")
    print(report.nlargest(8, "max_abs_ret")[["max_abs_ret"]].to_string())


if __name__ == "__main__":
    main()