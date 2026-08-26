"""Download and cache raw daily OHLCV for the universe.

Writes : data/raw/prices_raw.parquet   (long: date, ticker, o/h/l/c, volume)
         data/raw/_batches/*.parquet   (per-batch cache, enables resume)

This module  downloads, reshapes, and checks that
the *download* succeeded. All data-quality judgement lives in clean.py.

Run from repo root:  python -m src.data.ingest
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

FIELDS = ["open", "high", "low", "close", "volume"]


# ----------------------------------------------------------------------
# Reshaping
# ----------------------------------------------------------------------
def _stack_compat(df: pd.DataFrame, level: int) -> pd.DataFrame:
    """pandas >= 2.1 changed stack()'s NaN handling; keep both working."""
    try:
        return df.stack(level=level, future_stack=True)
    except TypeError:  # pandas < 2.1
        return df.stack(level=level, dropna=False)


def reshape(df: pd.DataFrame, batch: list[str]) -> pd.DataFrame:
    """Wide (date x [ticker, field]) -> long (date, ticker, o/h/l/c, volume).

    yfinance returns a flat column index for a single ticker and a
    MultiIndex for several, so normalise the single case first.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        df = pd.concat({batch[0]: df}, axis=1)

    df.columns.names = ["ticker", "field"]
    out = _stack_compat(df, level=0).reset_index()
    out.columns = [str(c).lower() for c in out.columns]

    missing = [f for f in FIELDS if f not in out.columns]
    if missing:
        raise ValueError(f"missing fields after reshape: {missing}")

    return out[["date", "ticker"] + FIELDS]


# ----------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------
def _download_batch(batch: list[str], start: str, end: str | None,
                    retries: int, pause: float) -> pd.DataFrame | None:
    """Download one batch with retries. Returns None if it never succeeds."""
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(batch, start=start, end=end,
                              auto_adjust=True, progress=False,
                              group_by="ticker", threads=True)
        except Exception as exc:               # network / parse failures
            print(f"    attempt {attempt} raised: {exc}")
            raw = None

        # A throttled request returns an EMPTY FRAME, not an exception.
        # This is the failure mode that silently produces missing tickers.
        if raw is not None and not raw.empty:
            return reshape(raw, batch)

        if attempt < retries:
            backoff = pause * (2 ** attempt)
            print(f"    attempt {attempt} empty; retrying in {backoff:.0f}s")
            time.sleep(backoff)

    return None


def download_raw(tickers: list[str], start: str, end: str | None,
                 cache_dir: Path, batch_size: int = 25,
                 pause: float = 1.0, retries: int = 3,
                 use_cache: bool = True) -> pd.DataFrame:
    """Download in batches, caching each batch so a run can be resumed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames, failed = [], []

    batches = [tickers[i:i + batch_size]
               for i in range(0, len(tickers), batch_size)]

    for i, batch in enumerate(batches, start=1):
        cache_file = cache_dir / f"batch_{i:03d}.parquet"

        if use_cache and cache_file.exists():
            print(f"  batch {i}/{len(batches)}: cached")
            frames.append(pd.read_parquet(cache_file))
            continue

        print(f"  batch {i}/{len(batches)}: downloading {len(batch)} tickers")
        df = _download_batch(batch, start, end, retries, pause)

        if df is None:
            print(f"    FAILED after {retries} attempts: {batch}")
            failed.extend(batch)
            continue

        df.to_parquet(cache_file, index=False)
        frames.append(df)
        time.sleep(pause)

    if failed:
        print(f"\n{len(failed)} tickers in failed batches: {sorted(failed)}")
    if not frames:
        raise RuntimeError("no data downloaded at all — check network/symbols")

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.tz_localize(None).dt.normalize()
    panel = panel.dropna(subset=["close"])

    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# Download validation (NOT data quality — see clean.py for that)
# ----------------------------------------------------------------------
def download_report(panel: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
    """Did we get, for each requested ticker, a plausible response?"""
    returned = set(panel["ticker"].unique())
    absent = sorted(set(requested) - returned)
    unexpected = sorted(returned - set(requested))

    if absent:
        print(f"\nNO DATA RETURNED ({len(absent)}): {absent}")
        print("  -> usually a bad symbol or a dot/dash convention mismatch")
    if unexpected:
        print(f"\nUNEXPECTED TICKERS ({len(unexpected)}): {unexpected}")

    rep = panel.groupby("ticker").agg(
        n_rows=("date", "size"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        n_zero_volume=("volume", lambda s: int((s.fillna(0) <= 0).sum())),
    )

    modal_last = rep["last_date"].mode().iloc[0]
    rep["ends_early"] = rep["last_date"] < modal_last - pd.Timedelta(days=7)

    return rep.sort_values("n_rows")


def load_universe(path: str | Path) -> list[str]:
    """Read the committed universe CSV and resolve the symbols to request.

    The CSV is the single source of truth for symbol resolution: the
    `yahoo_ticker` column pins renames (BK -> BNY) and the `status` column
    marks names delisted by M&A. Delisted rows are kept in the file as the
    documented record of what was excluded, but are never requested.
    """
    uni = pd.read_csv(path).fillna({"status": "active", "notes": ""})
    status = uni["status"].astype(str).str.strip().str.lower()

    live = uni[status != "delisted"]
    tickers = (live["yahoo_ticker"].astype(str)
               .str.strip()
               .str.upper()
               .str.replace(".", "-", regex=False))   # BRK.B -> BRK-B

    n_renamed = int((status == "renamed").sum())
    n_delisted = int((status == "delisted").sum())
    print(f"universe csv: {len(uni)} rows -> "
          f"{len(live)} requested "
          f"({len(live) - n_renamed} active, {n_renamed} renamed), "
          f"{n_delisted} excluded as delisted")

    return sorted(tickers.unique())


# ----------------------------------------------------------------------
def main() -> None:
    cfg = yaml.safe_load(open("config/base.yaml"))
    dcfg = cfg["data"]

    raw_dir = Path(dcfg["raw_dir"])
    tickers = load_universe("data/universe_sp100.csv")
    print(f"universe: {len(tickers)} tickers")

    panel = download_raw(
        tickers,
        start=dcfg["start"],
        end=dcfg["end"],
        cache_dir=raw_dir / "_batches",
        batch_size=dcfg.get("batch_size", 25),
        pause=dcfg.get("pause", 1.0),
    )

    out = raw_dir / "prices_raw.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, index=False)

    print(f"\nwrote {len(panel):,} rows, "
          f"{panel['ticker'].nunique()} tickers -> {out}")

    rep = download_report(panel, tickers)
    print("\nfewest rows:")
    print(rep.head(8).to_string())
    if rep["ends_early"].any():
        print("\nseries ending early (possible delisting/acquisition):")
        print(rep[rep["ends_early"]].to_string())


if __name__ == "__main__":
    main()