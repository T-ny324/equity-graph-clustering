"""Phase 2 node features: collapse the daily panel into a [T, N, F] tensor.

Reads  : data/processed/panel.parquet   (long: date, ticker, close, log_ret,
                                         dollar_volume, ...)
Writes : data/processed/features.npz    (X, tickers, dates, feature_names)

Two clocks
----------
The DAILY clock carries returns and volumes. The REBALANCE clock (month-end)
carries graphs, embeddings and clusters. A node feature is a function that
collapses a trailing window of the daily clock into one number, evaluated on
the rebalance clock. Every feature here is therefore computed on the *full*
daily clock first and only then sampled on the rebalance dates. Looping over
rebalance dates and recomputing each window is both far slower and an
invitation to off-by-one alignment bugs.

Design boundary
---------------
These are NODE features: marginal properties of a single asset. Pairwise
quantities (correlation, partial correlation, co-movement) are EDGE features
and belong to Phase 3 (src/graphs/). Nothing pairwise is computed here.
`beta_252` and `idio_vol_252` are the deliberate exception: they are measured
against a market aggregate but describe the asset's own character.

Leakage rules
-------------
These are the highest-priority constraints in this module. A leaked feature
produces coherent-looking clusters that are worthless, and unlike a supervised
task there is no accuracy metric that will reveal it.

1. Every rolling window is right-aligned and backward-looking. X[t] may use
   data on (-inf, t] only.
2. `center=True` is never passed to any rolling or expanding call.
3. Negative shifts are forbidden. `shift(-k)` appears nowhere in this module.
4. `bfill` / `backfill` / forward-reading `interpolate` are never used.
5. Cross-sectional standardisation operates across tickers at a single date
   (axis=1). It must never operate across time.

Run from repo root:  python -m src.features.node_features
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TRADING_DAYS = 252
ANNUALISE = np.sqrt(TRADING_DAYS)

# Gaussian consistency constant: for normal data, MAD * 1.4826 -> sigma.
MAD_TO_SIGMA = 1.4826

# Fields lifted out of the long panel.
WIDE_FIELDS = ["close", "log_ret", "dollar_volume"]

# Canonical feature order. The last axis of X follows this list exactly.
FEATURE_NAMES = [
    # risk
    "vol_21", "vol_63", "semivol_63", "volofvol_63", "skew_252", "kurt_252",
    # liquidity
    "log_adv_63", "amihud_63", "zero_ret_frac_63",
    # factor
    "beta_252", "idio_vol_252",
    # trend
    "mom_21", "mom_63", "mom_252_21", "dist_52w_high",
]

# A window must be 80% populated before it produces a number.
MIN_PERIODS_FRAC = 0.8

# Semivariance is the one estimator whose sample is not the window: masking up
# days discards roughly half of it by construction, so applying the 80% rule to
# the surviving observations would leave the feature NaN almost everywhere.
# The 80%-of-window coverage requirement is instead enforced explicitly on the
# session count, and the std gets its own sample-size floor. A 63d window holds
# ~31 down days, so 20 is a comfortable floor that still catches a genuinely
# one-sided window.
MIN_DOWN_DAYS = 20


def _min_periods(window: int) -> int:
    """Minimum observations for a rolling window: 80% of its length.

    Parameters
    ----------
    window : int
        Window length in trading days.

    Returns
    -------
    int
        `int(0.8 * window)`.
    """
    return int(MIN_PERIODS_FRAC * window)


# ----------------------------------------------------------------------
# Reshaping
# ----------------------------------------------------------------------
def to_wide(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Long panel -> {field: DataFrame indexed by date, columns by ticker}.

    Parameters
    ----------
    panel : pd.DataFrame
        Long panel with columns date, ticker, close, log_ret, dollar_volume.

    Returns
    -------
    dict[str, pd.DataFrame]
        One date x ticker frame per field in `WIDE_FIELDS`.

    Raises
    ------
    ValueError
        If the close frame contains NaNs. Phase 1 guarantees a dense close
        rectangle; a hole here means the panel was rebuilt with different
        settings and every trailing window downstream is silently shortened.
    """
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])

    wide = {
        f: panel.pivot(index="date", columns="ticker", values=f).sort_index()
        for f in WIDE_FIELDS
    }

    close = wide["close"]
    offenders = close.columns[close.isna().any()]
    if len(offenders):
        counts = close[offenders].isna().sum().sort_values(ascending=False)
        detail = ", ".join(f"{t}={int(n)}" for t, n in counts.items())
        raise ValueError(
            f"close has NaNs for {len(offenders)} tickers; the panel is not "
            f"the dense rectangle Phase 1 promises -> {detail}"
        )

    print(f"  wide frames: {close.shape[1]} tickers x {close.shape[0]} sessions "
          f"({close.index[0].date()} to {close.index[-1].date()})")
    return wide


# ----------------------------------------------------------------------
# The rebalance clock
# ----------------------------------------------------------------------
def rebalance_dates(index: pd.DatetimeIndex, freq: str = "ME",
                    burn_in: int = 252) -> pd.DatetimeIndex:
    """Sample the daily clock down to the rebalance clock.

    Returns the last ACTUAL trading session in each period -- not the calendar
    period-end label, which is frequently a weekend or holiday and would not
    exist in `index`. The first `burn_in` sessions are dropped first, so that
    the longest trailing window (252 days) is fully populated on every date
    returned.

    Parameters
    ----------
    index : pd.DatetimeIndex
        The full daily trading calendar, sorted ascending.
    freq : str, default 'ME'
        Pandas offset alias for the rebalance period. 'ME' is month-end.
    burn_in : int, default 252
        Number of leading sessions to discard before sampling.

    Returns
    -------
    pd.DatetimeIndex
        Rebalance dates, every one of which is a member of `index`.
    """
    if not index.is_monotonic_increasing:
        raise ValueError("index must be sorted ascending")
    if burn_in >= len(index):
        raise ValueError(
            f"burn_in={burn_in} consumes the whole calendar "
            f"({len(index)} sessions)"
        )

    # Drop the burn-in so the longest trailing window is full on every date
    # returned, then take the maximum ACTUAL session within each period.
    # Resampling a Series whose *values* are the sessions is what keeps this
    # honest: resample's own labels are calendar period-ends, routinely
    # weekends or holidays, and those are not members of `index`.
    tail = index[burn_in:]
    last = pd.Series(tail, index=tail).resample(freq).max().dropna()

    dates = pd.DatetimeIndex(last.to_numpy(), name=index.name)
    assert dates.isin(index).all(), "rebalance date off the trading calendar"
    return dates


# ----------------------------------------------------------------------
# Factor features
# ----------------------------------------------------------------------
def market_beta(ret: pd.DataFrame, window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling beta and idiosyncratic volatility against the equal-weighted market.

    The market proxy is the equal-weighted cross-sectional mean return, i.e.
    `ret.mean(axis=1)`. Beta is computed as a rolling covariance divided by a
    rolling variance -- NOT a per-ticker regression loop, which is orders of
    magnitude slower for an identical answer. Idiosyncratic volatility is the
    annualised standard deviation of the residual `ret - beta * mkt`, obtained
    in closed form as `sqrt(Var(ret) - beta * Cov(ret, mkt))`.

    Parameters
    ----------
    ret : pd.DataFrame
        Daily log returns, date x ticker.
    window : int
        Trailing window length in trading days.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        `(beta, idio_vol)`, both date x ticker and aligned to `ret`.
    """
    mp = _min_periods(window)

    # Equal-weighted market proxy. This is the only place a cross-sectional
    # aggregate enters a node feature, and it enters as one scalar per date,
    # so nothing pairwise is computed or retained here.
    mkt = ret.mean(axis=1)

    # cov / var rather than a per-ticker regression loop: identical answer,
    # one pass over the frame instead of N.
    cov = ret.rolling(window, min_periods=mp).cov(mkt)
    var_mkt = mkt.rolling(window, min_periods=mp).var()
    beta = cov.div(var_mkt.where(var_mkt > 0), axis=0)

    # Residual variance in closed form. Substituting beta = cov / var_mkt into
    #   Var(r - beta*m) = Var(r) - 2*beta*Cov(r, m) + beta^2 * Var(m)
    # collapses it to Var(r) - beta*Cov(r, m), which is exactly the in-window
    # OLS residual variance and costs no second pass.
    #
    # Materialising `ret - beta*mkt` and rolling a std over that instead would
    # cost a whole extra window of history: the residual series only exists
    # once beta does, so the first idio_vol would land `window` sessions after
    # the first beta and blank out the early rebalance dates for every ticker.
    var_resid = ret.rolling(window, min_periods=mp).var() - beta * cov
    # The clip absorbs floating-point noise around an exact zero only.
    idio_vol = np.sqrt(var_resid.clip(lower=0.0)) * ANNUALISE

    return beta, idio_vol


# ----------------------------------------------------------------------
# Daily-clock feature blocks
# ----------------------------------------------------------------------
def _risk_features(ret: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Six dispersion and shape descriptors of the trailing return window."""
    vol_21 = ret.rolling(21, min_periods=_min_periods(21)).std() * ANNUALISE

    # Mask up days to NaN so the std is taken over down days only. Zero
    # returns are kept: a flat day is a real (non-positive) observation.
    down = ret.mask(ret > 0)
    # The shared 80%-of-window rule, applied to sessions rather than to the
    # post-mask sample. See MIN_DOWN_DAYS.
    covered = (ret.notna().rolling(63, min_periods=1).sum()
               >= _min_periods(63))
    semivol = down.rolling(63, min_periods=MIN_DOWN_DAYS).std() * ANNUALISE

    return {
        "vol_21": vol_21,
        "vol_63": ret.rolling(63, min_periods=_min_periods(63)).std() * ANNUALISE,
        "semivol_63": semivol.where(covered),
        # Vol-of-vol rolls over the already-rolling 21d vol series, so its
        # first value lands 21 + 63 sessions in, not 63.
        "volofvol_63": vol_21.rolling(63, min_periods=_min_periods(63)).std(),
        "skew_252": ret.rolling(252, min_periods=_min_periods(252)).skew(),
        # pandas .kurt() is already excess kurtosis (normal -> 0).
        "kurt_252": ret.rolling(252, min_periods=_min_periods(252)).kurt(),
    }


def _liquidity_features(ret: pd.DataFrame,
                        dollar_volume: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Three trading-friction descriptors."""
    mp63 = _min_periods(63)

    # Average dollar volume. The log is mandatory: dollar volume spans orders
    # of magnitude, and untransformed it behaves as a single-name outlier
    # detector rather than as a feature.
    adv = dollar_volume.rolling(63, min_periods=mp63).mean()

    # Amihud illiquidity: price impact per dollar traded. dollar_volume is
    # already NaN wherever volume was non-positive (handled in clean.py), so
    # no extra zero guard is needed on the daily ratio -- but check it.
    illiq = ret.abs() / dollar_volume
    assert not np.isinf(illiq.to_numpy()).any(), (
        "amihud_63: inf in the daily |log_ret| / dollar_volume ratio; "
        "clean.py should have masked non-positive volume"
    )
    amihud = illiq.rolling(63, min_periods=mp63).mean()

    # A stale price implies no trade, so log_ret is NaN on filled days
    # (clean.py). Mask those rather than counting them as "not a zero return",
    # which would understate illiquidity exactly where it is worst.
    zero_ret = (ret.abs() < 1e-8).astype(float).mask(ret.isna())

    return {
        "log_adv_63": np.log(adv),
        # log is mandatory here too: Amihud is severely right-skewed. The mask
        # only bites in the degenerate case of a window of exact zeros.
        "amihud_63": np.log(amihud.mask(amihud <= 0)),
        "zero_ret_frac_63": zero_ret.rolling(63, min_periods=mp63).mean(),
    }


def _trend_features(ret: pd.DataFrame,
                    close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Four path-shape descriptors."""
    mom_21 = ret.rolling(21, min_periods=_min_periods(21)).sum()
    mom_252 = ret.rolling(252, min_periods=_min_periods(252)).sum()

    high_252 = close.rolling(252, min_periods=_min_periods(252)).max()

    return {
        "mom_21": mom_21,
        "mom_63": ret.rolling(63, min_periods=_min_periods(63)).sum(),
        # Skip-month momentum: drop the most recent 21 sessions, which tend to
        # reverse rather than continue.
        "mom_252_21": mom_252 - mom_21,
        # The rolling max includes today, so this is <= 0 always and exactly 0
        # on a new 52-week high.
        "dist_52w_high": np.log(close / high_252),
    }


def build_features(wide: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compute all fifteen node features on the daily clock.

    Parameters
    ----------
    wide : dict[str, pd.DataFrame]
        Output of `to_wide`: close, log_ret and dollar_volume frames.

    Returns
    -------
    dict[str, pd.DataFrame]
        `{feature_name: DataFrame(date x ticker)}`, keyed and ordered by
        `FEATURE_NAMES`.
    """
    ret, close, dv = wide["log_ret"], wide["close"], wide["dollar_volume"]

    feats: dict[str, pd.DataFrame] = {}
    feats.update(_risk_features(ret))
    feats.update(_liquidity_features(ret, dv))

    beta, idio_vol = market_beta(ret, TRADING_DAYS)
    feats["beta_252"] = beta
    feats["idio_vol_252"] = idio_vol

    feats.update(_trend_features(ret, close))

    missing = set(FEATURE_NAMES) - set(feats)
    extra = set(feats) - set(FEATURE_NAMES)
    assert not missing and not extra, (
        f"feature set drifted from FEATURE_NAMES: missing={sorted(missing)}, "
        f"extra={sorted(extra)}"
    )
    return {name: feats[name] for name in FEATURE_NAMES}


# ----------------------------------------------------------------------
# Cross-sectional standardisation
# ----------------------------------------------------------------------
def standardise(X: pd.DataFrame, winsor: float = 3.0) -> pd.DataFrame:
    """Robust cross-sectional standardisation, one row (date) at a time.

    Centres on the row median, scales by row MAD * 1.4826, and clips to
    +/- `winsor`. The clip is the LAST operation and nothing may rescale after
    it: the bound is a hard guarantee that downstream code can rely on.

    Median/MAD rather than mean/std is not a stylistic choice: the standard
    deviation is itself inflated by the very outliers being clipped, so a
    mean/std clip at three sigma barely bites on a heavy-tailed cross-section.

    Parameters
    ----------
    X : pd.DataFrame
        A single feature, date x ticker.
    winsor : float, default 3.0
        Clip bound in robust standard deviations.

    Returns
    -------
    pd.DataFrame
        Same shape and labels as `X`, with every finite value in
        [-winsor, +winsor]. All-NaN rows stay NaN; degenerate rows (zero
        spread) become zeros.

    Notes
    -----
    Operates on axis=1 only. Standardising across time would leak the future
    into every past date.

    An earlier version re-centred and re-scaled after clipping, which silently
    undid it: on a zero-inflated feature the post-clip MAD is ~0, so the std
    fallback divides by a small number and pushes values back past the bound
    (observed: 7.28 at winsor 3.0). Do not reintroduce a post-clip transform.
    """
    med = X.median(axis=1)
    dev = X.sub(med, axis=0)
    mad = dev.abs().median(axis=1) * MAD_TO_SIGMA

    # MAD is zero whenever more than half the cross-section shares one value.
    # Fall back to the row std; if that is zero too the row is degenerate and
    # dev is already all zeros, so dividing by 1.0 yields the zeros we want.
    scale = mad.where(mad > 0, X.std(axis=1))

    # Clip last. Anything applied after this point can push values back over
    # the bound; see the note in the docstring.
    return dev.div(scale.where(scale > 0, 1.0), axis=0).clip(-winsor, winsor)


# ----------------------------------------------------------------------
# Assembly onto the rebalance clock
# ----------------------------------------------------------------------
def assemble(features: dict[str, pd.DataFrame], dates: pd.DatetimeIndex,
             winsor: float = 3.0,
             exclude: list[str] | None = None,
             ) -> tuple[np.ndarray, list[str], list[str]]:
    """Standardise, sample on the rebalance clock, and stack into [T, N, F].

    Parameters
    ----------
    features : dict[str, pd.DataFrame]
        Output of `build_features`.
    dates : pd.DatetimeIndex
        Rebalance dates; every one must exist on the daily clock.
    winsor : float, default 3.0
        Passed through to `standardise`.
    exclude : list[str] or None, default None
        Feature names to drop before stacking. They are still computed and
        standardised upstream, so the decision stays visible and reversible
        from config alone.

    Returns
    -------
    tuple[np.ndarray, list[str], list[str]]
        `(X, tickers, feature_names)` with X of shape [T, N, F] and axis order
        (date, ticker, feature).
    """
    dropped = set(exclude or [])
    unknown = dropped - set(features)
    if unknown:
        raise ValueError(
            f"features.exclude names {len(unknown)} unknown feature(s): "
            f"{sorted(unknown)}; known features are {list(features)}"
        )

    feature_names = [n for n in features if n not in dropped]
    if dropped:
        print(f"  excluded {len(dropped)}: {sorted(dropped)}")
    print(f"  F = {len(feature_names)} features retained")
    if not feature_names:
        raise ValueError("features.exclude removed every feature")

    first = features[feature_names[0]]
    tickers = list(first.columns)

    missing = pd.DatetimeIndex(dates).difference(first.index)
    assert not len(missing), (
        f"{len(missing)} rebalance dates are not trading sessions: "
        f"{list(missing[:5])}"
    )

    T, N, F = len(dates), len(tickers), len(feature_names)
    X = np.full((T, N, F), np.nan, dtype=float)

    print(f"  residual NaNs per feature (before filling, out of {T * N:,}):")
    for j, name in enumerate(feature_names):
        Z = standardise(features[name][tickers], winsor=winsor)
        X[:, :, j] = Z.loc[dates].to_numpy(dtype=float)
        n_nan = int(np.isnan(X[:, :, j]).sum())
        print(f"    {name:<18} {n_nan:>7,}  ({n_nan / (T * N):6.2%})")

    # 0.0 is the cross-sectional centre after standardisation, so an unfilled
    # asset is placed at the median rather than at an arbitrary extreme.
    nan_mask = np.isnan(X)
    n_filled = int(nan_mask.sum())
    X[nan_mask] = 0.0
    print(f"  filled {n_filled:,} NaNs with 0.0 "
          f"({n_filled / X.size:.3%} of X.size)")

    assert np.isfinite(X).all(), "X contains NaN or inf after filling"
    return X, tickers, feature_names


def feature_correlation(X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Pooled F x F Pearson correlation across all dates and tickers.

    Parameters
    ----------
    X : np.ndarray
        Feature tensor of shape [T, N, F].
    feature_names : list[str]
        Labels for the last axis.

    Returns
    -------
    pd.DataFrame
        F x F correlation matrix, labelled on both axes.
    """
    T, N, F = X.shape
    flat = pd.DataFrame(X.reshape(T * N, F), columns=feature_names)
    return flat.corr()


def top_correlated_pairs(corr: pd.DataFrame, k: int = 10) -> pd.Series:
    """The `k` feature pairs with the largest absolute correlation."""
    upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.where(upper).stack().dropna()
    return pairs.reindex(pairs.abs().sort_values(ascending=False).index[:k])


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def main() -> None:
    cfg = yaml.safe_load(open("config/base.yaml"))
    fcfg = cfg.get("features", {})
    freq = fcfg.get("rebalance_freq", "ME")
    burn_in = int(fcfg.get("burn_in", TRADING_DAYS))
    winsor = float(fcfg.get("winsor", 3.0))
    exclude = list(fcfg.get("exclude", []) or [])

    proc_dir = Path(cfg["data"]["processed_dir"])
    panel_path = proc_dir / "panel.parquet"
    out_path = proc_dir / "features.npz"

    print(f"loading {panel_path}")
    panel = pd.read_parquet(panel_path)

    print("[1] reshaping to wide")
    wide = to_wide(panel)

    print(f"[2] rebalance clock (freq={freq}, burn_in={burn_in})")
    dates = rebalance_dates(wide["close"].index, freq=freq, burn_in=burn_in)
    print(f"  {len(dates)} rebalance dates, "
          f"{dates[0].date()} to {dates[-1].date()}")

    print("[3] daily-clock features")
    features = build_features(wide)

    print(f"[4] standardising (winsor={winsor}) and sampling")
    X, tickers, feature_names = assemble(features, dates, winsor=winsor,
                                         exclude=exclude)
    print(f"  X.shape = {X.shape}  (dates x tickers x features)")

    np.savez_compressed(
        out_path,
        X=X,
        tickers=np.array(tickers, dtype=object),
        dates=np.array([d.strftime("%Y-%m-%d") for d in dates], dtype=object),
        feature_names=np.array(feature_names, dtype=object),
    )
    print(f"\nwrote {out_path}")

    print("\nten most correlated feature pairs:")
    for (a, b), v in top_correlated_pairs(
        feature_correlation(X, feature_names), k=10
    ).items():
        print(f"  {a:<18} {b:<18} {v:+.3f}")


if __name__ == "__main__":
    main()
