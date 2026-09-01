"""Tests for the Phase 2 node-feature module.

Everything here is built from small synthetic frames. Nothing reads
data/processed/panel.parquet: a test that reads real data is a description of
today's dataset, not a test of the code.

Run from repo root:  python -m pytest tests/test_node_features.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.features.node_features as nf

TRADING_DAYS = 252


# ----------------------------------------------------------------------
# Synthetic fixtures
# ----------------------------------------------------------------------
def _inputs(n: int = 400, n_tickers: int = 6,
            seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A dense (ret, close, dollar_volume) triple on a business-day index."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    cols = [f"T{i}" for i in range(n_tickers)]

    ret = pd.DataFrame(rng.normal(0.0, 0.012, size=(n, n_tickers)),
                       index=idx, columns=cols)
    close = 100.0 * np.exp(ret.cumsum())
    dv = pd.DataFrame(rng.lognormal(16.0, 0.5, size=(n, n_tickers)),
                      index=idx, columns=cols)
    return ret, close, dv


def _all_features(ret: pd.DataFrame, close: pd.DataFrame,
                  dv: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All fifteen features, straight through the public entry point."""
    return nf.build_features(
        {"log_ret": ret, "close": close, "dollar_volume": dv}
    )


def _non_factor_features(ret: pd.DataFrame, close: pd.DataFrame,
                         dv: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The blocks that need no market aggregate, for block-level tests."""
    feats = {}
    feats.update(nf._risk_features(ret))
    feats.update(nf._liquidity_features(ret, dv))
    feats.update(nf._trend_features(ret, close))
    return feats


# ----------------------------------------------------------------------
# Realised volatility
# ----------------------------------------------------------------------
def test_realised_vol_recovers_known_sigma() -> None:
    """vol_63 on i.i.d. normal returns recovers sigma * sqrt(252)."""
    sigma = 0.02
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2015-01-01", periods=2000)
    ret = pd.DataFrame({"A": rng.normal(0.0, sigma, size=2000)}, index=idx)

    vol = nf._risk_features(ret)["vol_63"]["A"].dropna()
    assert len(vol) > 1900
    assert vol.mean() == pytest.approx(sigma * np.sqrt(TRADING_DAYS), rel=0.10)


def test_semivol_is_actually_computed() -> None:
    """Masking up days must not leave the whole feature NaN.

    Guards the one place the module departs from the flat 80%-of-window rule:
    semivariance discards roughly half of every window by construction, so the
    coverage test runs on sessions and the std gets its own sample floor.
    """
    ret, _, _ = _inputs(n=400)
    risk = nf._risk_features(ret)
    semivol = risk["semivol_63"]

    # The rule this replaces: 80% of 63 masked observations is unreachable
    # when half of them are masked away, so it yields nothing at all.
    naive = ret.mask(ret > 0).rolling(63, min_periods=nf._min_periods(63)).std()
    assert naive.isna().all().all()

    # What the module does instead is almost always defined. It still declines
    # a window that is genuinely one-sided (fewer than MIN_DOWN_DAYS down
    # days), which is the point of keeping a floor at all.
    tail = semivol.iloc[100:].to_numpy()
    assert np.isnan(tail).mean() < 0.03
    assert (tail[~np.isnan(tail)] > 0).all()

    # Coverage must not start earlier than the 80%-of-window session rule
    # allows, exactly like vol_63.
    assert (semivol.notna().idxmax() >= risk["vol_63"].notna().idxmax()).all()


# ----------------------------------------------------------------------
# Momentum
# ----------------------------------------------------------------------
def test_mom_252_21_excludes_the_most_recent_month() -> None:
    """On a constant daily return r, mom_252_21 is exactly 231 * r."""
    r = 0.001
    idx = pd.bdate_range("2015-01-01", periods=300)
    ret = pd.DataFrame({"A": np.full(300, r)}, index=idx)
    close = pd.DataFrame({"A": 100.0 * np.exp(ret["A"].cumsum())}, index=idx)

    trend = nf._trend_features(ret, close)
    # 252 - 21 = 231 sessions of contribution; the most recent 21 are excluded.
    assert trend["mom_252_21"]["A"].iloc[-1] == pytest.approx(231 * r)
    assert trend["mom_21"]["A"].iloc[-1] == pytest.approx(21 * r)


# ----------------------------------------------------------------------
# Cross-sectional standardisation
# ----------------------------------------------------------------------
def test_standardise_centres_and_scales_each_row() -> None:
    """Output rows have median ~0 and a MAD-based scale ~1."""
    rng = np.random.default_rng(3)
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    X = pd.DataFrame(rng.normal(size=(4, 200)), index=idx)

    Z = nf.standardise(X)
    med = Z.median(axis=1)
    scale = Z.sub(med, axis=0).abs().median(axis=1) * nf.MAD_TO_SIGMA

    assert np.allclose(med.to_numpy(), 0.0, atol=1e-9)
    assert np.allclose(scale.to_numpy(), 1.0, atol=1e-9)


def test_standardise_clips_an_extreme_outlier_to_the_winsor_bound() -> None:
    winsor = 3.0
    rng = np.random.default_rng(0)
    row = rng.normal(size=50)
    row[7] = 500.0                       # raw robust z-score is ~494
    X = pd.DataFrame([row], index=pd.to_datetime(["2020-01-31"]))

    z = nf.standardise(X, winsor=winsor).iloc[0]

    assert z.idxmax() == 7               # still the largest, just bounded
    assert z.max() == pytest.approx(winsor, abs=1e-9)
    assert z.abs().max() <= winsor + 1e-9


def test_standardise_degenerate_row_returns_zeros() -> None:
    """A row of identical values yields zeros, not NaN or inf."""
    X = pd.DataFrame([[5.0] * 10], index=pd.to_datetime(["2020-01-31"]))
    Z = nf.standardise(X)

    assert np.isfinite(Z.to_numpy()).all()
    assert (Z.to_numpy() == 0.0).all()


def test_standardise_all_nan_row_stays_nan() -> None:
    """An all-NaN row is not information; assemble counts and fills it."""
    X = pd.DataFrame([[np.nan] * 10], index=pd.to_datetime(["2020-01-31"]))
    assert nf.standardise(X).isna().all().all()


def test_standardise_never_operates_across_time() -> None:
    """Perturbing one date must leave every other date's row untouched."""
    rng = np.random.default_rng(5)
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    X = pd.DataFrame(rng.normal(size=(6, 40)), index=idx)

    base = nf.standardise(X)
    bumped = X.copy()
    bumped.iloc[3] += 1e6
    after = nf.standardise(bumped)

    others = base.index.drop(idx[3])
    pd.testing.assert_frame_equal(base.loc[others], after.loc[others])


# ----------------------------------------------------------------------
# Distance from the 52-week high
# ----------------------------------------------------------------------
def test_dist_52w_high_is_non_positive_and_zero_at_a_new_high() -> None:
    ret, close, _ = _inputs(n=500, n_tickers=3, seed=7)
    dist = nf._trend_features(ret, close)["dist_52w_high"]

    values = dist.to_numpy()
    values = values[~np.isnan(values)]
    assert values.size > 0
    assert (values <= 1e-12).all()

    high = close.rolling(252, min_periods=nf._min_periods(252)).max()
    at_high = (close >= high) & high.notna()
    assert at_high.to_numpy().sum() > 0
    assert np.allclose(dist.to_numpy()[at_high.to_numpy()], 0.0, atol=1e-12)


def test_dist_52w_high_is_zero_throughout_a_monotone_rise() -> None:
    idx = pd.bdate_range("2015-01-01", periods=300)
    close = pd.DataFrame({"A": np.linspace(100.0, 300.0, 300)}, index=idx)
    ret = pd.DataFrame({"A": np.log(close["A"]).diff()}, index=idx)

    dist = nf._trend_features(ret, close)["dist_52w_high"]["A"].dropna()
    assert np.allclose(dist.to_numpy(), 0.0, atol=1e-12)


# ----------------------------------------------------------------------
# The canary
# ----------------------------------------------------------------------
def test_no_lookahead_canary() -> None:
    """The single most important test in this file.

    Corrupt the last row of the inputs to an extreme value and recompute.
    Every feature value strictly before that date must be bit-identical. A
    centred window, a negative shift, or a backward fill anywhere in the
    module would move an earlier value and fail here.
    """
    ret, close, dv = _inputs(n=400, n_tickers=5, seed=13)
    before = _all_features(ret, close, dv)

    last = ret.index[-1]
    ret_c, close_c, dv_c = ret.copy(), close.copy(), dv.copy()
    ret_c.loc[last] = 5.0                # a +500% day
    close_c.loc[last] = close.loc[last] * 100.0
    dv_c.loc[last] = dv.loc[last] * 1e6
    after = _all_features(ret_c, close_c, dv_c)

    assert set(before) == set(after)
    for name in before:
        pd.testing.assert_frame_equal(
            before[name].loc[:before[name].index[-2]],
            after[name].loc[:after[name].index[-2]],
            check_exact=True,
            obj=f"{name} changed before {last.date()}",
        )

    # And the corruption really did reach the module: the last row must move,
    # otherwise the test above passes vacuously.
    assert not np.allclose(
        before["vol_63"].loc[last].to_numpy(),
        after["vol_63"].loc[last].to_numpy(),
    )


def test_no_negative_shift_or_centred_window_in_source() -> None:
    """A grep-level backstop for the leakage rules in the module docstring."""
    src = (nf.__file__ and open(nf.__file__).read()) or ""
    body = src.split('"""', 2)[-1]       # skip the module docstring
    for forbidden in ("center=True", "shift(-", ".bfill(", ".backfill(",
                      "method=\"bfill\"", "method='bfill'"):
        assert forbidden not in body, f"forbidden construct in source: {forbidden}"


# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------
def test_assemble_stacks_in_date_ticker_feature_order() -> None:
    ret, close, dv = _inputs(n=400, n_tickers=4, seed=21)
    features = _all_features(ret, close, dv)
    dates = ret.index[[300, 320, 340, 360]]

    X, tickers, names = nf.assemble(features, dates)

    assert X.shape == (4, 4, 15)
    assert names == nf.FEATURE_NAMES
    assert tickers == list(ret.columns)
    assert np.isfinite(X).all()

    # X[:, :, j] must be feature names[j], sampled on `dates`.
    j = names.index("vol_63")
    expected = nf.standardise(features["vol_63"][tickers]).loc[dates].to_numpy()
    assert np.allclose(X[:, :, j], np.nan_to_num(expected), atol=1e-12)


def test_assemble_rejects_a_date_off_the_daily_clock() -> None:
    ret, close, dv = _inputs(n=400, n_tickers=3, seed=2)
    features = _non_factor_features(ret, close, dv)
    dates = pd.DatetimeIndex([ret.index[300], pd.Timestamp("1999-01-04")])

    with pytest.raises(AssertionError, match="not trading sessions"):
        nf.assemble(features, dates)


def test_feature_correlation_is_square_and_labelled() -> None:
    rng = np.random.default_rng(1)
    names = ["a", "b", "c"]
    X = rng.normal(size=(5, 7, 3))

    corr = nf.feature_correlation(X, names)
    assert list(corr.index) == names and list(corr.columns) == names
    assert np.allclose(np.diag(corr.to_numpy()), 1.0)


# ----------------------------------------------------------------------
# The rebalance clock and the factor block
# ----------------------------------------------------------------------
def test_rebalance_dates_returns_actual_sessions_after_burn_in() -> None:
    idx = pd.bdate_range("2015-01-01", periods=1000)
    dates = nf.rebalance_dates(idx, freq="ME", burn_in=252)

    assert dates.isin(idx).all()                 # never a weekend or holiday
    assert (dates >= idx[252]).all()             # burn-in respected
    assert dates.is_monotonic_increasing and not dates.duplicated().any()
    # One date per month, and each is that month's last actual session.
    for d in dates:
        month = idx[(idx.year == d.year) & (idx.month == d.month)]
        assert d == month.max()


def test_market_beta_recovers_a_known_beta() -> None:
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2015-01-01", periods=1000)
    mkt = rng.normal(0.0, 0.01, size=1000)
    ret = pd.DataFrame(
        {f"B{b}": b * mkt + rng.normal(0.0, 0.002, size=1000)
         for b in (0.5, 1.0, 1.5)},
        index=idx,
    )

    beta, idio = nf.market_beta(ret, 252)
    assert beta.shape == ret.shape and idio.shape == ret.shape
    # The market proxy is the equal-weighted mean, so betas are rescaled by
    # the mean beta (1.0 here); the ordering must survive regardless.
    last = beta.iloc[-1]
    assert last["B0.5"] < last["B1.0"] < last["B1.5"]
    assert (idio.dropna(how="all") >= 0).all().all()


def test_idio_vol_matches_an_explicit_ols_residual_std() -> None:
    """The closed form is the in-window OLS residual std, not an approximation."""
    window = 252
    rng = np.random.default_rng(4)
    n = 700
    idx = pd.bdate_range("2015-01-01", periods=n)
    mkt_true = rng.normal(0.0, 0.01, size=n)
    ret = pd.DataFrame(
        {f"B{b}": b * mkt_true + rng.normal(0.0, 0.004, size=n)
         for b in (0.5, 1.0, 1.5)},
        index=idx,
    )

    _, idio = nf.market_beta(ret, window)
    mkt = ret.mean(axis=1)

    t = n - 1
    x = mkt.iloc[t - window + 1:t + 1].to_numpy()
    design = np.column_stack([np.ones(window), x])       # intercept + market
    for col in ret.columns:
        y = ret[col].iloc[t - window + 1:t + 1].to_numpy()
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        expected = (y - design @ coef).std(ddof=1) * nf.ANNUALISE
        assert idio[col].iloc[t] == pytest.approx(expected, rel=1e-9)


def test_idio_vol_is_available_as_early_as_beta() -> None:
    """Both must clear the same 80%-of-window bar.

    Materialising the residual series and rolling a std over it would push
    idio_vol a full window later than beta and blank out the early rebalance
    dates for every ticker at once.
    """
    ret, _, _ = _inputs(n=700, n_tickers=4, seed=17)
    beta, idio = nf.market_beta(ret, 252)

    assert (beta.notna().idxmax() == idio.notna().idxmax()).all()
    assert beta.notna().to_numpy().sum() == idio.notna().to_numpy().sum()
    assert (idio.to_numpy()[~np.isnan(idio.to_numpy())] >= 0).all()


def test_build_features_returns_all_fifteen_named_features() -> None:
    ret, close, dv = _inputs(n=600, n_tickers=5, seed=9)
    wide = {"log_ret": ret, "close": close, "dollar_volume": dv}

    features = nf.build_features(wide)
    assert list(features) == nf.FEATURE_NAMES
    for name, frame in features.items():
        assert frame.shape == ret.shape, name
        assert list(frame.columns) == list(ret.columns), name
