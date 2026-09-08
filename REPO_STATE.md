# Repository state report

Read-only reconnaissance of `equity-graph-clustering`, generated 2026-09-08.
Working tree at commit `e9d3662`, branch `phase3-knn-graphs`.

---

## 1. Directory tree

Excludes `.venv`, `.git`, `__pycache__`, `.ipynb_checkpoints`, `.pytest_cache`.
Contents of `data/raw` and `data/processed` are listed by filename only.

```
.
├── .claude/
│   └── settings.local.json
├── .gitignore
├── README.md
├── requirements.txt
├── sp100_major_constituents.csv
├── tutorial.ipynb
├── config/
│   └── base.yaml
├── data/
│   ├── sp100_major_constituents.csv
│   ├── universe_sp100.csv
│   ├── raw/                        # filenames only
│   │   ├── prices_raw.parquet
│   │   └── _batches/
│   │       ├── batch_001.parquet
│   │       ├── batch_002.parquet
│   │       ├── batch_003.parquet
│   │       ├── batch_004.parquet
│   │       └── batch_005.parquet
│   └── processed/                  # filenames only
│       ├── coverage.csv
│       ├── features.npz
│       ├── graphs.npz
│       ├── graphs_mi.npz
│       ├── homophily.csv
│       ├── panel.parquet
│       └── _mi_cache/              # 128 files, mi_ksg_<YYYY-MM-DD>.npy
├── docs/
│   ├── phase2_maths_reference.md
│   └── phase3_maths_reference.md
├── notebooks/
│   ├── 02_features.ipynb
│   └── 03_graphs.ipynb
├── prompts/
│   ├── phase1_universe_fix.json
│   ├── phase2_node_features.json
│   ├── phase3a_knn_graphs.json
│   ├── phase3b_mutual_info.json
│   └── phase3c_homophily.json
├── src/
│   ├── __init__.py
│   ├── cluster/                    # EMPTY (0 entries)
│   ├── eval/                       # EMPTY (0 entries)
│   ├── llm/                        # EMPTY (0 entries)
│   ├── models/                     # EMPTY (0 entries)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── clean.py
│   │   ├── ingest.py
│   │   └── scratch.ipynb           # 0 bytes
│   ├── features/
│   │   ├── __init__.py
│   │   └── node_features.py
│   └── graphs/
│       ├── __init__.py
│       ├── construct.py
│       ├── homophily.py
│       └── mutual_info.py
└── tests/
    ├── test_construct.py
    ├── test_homophily.py
    ├── test_mutual_info.py
    ├── test_node_features.py
    └── test_universe.py
```

---

## 2. Module inventory

No file under `src/` or `tests/` contains `TODO`, raises `NotImplementedError`,
or carries a `pytest.mark.skip` / `pytest.mark.xfail` marker.

### `src/__init__.py`
Empty (0 lines).

### `src/data/__init__.py`
Empty (0 lines).

### `src/data/clean.py` — 297 lines
Module docstring: *Clean raw OHLCV into a canonical, calendar-aligned panel.*

| Signature | First docstring line |
|---|---|
| `def load_raw(path: str \| Path) -> pd.DataFrame` | Load the raw long panel and enforce basic invariants. |
| `def to_wide(long: pd.DataFrame) -> dict[str, pd.DataFrame]` | Long -> {field: DataFrame indexed by date, columns by ticker}. |
| `def to_long(wide: dict[str, pd.DataFrame]) -> pd.DataFrame` | {field: wide frame} -> long frame, dropping all-NaN observations. |
| `def coverage_stats(close: pd.DataFrame) -> pd.DataFrame` | Per-ticker coverage, computed on a wide close frame. |
| `def build_calendar(close: pd.DataFrame, min_coverage: float) -> pd.DatetimeIndex` | Sessions on which at least `min_coverage` of tickers have a price. |
| `def filter_universe(close: pd.DataFrame, min_history: int, max_internal_nan: float) -> list[str]` | Return the tickers that pass the history and completeness filters. |
| `def filter_full_history(close: pd.DataFrame, tolerance_days: int) -> list[str]` | Return the tickers already trading at the start of the sample. |
| `def fill_prices(wide: dict[str, pd.DataFrame], max_ffill: int) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]` | Forward-fill price fields with a hard limit; flag what was filled. |
| `def add_derived(wide: dict[str, pd.DataFrame], filled_mask: pd.DataFrame) -> dict[str, pd.DataFrame]` | Log returns, dollar volume, and the is_filled flag. |
| `def build_panel(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]` | Raw long frame -> (canonical panel, coverage report). |
| `def main() -> None` | *(no docstring)* |

### `src/data/ingest.py` — 214 lines
Module docstring: *Download and cache raw daily OHLCV for the universe.*

| Signature | First docstring line |
|---|---|
| `def reshape(df: pd.DataFrame, batch: list[str]) -> pd.DataFrame` | Wide (date x [ticker, field]) -> long (date, ticker, o/h/l/c, volume). |
| `def download_raw(tickers: list[str], start: str, end: str \| None, cache_dir: Path, batch_size: int = 25, pause: float = 1.0, retries: int = 3, use_cache: bool = True) -> pd.DataFrame` | Download in batches, caching each batch so a run can be resumed. |
| `def download_report(panel: pd.DataFrame, requested: list[str]) -> pd.DataFrame` | Did we get, for each requested ticker, a plausible response? |
| `def load_universe(path: str \| Path) -> list[str]` | Read the committed universe CSV and resolve the symbols to request. |
| `def main() -> None` | *(no docstring)* |

Private: `_stack_compat`, `_download_batch`.

### `src/features/__init__.py`
Empty (0 lines).

### `src/features/node_features.py` — 569 lines
Module docstring: *Phase 2 node features: collapse the daily panel into a [T, N, F] tensor.*

| Signature | First docstring line |
|---|---|
| `def to_wide(panel: pd.DataFrame) -> dict[str, pd.DataFrame]` | Long panel -> {field: DataFrame indexed by date, columns by ticker}. |
| `def rebalance_dates(index: pd.DatetimeIndex, freq: str = 'ME', burn_in: int = 252) -> pd.DatetimeIndex` | Sample the daily clock down to the rebalance clock. |
| `def market_beta(ret: pd.DataFrame, window: int) -> tuple[pd.DataFrame, pd.DataFrame]` | Rolling beta and idiosyncratic volatility against the equal-weighted market. |
| `def build_features(wide: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]` | Compute all fifteen node features on the daily clock. |
| `def standardise(X: pd.DataFrame, winsor: float = 3.0) -> pd.DataFrame` | Robust cross-sectional standardisation, one row (date) at a time. |
| `def assemble(features: dict[str, pd.DataFrame], dates: pd.DatetimeIndex, winsor: float = 3.0, exclude: list[str] \| None = None) -> tuple[np.ndarray, list[str], list[str]]` | Standardise, sample on the rebalance clock, and stack into [T, N, F]. |
| `def feature_correlation(X: np.ndarray, feature_names: list[str]) -> pd.DataFrame` | Pooled F x F Pearson correlation across all dates and tickers. |
| `def top_correlated_pairs(corr: pd.DataFrame, k: int = 10) -> pd.Series` | The `k` feature pairs with the largest absolute correlation. |
| `def main() -> None` | *(no docstring)* |

Private: `_min_periods`, `_risk_features`, `_liquidity_features`, `_trend_features`.

Note: `rebalance_dates` and `market_beta` were originally specified as stubs
raising `NotImplementedError`. Both are now fully implemented.

### `src/graphs/__init__.py`
Empty (0 lines).

### `src/graphs/construct.py` — 546 lines
Module docstring: *Phase 3 Part A: a temporal sequence of correlation kNN graphs.*

| Signature | First docstring line |
|---|---|
| `def load_returns(panel_path: str, tickers: list[str]) -> pd.DataFrame` | Wide date x ticker log-return frame in the exact given ticker order. |
| `def residualise(R: np.ndarray, n_factors: int = 1) -> np.ndarray` | Project out the leading `n_factors` principal components of a window. |
| `def shrunk_correlation(R: np.ndarray, method: str = 'ledoit_wolf') -> tuple[np.ndarray, float]` | Shrunk correlation matrix from a return window. |
| `def marchenko_pastur_edge(n: int, t: int) -> float` | Upper edge of the Marchenko-Pastur bulk, `(1 + sqrt(n / t))^2`. |
| `def knn_adjacency(S: np.ndarray, k: int, symmetrise: str = 'union', weighted: bool = True) -> np.ndarray` | k-nearest-neighbour adjacency from an N x N similarity matrix. |
| `def build_graph_sequence(ret: pd.DataFrame, dates: pd.DatetimeIndex, cfg: dict, k: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]` | Build the full graph sequence on the rebalance clock. |
| `def edge_persistence(A1: np.ndarray, A2: np.ndarray) -> float` | Jaccard overlap of two adjacency matrices' edge sets. |
| `def graph_diagnostics(A: np.ndarray) -> pd.DataFrame` | Per-date structural diagnostics for a [T, N, N] adjacency stack. |
| `def sweep_k(ret: pd.DataFrame, dates: pd.DatetimeIndex, cfg: dict, k_values: list[int]) -> pd.DataFrame` | Summarise the graph sequence at each candidate k. |
| `def main() -> None` | *(no docstring)* |

Private: `_correlation_stack`, `_adjacency_stack`, `_gini`.

### `src/graphs/mutual_info.py` — 856 lines
Module docstring: *Phase 3 Part B: mutual-information kNN graphs, as a control against Part A.*

| Signature | First docstring line |
|---|---|
| `def gaussian_mi(rho: np.ndarray \| float) -> np.ndarray \| float` | Exact mutual information of a bivariate Gaussian pair, in nats. |
| `def mi_binned(x: np.ndarray, y: np.ndarray, bins: int = 6, scheme: str = 'quantile', correct_bias: bool = True) -> float` | Plug-in (histogram) mutual information estimator, in nats. |
| `def mi_ksg(x: np.ndarray, y: np.ndarray, n_neighbors: int = 4) -> float` | Kraskov-Stogbauer-Grassberger mutual information, in nats. |
| `def mi_matrix(R: np.ndarray, estimator: str = 'ksg', n_jobs: int = -1, **kwargs) -> np.ndarray` | Full N x N mutual information matrix for a return window. |
| `def marginal_entropies(R: np.ndarray, bins: int = 6) -> np.ndarray` | Per-asset Shannon entropy of the binned return series, in nats. |
| `def normalise_mi(M: np.ndarray, H: np.ndarray) -> np.ndarray` | Normalised MI, `NMI_ij = M_ij / sqrt(H_i * H_j)`, clipped to [0, 1]. |
| `def mi_similarity(M: np.ndarray, C: np.ndarray, H: np.ndarray, signed: bool) -> np.ndarray` | The similarity matrix fed to `knn_adjacency`. |
| `def nonlinear_excess(M: np.ndarray, C: np.ndarray) -> np.ndarray` | Dependence not explained by linear correlation: `M - gaussian_mi(C)`. |
| `def validate_estimator(rhos: list[float], T: int = 252, n_trials: int = 20, estimators: tuple = ('binned', 'ksg'), seed: int = 0, **kwargs) -> pd.DataFrame` | Recover the Gaussian MI curve; the estimator's unit test as a report. |
| `def mi_null_floor(R: np.ndarray, estimator: str = 'ksg', n_perm: int = 100, n_pairs: int = 200, seed: int = 0, n_jobs: int = -1, **kwargs) -> dict` | Permutation null distribution for MI on this window. |
| `def timing_probe(R: np.ndarray, estimator: str, n_pairs: int = 200, n_dates: int = 128, **kwargs) -> dict` | Time `n_pairs` MI estimates and extrapolate to the full run. |
| `def build_mi_graph_sequence(ret: pd.DataFrame, dates: pd.DatetimeIndex, C_stack: np.ndarray, cfg: dict, k: int, estimator: str, signed: bool, cache_dir: Path \| None = None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]` | Build the MI-kNN graph sequence on the rebalance clock. |
| `def compare_graphs(A_corr: np.ndarray, A_mi: np.ndarray) -> pd.DataFrame` | Per-date comparison of the correlation and MI graphs. |
| `def main() -> None` | *(no docstring)* |

Private: `_bin_index`, `_estimator_fn`, `_pair_chunk`, `_estimator_kwargs`.
Imports `residualise`, `knn_adjacency`, `load_returns`, `edge_persistence`
from `src.graphs.construct`.

### `src/graphs/homophily.py` — 380 lines
Module docstring: *Phase 3c: homophily and feature-smoothness diagnostics for the graphs.*

| Signature | First docstring line |
|---|---|
| `def load_sector_labels(universe_csv: str \| Path, tickers: list[str]) -> np.ndarray` | GICS sector per ticker, in the exact order of `tickers`. |
| `def edge_homophily(A: np.ndarray, labels: np.ndarray) -> dict` | Fraction of edges joining same-label nodes, raw and adjusted. |
| `def assortativity(A: np.ndarray, labels: np.ndarray) -> float` | Newman's attribute assortativity coefficient for a categorical label. |
| `def feature_smoothness(A: np.ndarray, X_t: np.ndarray) -> float` | Normalised Dirichlet energy: connected-pair distance over all-pair distance. |
| `def per_feature_smoothness(A: np.ndarray, X_t: np.ndarray, feature_names: list[str]) -> pd.Series` | `feature_smoothness` computed one feature at a time. |
| `def analyse(A_stack: np.ndarray, X: np.ndarray, labels: np.ndarray, dates: pd.DatetimeIndex, feature_names: list[str], name: str) -> tuple[pd.DataFrame, pd.Series]` | Run every metric over a whole graph sequence. |
| `def main() -> None` | *(no docstring)* |

Private: `_pairwise_sq_dists`.

### Tests — 107 collected, 0 skipped

| File | Lines | Test functions |
|---|---|---|
| `tests/test_construct.py` | 307 | 21 (1 parametrized) |
| `tests/test_homophily.py` | 204 | 17 |
| `tests/test_mutual_info.py` | 381 | 26 (4 parametrized) |
| `tests/test_node_features.py` | 444 | 26 (1 parametrized) |
| `tests/test_universe.py` | 70 | 4 (1 parametrized) |

Parametrized tests, with exact signatures:

- `tests/test_construct.py` — `def test_shrunk_correlation_is_a_valid_correlation_matrix(method) -> None`
  `@pytest.mark.parametrize('method', ['ledoit_wolf', 'sample'])`
- `tests/test_mutual_info.py` — `def test_ksg_recovers_the_gaussian_curve(rho) -> None`
  `@pytest.mark.parametrize('rho', [0.0, 0.3, 0.6, 0.9])`
- `tests/test_mutual_info.py` — `def test_binned_recovers_the_gaussian_curve(rho, tol) -> None`
  `@pytest.mark.parametrize('rho,tol', [(0.0, 0.03), (0.3, 0.04), (0.6, 0.04), (0.9, 0.1)])`
- `tests/test_mutual_info.py` — `def test_estimators_are_non_negative_and_symmetric(estimator) -> None`
  `@pytest.mark.parametrize('estimator', ['binned', 'ksg'])`
- `tests/test_mutual_info.py` — `def test_independent_series_have_a_small_but_non_zero_floor(estimator) -> None`
  `@pytest.mark.parametrize('estimator', ['binned', 'ksg'])`
- `tests/test_node_features.py` — `def test_clip_is_final_across_zero_inflation(zero_frac: float) -> None`
  `@pytest.mark.parametrize('zero_frac', [0.5, 0.75, 0.8, 0.9, 0.95])`
- `tests/test_universe.py` — `def test_filter_full_history_respects_tolerance(tolerance, expected) -> None`
  `@pytest.mark.parametrize('tolerance,expected', [(0, ['DAY0']), (3, ['DAY0', 'DAY3']), (60, ['DAY0', 'DAY3', 'DAY60'])])`

---

## 3. Config — `config/base.yaml` verbatim

```yaml
universe:
  name: sp100
  min_history_days: 1260        # 5y — names with less are dropped
  max_missing_frac: 0.02
  min_session_coverage: 0.80    # fraction of tickers needed for a valid session
  max_ffill_days: 3             # hard cap on carrying a stale price
  # Keep N fixed: drop names not already trading at the sample start, so the
  # panel is a complete rectangle and Phase 3 correlations are never NaN.
  require_full_history: true
  # Grace period for the first valid close, in calendar days after the first
  # session — absorbs holidays and a late first print without admitting IPOs.
  start_tolerance_days: 7

data:
  start: "2015-01-01"
  end: null                      # null = today
  raw_dir: data/raw
  processed_dir: data/processed
  batch_size: 25
  pause: 1.0
  end: "2026-08-01"     # pinned for reproducibility; bump deliberately, never implicitly

features:
  rebalance_freq: "ME"           # node features are sampled month-end
  burn_in: 252                   # sessions dropped so the 252d window is full
  winsor: 3.0                    # cross-sectional clip, in robust sigmas
  exclude:                       # computed, then dropped in assemble
    - amihud_63                  # -0.962 with log_adv_63: dollar volume is
                                 # its denominator, so near-duplicate
    - zero_ret_frac_63           # 77.6% exactly zero: a near-constant
                                 # cross-section cannot separate clusters

graph:
  corr_window: 252               # T, from section 1.0
  rebalance: "ME"                # month-end
  n_factors: 1                   # PCs removed per window; 1 = the market mode
  shrinkage: ledoit_wolf         # intensity re-estimated on every window
  # signed uses the raw correlation, so an edge means co-movement in the SAME
  # direction. absolute would use |rho|, which makes hedges into neighbours:
  # two stocks at rho = -0.8 are opposites, not peers, and grouping them
  # destroys the very structure the clustering is meant to find.
  similarity: signed
  k: 10                          # neighbours per node, before symmetrisation
  k_sweep: [5, 7, 10, 15, 20]
  symmetrise: union              # max(D, D.T); intersection isolates periphery
  weighted: true                 # multiply the mask by the correlation

mi:
  estimator: ksg                 # KSG avoids binning; see B.6 of the maths ref
  n_neighbors: 4                 # KSG kappa: small = less bias, more variance
  bins: 6                        # binned estimator + marginal entropies only
  bin_scheme: quantile           # equal-frequency: returns are heavy-tailed
  # MI depends on rho^2, so rho = +0.9 and rho = -0.9 are identical (0.83
  # nats). Unsigned, the graph wires an asset to its strongest HEDGE as
  # readily as to its closest peer. signed: true masks MI by sign(rho) and
  # keeps only positive entries, so an edge means same-direction dependence.
  signed: true
  n_perm: 100                    # permutations per pair for the null floor
  # Evenly spaced subset, so a first run costs minutes rather than hours.
  # Set to null for the full 128-date run once the timing probe has confirmed
  # the cost is acceptable.
  subsample_dates: null          # full 128-date run
  cache_dir: data/processed/_mi_cache
```

---

## 4. Data artefacts under `data/processed`

### `features.npz`

| Key | Shape | dtype |
|---|---|---|
| `X` | `(128, 105, 13)` | float64 |
| `dates` | `(128,)` | object (ISO strings) |
| `feature_names` | `(13,)` | object |
| `tickers` | `(105,)` | object |

Axis order `(date, ticker, feature)`. `feature_names` =
`vol_21, vol_63, semivol_63, volofvol_63, skew_252, kurt_252, log_adv_63,
beta_252, idio_vol_252, mom_21, mom_63, mom_252_21, dist_52w_high`.
All values finite; `max|X| = 3.0000`.

### `graphs.npz`

| Key | Shape | dtype |
|---|---|---|
| `A` | `(128, 105, 105)` | float32 |
| `A_binary` | `(128, 105, 105)` | uint8 |
| `C` | `(128, 105, 105)` | float32 |
| `config_json` | `()` | `<U188` |
| `dates` | `(128,)` | object |
| `tickers` | `(105,)` | object |

`config_json` records:
`{corr_window: 252, k: 10, k_sweep: [5,7,10,15,20], n_factors: 1,
rebalance: "ME", shrinkage: "ledoit_wolf", similarity: "signed",
symmetrise: "union", weighted: true}`

### `graphs_mi.npz`

| Key | Shape | dtype |
|---|---|---|
| `A` | `(128, 105, 105)` | float32 |
| `A_binary` | `(128, 105, 105)` | uint8 |
| `Delta` | `(128, 105, 105)` | float32 |
| `M` | `(128, 105, 105)` | float32 |
| `config_json` | `()` | `<U379` |
| `dates` | `(128,)` | object |
| `tickers` | `(105,)` | object |

`config_json` records the full `graph` block plus
`mi: {bin_scheme: "quantile", bins: 6, cache_dir: "data/processed/_mi_cache",
estimator: "ksg", n_neighbors: 4, n_perm: 100, signed: true,
subsample_dates: null}`.

### `panel.parquet` — 307,545 rows

| Column | dtype |
|---|---|
| `date` | datetime64[ms] |
| `ticker` | str |
| `open` | float64 |
| `high` | float64 |
| `low` | float64 |
| `close` | float64 |
| `volume` | float64 |
| `log_ret` | float64 |
| `dollar_volume` | float64 |
| `is_filled` | bool |

105 tickers × 2,929 sessions, 2015-01-02 to 2026-08-26.

### `coverage.csv` — 105 rows
Columns: `n_obs, first_date, last_date, frac_sessions, frac_internal_nan,
max_abs_ret, n_filled` (index = ticker).

### `homophily.csv` — 256 rows
Columns: `graph, homophily_raw, homophily_expected, homophily_adjusted,
assortativity, smoothness` (index = date). 128 rows per graph, for
`correlation` and `mutual_information`.

### `_mi_cache/`
128 `.npy` files, one per rebalance date, named `mi_ksg_<YYYY-MM-DD>.npy`.

### Cross-artefact consistency

| Check | Result |
|---|---|
| Ticker count in `features.npz` / `graphs.npz` / `graphs_mi.npz` | 105 / 105 / 105 |
| Ticker **order** identical across all three | **True** |
| Date count in all three | 128 / 128 / 128 |
| Date **list** identical across all three | **True** |
| Date range | 2016-01-29 .. 2026-08-26 |
| `N` in `X` equals `N` in `A` | True (105) |
| `T` in `X` equals `T` in `A` | True (128) |
| Panel ticker set equals artefact ticker set | True |
| All 128 rebalance dates are real panel sessions | True |

---

## 5. Notebooks

### `notebooks/02_features.ipynb`
38 cells (19 markdown, 19 code). 0 unrun, 0 errored.
Markdown cells in order — note that only 7 use `#` header syntax; the rest are
plain-text pseudo-headers, reproduced here as their first line:

```
# Phase 2 — node features
Load the cached data panel
Check 1 — Tensor shape
Check 2 — Pre-standardisation value sanity
2a — Beta must average to 1 (the strongest check available)
2b — Idiosyncratic vol must be below total vol
2c — Drawdown must be non-positive
2d — Amihud and ADV must be logged
2e — Volatility must spike in March 2020
Check 3 — NaN and clipping rates
### Clip rate
### Check 3b — the winsor clip is the final operation
Check 4 — Feature redundancy
(prose cell: correlation heatmap)
### Check 4b — why two features are excluded
Check 5 — Dead features
### Check 6 — standardisation behaved
### Check 7 — no-lookahead canary
### Summary
```

### `notebooks/03_graphs.ipynb`
30 cells (16 markdown, 14 code). 0 unrun, 0 errored.
Markdown headers (`#` syntax) in order:

```
# Phase 3 — graph construction
## Load Part A
## Check 1 — is the graph dynamic?
## Check 2 — is there signal above the noise floor?
## Check 3 — does the structure line up with sectors?
## Load Part B
## Check 4 — does MI agree with the Gaussian benchmark?
## Check 5 — how much of the excess is real?
### Correlation vs mutual-information topology, same date
# Phase 3c — homophily and feature smoothness
### Reading the table
### Which features does the graph actually smooth?
### Verdict for Phase 4
```

### Notebooks outside `notebooks/`
- `tutorial.ipynb` (repo root) — 1 cell (0 markdown, 1 code). Stored output is
  an error: `FileNotFoundError: [Errno 2] No such file or directory:
  'data/processed/panel.parquet'`.
- `src/data/scratch.ipynb` — 0 bytes, not valid JSON.

---

## 6. Dependencies

### `requirements.txt`

```
# Pinned environment for equity-graph-clustering.
#
# Built and verified on Python 3.14.6 (darwin/arm64).
# Pins are exact on purpose: bump them deliberately, never implicitly — the
# same rule config/base.yaml applies to data.end.
#
#   python3 -m venv .venv
#   .venv/bin/pip install -r requirements.txt
#
# Every entry below is imported by src/, tests/ or notebooks/, or is required
# at runtime by something that is.

# --- core pipeline (src/data, src/features, src/graphs) ----------------
numpy==2.5.2
pandas==3.0.5
PyYAML==6.0.3
pyarrow==25.0.1          # pandas parquet engine for data/processed/*.parquet

# --- Phase 1: ingest ---------------------------------------------------
yfinance==1.6.0          # src/data/ingest.py only

# --- Phase 3: graphs ---------------------------------------------------
scipy==1.18.1            # sparse.csgraph.connected_components
scikit-learn==1.9.0      # covariance.LedoitWolf

# --- tests -------------------------------------------------------------
pytest==9.1.1

# --- notebooks ---------------------------------------------------------
matplotlib==3.11.1
networkx==3.6.1          # notebooks/03_graphs.ipynb graph drawing
jupyter==1.1.1           # metapackage: notebook, ipykernel, jupyterlab
nbconvert==7.17.1        # `jupyter nbconvert --execute`, used to run 02_features
```

### Deep-learning stack

```
$ python -c "import torch"
ModuleNotFoundError: No module named 'torch'

$ python -c "import torch_geometric"
ModuleNotFoundError: No module named 'torch_geometric'
```

Neither `torch` nor `torch_geometric` is installed, and neither appears in
`requirements.txt`. `joblib` is used by `src/graphs/mutual_info.py` and is
present in the environment (1.6.0) as a transitive dependency of
`scikit-learn`, but is not listed explicitly in `requirements.txt`.

---

## 7. Git state

- **Current branch:** `phase3-knn-graphs`
- **Working tree:** clean (`git status --porcelain` returns nothing)
- **Tracked files:** 32

Last five commit subjects:

```
e9d3662  Phase 3c: homophily and feature-smoothness diagnostics
0dadcd7  Notebooks: add 03_graphs, document both, fix the sector join
c2d808f  Phase 3b: mutual-information graphs as a control against correlation
b55e378  Pin the environment in requirements.txt
844958c  Phase 3a: correlation kNN graph sequence (128 x 105 x 105)
```

Branches: `main` and `phase3-knn-graphs` both at `e9d3662` and in sync with
`origin/main`; `phase2-node-features` at `d14895e` (fully merged, 5 commits
behind).

`.gitignore` contents:

```
.venv/
data/
!data/universe_sp100.csv
__pycache__/
.ipynb_checkpoints/
```

`data/universe_sp100.csv` is tracked. Everything else under `data/` is
ignored, including all `.npz`, `panel.parquet`, `coverage.csv`,
`homophily.csv` and `_mi_cache/`.

---

## 8. Gaps and inconsistencies

Factual observations only.

**Unimplemented code**
- None. No `NotImplementedError`, no `TODO`, no skipped or xfailed tests
  anywhere in `src/` or `tests/`. The two Phase 2 stubs (`rebalance_dates`,
  `market_beta`) are both implemented. 107 tests collected, 0 skipped.

**Empty package directories**
- `src/cluster/`, `src/eval/`, `src/llm/`, `src/models/` exist but contain
  zero entries — no `__init__.py`, no modules. They are not importable as
  packages and are not referenced anywhere in `src/`, `tests/`, or
  `config/base.yaml`.

**Config**
- `config/base.yaml` defines `data.end` **twice**: line 16 (`end: null`) and
  line 21 (`end: "2026-08-01"`). YAML takes the last occurrence, so the
  effective value is `"2026-08-01"`. The first is dead.
- The effective `data.end` of `2026-08-01` is **earlier** than the panel's
  actual last session, `2026-08-26`, so `panel.parquet` extends 25 calendar
  days past the configured end date.
- No config section exists for models, clustering, evaluation, or training.

**Missing dependencies for the next phase**
- Neither `torch` nor `torch_geometric` is installed or declared.
- `joblib` is imported directly by `src/graphs/mutual_info.py` but is not
  listed in `requirements.txt`; it is present only as a transitive
  dependency of `scikit-learn`.

**Duplicate and stray files**
- `sp100_major_constituents.csv` exists at both the repo root and
  `data/sp100_major_constituents.csv`. The two files are byte-identical
  (442 bytes). The root copy is tracked; the `data/` copy is gitignored.
  Neither is read by any module in `src/` — the modules read
  `data/universe_sp100.csv`, which has a different schema
  (`ticker, name, sector, yahoo_ticker, status, notes` versus
  `Symbol, Company Name, GICS Sector`).
- `src/data/scratch.ipynb` is 0 bytes and is not valid JSON. Any tool that
  parses notebooks across the tree will fail on it.
- `tutorial.ipynb` at the repo root contains one code cell whose stored
  output is a `FileNotFoundError` on `data/processed/panel.parquet`. Its last
  line indexes a DataFrame positionally (`ret[0]`, `ret[-1]`), which is a
  `KeyError` against ticker-labelled columns.

**Universe join**
- `data/universe_sp100.csv` stores `BRK.B` in `yahoo_ticker`, while every
  downstream artefact carries `BRK-B` (normalised by `src/data/ingest.py`).
  A naive join on the raw `yahoo_ticker` column yields **1 NaN sector**;
  normalising `.` to `-` first yields **0**. `load_sector_labels` in
  `src/graphs/homophily.py` performs the normalisation and asserts on the
  result; the CSV itself is unchanged.

**Artefact reproducibility**
- All derived artefacts (`features.npz`, `graphs.npz`, `graphs_mi.npz`,
  `panel.parquet`, `coverage.csv`, `homophily.csv`, `_mi_cache/`) are
  gitignored. A fresh clone contains no data, and regenerating from
  `data/universe_sp100.csv` requires a network download via
  `src/data/ingest.py`.

**Notebook coverage**
- There is no `01_*.ipynb` for Phase 1 (ingest/clean); notebooks exist only
  for Phases 2 and 3.
- There is no prompt spec for Phase 4 in `prompts/` (phases 1, 2, 3a, 3b, 3c
  are present).
- `notebooks/02_features.ipynb` uses plain-text pseudo-headers in 12 of its
  19 markdown cells, whereas `notebooks/03_graphs.ipynb` uses `#` markdown
  header syntax throughout.

**No inconsistencies found in**
- Ticker ordering: identical across `features.npz`, `graphs.npz`,
  `graphs_mi.npz`, and the panel.
- Date counts and date lists: identical (128) across all three artefacts;
  every rebalance date is a real trading session in the panel.
- Tensor dimensions: `N = 105` and `T = 128` agree between `X` and both
  adjacency stacks.
