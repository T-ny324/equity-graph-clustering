## Data provenance and known limitations

The universe in `data/universe_sp100.csv` is a **single historical snapshot of
S&P 100 membership applied retrospectively** across the whole sample. It is not
a point-in-time membership series: a name that entered or left the index during
the sample is treated as if its current status held throughout. Results are
therefore conditioned on survival, and index-inclusion effects are not
represented.

Symbol resolution is pinned in that CSV rather than delegated to the data
vendor. `yahoo_ticker` carries the symbol actually requested and `status`
records why it differs from `ticker`.

**Delisted (excluded, rows retained in the CSV as the record of the exclusion)**

| Ticker | Reason |
| --- | --- |
| DFS | Discover Financial merged into Capital One, 2025-05-18 |
| HES | Hess acquired by Chevron |
| IPG | Interpublic merged with Omnicom |

No recoverable price series exists for these three under any symbol, so the
panel omits them entirely rather than truncating them at the merger date.

**Renamed (resolved, history intact)**

| Ticker | Requested as | Reason |
| --- | --- | --- |
| BK | BNY | Ticker changed BK → BNY, 2026-05-21 |
| FI | FISV | Listing moved NYSE → Nasdaq, ticker FI → FISV |

**Excluded to keep N fixed**

Names not already trading at the sample start (2015-01-02, within a 7-day
tolerance) are dropped, so the panel is a complete rectangle with no leading
NaN block. With `universe.require_full_history: true` this removes six names:

| Ticker | First observation |
| --- | --- |
| HPE | 2015-10-19 |
| INVH | 2017-02-01 |
| IR | 2017-05-12 |
| FOXA | 2019-03-12 |
| FOX | 2019-03-13 |
| DOW | 2019-03-20 |

This keeps N constant at 105 names over 2929 sessions, which the correlation
graphs in Phase 3 require — a time-varying membership would otherwise produce
structurally NaN correlation windows. A properly time-varying universe, with
entries and exits handled explicitly in the graph construction, is the
methodologically superior treatment and is left as future work.
