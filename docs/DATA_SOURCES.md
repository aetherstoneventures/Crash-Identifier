# Data Sources and Point-in-Time Rules

Every input to the v6 engine, where it comes from, and what a given trading
day is allowed to know about it.

All series are pulled by `scripts/data/backfill_fred.py` into
`data/market_crash.db` (table `indicators`, 14 394 rows,
1971-02-05 → 2026-08-19). The script is idempotent and prints a before/after
coverage report.

---

## Point-in-time rules

A backtest is only honest if each row contains what a reader could actually
have known that morning. Three rules cover every series:

| Rule | Applies to | Behaviour | Max staleness |
|---|---|---|---|
| `close` | daily market data | value known at that day's close, no shift | 7 days |
| `lag` | weekly releases on a fixed schedule | shifted forward by the publication lag | 14 days |
| `vintage` | monthly macro | stamped on its **first release date** from ALFRED | 45 days |

**Why `vintage` matters.** The legacy database stamped monthly macro on the
observation date, so April 2020's unemployment rate sat on 2020-04-01 — five
weeks before its 2020-05-08 publication — and stored the *revised* 14.8 rather
than the 14.7 first print. Both are look-ahead. ALFRED vintages fix both.

**Why staleness is bounded.** Forward-filling without limit turns a
discontinued series into a long run of one value, which is indistinguishable
from the constant-fill artefacts the quality screen exists to reject. Past its
window a value becomes NaN, and downstream engines treat it as missing.

**Pre-vintage history.** ALFRED's vintage coverage starts later than the
observation history for some series (UMCSENT vintages begin 1998 though the
series runs from 1952). Older observations fall back to latest-vintage values
shifted by the series' own median release lag. Those are revised rather than
first-print figures — accepted in exchange for keeping 40+ years of history,
and noted here because it is a real compromise.

---

## Series catalogue

### Equity indices

| Column | FRED | Rule | Coverage | Notes |
|---|---|---|---|---|
| `nasdaq_close` | `NASDAQCOM` | close | 1971-02-05 → | **Primary price series** |
| `sp500_close` | `SP500` | close | 2016-08-22 → | Rolling 10-year licence window — secondary only |

> `SP500` on FRED carries **only a rolling 10-year window**. v6.0.0-alpha
> hard-coded it as the price column, which truncated every price-derived
> feature and the crash labels themselves to 2016+. `resolve_price_column()`
> now picks the longest quality-passing series by evidence. This is the single
> most consequential defect found in the post-mortem.

### Volatility

| Column | FRED | Rule | Coverage | Notes |
|---|---|---|---|---|
| `vix_close` | `VIXCLS` | close | 1986-01-02 → | Spliced with VXO to reach back to 1986 |
| `vxo_close` | `VXOCLS` | close | 1986-01-02 → 2021-09-23 | Discontinued; historical leg of the splice |
| `vxv_close` | `VXVCLS` | close | 2007-12-04 → | VIX3M; enables the term-structure feature |

Pre-repair, `vix_close` was the constant **17.24** for every row before 1990.

### Rates and curve

| Column | FRED | Rule | Coverage |
|---|---|---|---|
| `yield_10y` | `DGS10` | close | 1971-02-05 → |
| `yield_10y_3m` | `T10Y3M` | close | 1982-01-04 → |
| `yield_10y_2y` | `T10Y2Y` | close | 1976-06-01 → |
| `t10yie` | `T10YIE` | close | 2003-01-02 → |
| `dfii10` | `DFII10` | close | 2003-01-02 → |

### Credit

| Column | FRED | Rule | Coverage |
|---|---|---|---|
| `baa_10y_spread` | `BAA10Y` | close | 1986-01-02 → |
| `aaa_10y_spread` | `AAA10Y` | close | 1983-01-03 → |

> ICE BofA OAS series (`BAMLH0A0HYM2`, `BAMLC0A0CM`) are now restricted to a
> rolling ~3-year window on FRED, so they cannot support walk-forward folds.
> `BAA10Y` is the longest free credit-spread history available and is used as
> the high-yield proxy.

### Financial conditions (weekly, Wednesday release for Friday week-ending)

| Column | FRED | Rule | Coverage |
|---|---|---|---|
| `nfci` | `NFCI` | lag 5d | 1971-02-05 → |
| `anfci` | `ANFCI` | lag 5d | 1971-02-05 → |
| `nfci_leverage` | `NFCILEVERAGE` | lag 5d | 1971-02-05 → |
| `nfci_risk` | `NFCIRISK` | lag 5d | 1971-02-05 → |
| `nfci_credit` | `NFCICREDIT` | lag 5d | 1971-02-05 → |
| `stlfsi` | `STLFSI4` | lag 5d | 1994-01-05 → |

All five NFCI columns held **zero rows** before the repair — the alpha's
scorecard correctly identified `nfci` as missing.

### Labour, commodities, dollar

| Column | FRED | Rule | Coverage |
|---|---|---|---|
| `initial_claims` | `ICSA` | lag 5d | 1971-02-05 → |
| `oil_wti` | `DCOILWTICO` | close | 1986-01-02 → |
| `dollar_twi` | `DTWEXBGS` + `DTWEXM` | close | 1973-01-02 → |

`dollar_twi` splices the current broad index onto the discontinued major-
currencies index, level-matched at the overlap so the join has no artificial
jump.

### Monthly macro — ALFRED vintages

| Column | FRED | Rule | Coverage |
|---|---|---|---|
| `unemployment_rate` | `UNRATE` | vintage | 1971-02-05 → |
| `cpi` | `CPIAUCSL` | vintage | 1971-02-05 → |
| `m2_money_supply` | `M2SL` | vintage | 1971-02-05 → |
| `industrial_production` | `INDPRO` | vintage | 1971-02-05 → |
| `umcsent` | `UMCSENT` | vintage | 1971-02-05 → |
| `kcfsi` | `KCFSI` | vintage | 1992-08-25 → |

---

## Quarantined columns

`src/v6/features/quality.py` blocks these from ever becoming features. Each
entry is auditable and reversible — if a column is repopulated from a real
source, delete its entry.

| Column | Reason |
|---|---|
| `put_call_ratio` | Synthetic noise: mean 1.0052, std 0.033, lag-1 autocorrelation 0.086, and no reaction on 1987-10-19, 2008-10-10 or 2020-03-16. Real CBOE equity put/call begins 1995 and exceeds 1.2 in panics. No free full-history source exists. |
| `in_crash`, `pre_crash_30d`, `pre_crash_60d` | Target labels. Using them as features is leakage. |
| `data_quality_score` | Bookkeeping, not market data. |
| `credit_spread_bbb` | Constant fill for 93.2% of its history. |
| `margin_debt` | Constant fill for 93.2% of its history. FINRA margin debt needs a separate scraper. |
| `lei` | Constant fill for 14.3% of its history. |

Automated screens (constant-fill runs, degenerate variance, implausible
noise, insufficient history) catch the last three without a hard-coded entry.

---

## Features not built, and why

| Feature | Reason |
|---|---|
| `put_call_z` | Source column quarantined; no free full-history replacement. |
| `skew_z` | CBOE SKEW is not on FRED and `v6_skew` was never populated. Tail-risk pricing is partly covered by `vix_term_structure`. |
| `margin_debt_z` | Source column quarantined. |

Listed in `UNAVAILABLE_FEATURES` in `src/v6/features/builder.py` so the
omissions stay visible rather than forgotten.

---

## Known legacy wart

The SQLAlchemy `Indicator` model in `src/utils/database.py` declares 25
columns while the actual SQLite table has ~85. The v6 engine reads the table
directly with `sqlite3`/pandas and does not depend on the ORM; the drift only
affects legacy dashboard paths. Anything new should read through
`FeatureBuilder.load_raw()`.
