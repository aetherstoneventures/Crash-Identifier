# v6 Negative Result — SHELVED

## Verdict: **FAILED** kill criteria. v5 remains BENCHMARK (`v5-BENCHMARK` tag).

## Hypothesis tested

Add three new feature classes — all mathematically motivated and used by institutional risk desks:
1. **Options-implied tail risk**: CBOE SKEW, SKEW×VIX, VIX9D/VIX term-structure ratio
2. **Cross-sectional breadth**: RSP/SPY (equal-weight vs cap-weight 60d), IWM/SPY (small vs large 60d)
3. **Sector rotation + cross-asset**: defensive vs cyclical sector log-return spread, HYG-SPY 20d, TLT-SPY 60d, GLD-SPY 60d

Combined with v5's coincident model into an ensemble (max of predictive-blended, coincident-blended). Trained under nested walk-forward (4 folds 1999-2026; 2021-2026 held out blind).

## Result

### TUNE (1999-2020) — looked promising
- Median lead from peak: **+24 days** (vs v5: ~+4d AFTER peak)
- ev_det 87%, day_precision 81%, ev_prec 93%

### BLIND (2021-2026) — collapsed
| Metric | v5 (benchmark) | v6 (candidate) | Δ |
|---|---:|---:|---:|
| day_precision | 95.8% | 45.0% | **-50.9 pp** |
| ev_det | 100% | 100% | +0 |
| median lead | -9d | +148d | +157d |
| alarm-day pct | 26.0% | 47.4% | +21.4 pp |
| ev_prec | 100% | 56.5% | -43.5 pp |
| **Backtest CAGR** | 23.0% | 10.0% | **-13.0 pp** |
| Backtest Sharpe | 1.45 | 0.84 | -0.61 |
| Backtest MaxDD | -13.2% | -17.9% | -4.7 pp |

Full window (2000-2026) backtest: v5 16.3% CAGR / Sharpe 1.16 / MaxDD -15.2% vs v6 8.1% / 0.58 / **-52.1%**. v6 underperforms even buy-and-hold (7.0% / -77.9%).

### Kill criteria (4 of 4 must pass)
| Criterion | Δ (v6 - v5) | Threshold | Status |
|---|---:|---:|---|
| day_precision | -50.9 pp | ≥ -5 pp | **FAIL** |
| ev_det | +0 pp | ≥ -5 pp | PASS |
| median lead | +157 d | ≥ -5 d | PASS |
| backtest CAGR | -13.0 pp | ≥ -2 pp | **FAIL** |

## Why it failed

Same overfitting mode as v5.1, with different features:

1. **Predictive label is fundamentally low-precision.** Labelling positive the 20 days *before* every crash means most positive samples look statistically similar to many normal volatile periods. XGBoost minimises in-sample loss by reducing the decision threshold — increasing recall at the cost of precision. Out-of-sample, the precision drop dominates.

2. **2022 false-positive spiral.** A single early alarm in 2021-Q4 caught the 2022 bear market 148 days "in advance," but at the cost of being in cash from late 2021 through the entire 2023-2024 rally.

3. **Tail-risk features are coincident, not predictive, in equity drawdowns.** SKEW/VIX9D contain real information about *current* tail demand, but historically they spike with — not before — equity stress. Top features confirm: equity_drawdown (0.20), regime_num (0.08), equity_return_63d (0.05) all dominate over skew_x_vix (0.03) and rsp_spy_60d (0.03).

4. **Breadth proxies are too slow.** RSP/SPY and IWM/SPY 60-day spreads turn months *after* peaks, not before.

5. **Single-trial blind window.** With only 2 crashes in 2021-2026 (2022 bear, 2024-12 correction), one false-positive cluster from late 2021 swamps any genuine signal.

## What this confirms

The structural lesson, now demonstrated twice (v5.1 and v6) with different feature sets:

> **Equity crashes have heterogeneous causes that don't repeat. Models trained to "predict" earlier than coincident detection systematically overfit to past mechanisms and lose precision out-of-sample.**

This isn't an indictment of the data sources — SKEW, breadth, sector dispersion, and cross-asset spreads all carry real information. It's that they don't carry **forward-looking** information that survives blind validation in the equity-crash setting with our 17-event sample.

## What stays

- Production: **v5 (commit 925d8b7, tag `v5-BENCHMARK`)** — unchanged.
- Data: 18 new columns in `indicators` table (`v6_skew`, `v6_spy`, …, `v6_vix9d`). Harmless — v5 doesn't read them.
- DB predictions: `v6/v6_pred/v6_bot` rows purged. Only `v5/GBM_v4/StatV3_phase3` remain.
- Filesystem: `models/v6/` removed (gitignored anyway).
- Research scripts retained for reproducibility:
  - `scripts/data/fetch_v6_features.py`
  - `scripts/training/train_v6.py`
  - `scripts/utils/v6_kill_or_promote.py`
  - `data/v6_kill_verdict.json` (full verdict numbers)

## What we will NOT try again

- Predictive labels with a 20-day lookahead window — this is the third independent failure mode (v4 trough-based, v5.1 K=20, v6 K=20).
- Adding more features hoping to "lift" a lagging predictive model — features can only help if the label is honest.

## What might be worth trying (future work, with same kill discipline)

- **Conformal prediction intervals** on v5's coincident probability — quantify uncertainty per-day instead of trying to forecast.
- **Multi-asset crash labelling** — train on US + Europe + Asia crashes simultaneously to grow the event count from 17 to ~50, may stabilise blind-window estimates.
- **Bottom-finder against price-only baselines** — drop the requirement to "beat MA50" and instead test "beat 30/200d MA crossover," which is a stronger benchmark than MA50 alone. Bot-only models so far underperform either MA rule.
- **Volatility-regime gating** — apply v5 only when realized vol exceeds a threshold; in calm regimes, default to long. Mechanically reduces alarm-day pct in mid-bull windows.

None of these are in scope for now. **v5 stays.**
