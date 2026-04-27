# V5.1 Honest Findings — What Worked, What Didn't

## TL;DR

**v5.1 ensemble (predictive + coincident + bottom-finder) was BUILT but NOT SHIPPED.**
The only improvement that survived honest validation: **simplifying v5's re-entry rule**, not the new ML.

| Strategy                               | Full CAGR | Blind CAGR | MaxDD | Sharpe |
|----------------------------------------|----------:|-----------:|------:|-------:|
| Buy & Hold NASDAQ                      |      7.0% |      12.8% | -77.9%| 0.39 |
| **v5 (original)** — committed baseline |     14.7% |      21.4% | -15.2%| 1.07 |
| **v5 + faster MA50 re-entry** ← ship   | **16.3%** |  **23.0%** | -15.2%| 1.16 |
| v5.1 ensemble (predictive label)       |      7.2% |       2.1% | -31.6%| 0.57 |

Improvement of the simpler change: +1.6 pp CAGR full window, +1.6 pp blind.
Direction-consistent (both windows), MaxDD identical → real, generalizable.

## What was attempted

1. **New leading-indicator features** (added to DB):
   - VIX 3-month (`VXVCLS`) for term-structure ratio
   - VXO old VIX (1986+) to extend term-structure history
   - AAA-10Y credit spread (1983+ daily)
   - NFCI subindexes: leverage, risk, credit (1971+ weekly)
   - KCFSI Kansas City Financial Stress (1990+ monthly)
   - UMich consumer sentiment (1952+ monthly)
   - 10Y breakeven inflation, TIPS real yield (2003+ daily)

2. **Predictive label**: target = "crash starts within next 20 days" (instead of v5's coincident "in crash now"). Goal: shift alarm forward in time.

3. **Bottom-finder model**: separate XGBoost trained on "given drawdown ≥10%, will 60-day forward return > +10%?". WF AUCs: 0.72 / 0.85 / 0.98 / 0.88.

4. **Pareto sweep** of entry threshold {0.25 … 0.60} with all metrics measured on TUNE (1999-2020) and validated on BLIND (2021-2026).

## Why v5.1 ensemble failed (honest)

The predictive model showed +23-32d median lead from peak on TUNE — but **only -96 to -157d (i.e. AFTER peak) on BLIND**. The lead-time signal **did not generalize** to held-out crashes.

Mechanism: tune crashes (1987, 2000, 2008, 2020) are sharp, vol-driven. Blind crashes (2022 bear market, 2024-12 correction) are slow grinding declines with different feature signatures. The predictive XGBoost overfit to the sharp-crash regime.

This was a textbook **regime-shift overfit**. Reporting v5.1 as an improvement would have been the kind of misleading result you ruled out.

## Why the bottom-finder failed standalone

In every backtest variant, MA50-based re-entry fired before the bot signal hit any threshold. `bot_or_ma50 ≡ ma50_only` to four decimal places. `bot_only` drastically underperformed (full 11.6% CAGR; blind 6.9%) because the model was too conservative — it kept the strategy in cash through legitimate recoveries.

The model trains cleanly (good WF AUCs), but its operating threshold doesn't add information beyond what a 50-day moving average already contains.

## What we shipped

`data/v5_bot_reentry_config.json` — but only the rule change is used:
```
re-enter when: alarm-off ≥ 5 days AND price > 50-day MA
(removed:    10-day delay, 5% rally-off-low requirement)
```

This is the only honest, validated improvement.

## What we did NOT ship

- `models/v5_1/v5_1_final.pkl` — kept on disk for inspection/research, NOT loaded by the dashboard or refresh pipeline.
- `data/alarm_config_v5_1.json` — likewise inspection-only.
- The new FRED features remain in the DB (no harm, useful research material).

## Decisions encoded in this scorecard

- v5 (committed model `models/v5/v5_final.pkl`) remains the production alarm.
- Re-entry rule simplified to MA50 + 5-day alarm-off (no bottom-finder, no rally requirement).
- v5.1 artifacts retained on disk but explicitly NOT promoted.
- 17 historical crashes, 2 in blind window — single-trial uncertainty acknowledged; one regime-shift example sufficient to fail v5.1.

## Files added in this iteration

- `scripts/data/fetch_v5_1_features.py`              — pulls 10 new FRED series
- `scripts/training/train_v5_1.py`                   — trains 3 models, nested WF
- `scripts/utils/v5_1_backtest.py`                   — two-sided backtest
- `scripts/utils/v5_1_pareto.py`                     — Pareto sweep (entry ∈ [0.25,0.60])
- `scripts/utils/test_bot_reentry.py`                — bot vs original re-entry
- `scripts/utils/test_bot_isolation.py`              — bot value isolation test
- `data/v5_bot_reentry_config.json`                  — chosen re-entry params
- `docs/V51_HONEST_SCORECARD.md`                     — this document

## Caveats

- Blind window contains only 2 crash events; statistical power is limited.
- The new MA50 re-entry was tuned on TUNE and validated on BLIND, but the search space is small (3 vs 5 vs 10 day delays). Mild parameter optimism possible.
- All metrics are pre-tax, simulate liquid NASDAQ exposure with 5 bps round-trip cost; real-world slippage and dividend handling differ.
- Statistical model V3 contributes 50% of the blended signal; its hand-coded weights are an untracked source of selection bias inherited from prior commits.
