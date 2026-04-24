# V5 Crash Predictor — Honest Scorecard

_Generated 2026-04-24. Supersedes v4._

## What changed vs v4

| Problem in v4 | Fix in v5 |
|---|---|
| Pre-2016 SP500 was **monthly data padded to daily** (11 unique values/year). All 1982-2015 drawdown features were garbage. | Switched primary equity series to **NASDAQ Composite daily (1971+)** — 250+ unique values/year since 1987. |
| 24 FRED-placeholder holiday rows (`sp500=4515.77`) caused phantom -34% one-day crashes and a 56.8% false alarm on Presidents' Day 2026. | `scripts/data/purge_bad_rows.py` surgically detects round-trip spike anomalies and fixed-holiday rows; run automatically before training. |
| Alarm hysteresis parameters (`entry`, `exit`, `min_dur`, `max_dur`, `mf_min`) were grid-searched on the **same 2005-2026 window used to report the 50% / 100% / 150d headlines**. Classic parameter overfitting. | **Nested walk-forward**: XGBoost tuned via 4 expanding folds; alarm params grid-searched **only on 1999-2020**; 2021-2026 is a strictly blind test window. |
| Reported 150-day median lead was measured from **trough**, inflated by 2006 pre-GFC alarms and alarm-system re-triggers. | Lead now measured from **market peak**, which is the only number an allocator can act on. |
| Event-level precision inflated by bundling long GFC/2022 alarms into 1 "event" = 1 TP. | Reports both **event-level and day-level** precision+recall. |
| 4 features missing that matter for macro/geopolitical regimes. | Added `BAA-10Y` (credit), `DCOILWTICO` (oil), `DTWEXBGS` (dollar), `USEPUINDXD` (EPU daily). |

## Honest Blind-Test Results (2021-2026, never seen during any tuning)

**Overall walk-forward day-level AUC (1999-2026):** 0.788

**Blind test window (2021-01-01 → 2026-04-23) — crashes: 2022 bear, 2024-12 → 2025-04 correction**

| Rule | Alarm days % | Day Precision | Day Recall | F1 |
|---|---|---|---|---|
| VIX > 25 (sustained 10d) | 5.6% | 92.0% | 19.4% | 0.32 |
| Drawdown > 5% | 43.2% | 50.0% | 81.1% | 0.62 |
| VIX > 25 OR DD > 5% | 44.3% | 49.5% | 82.3% | 0.62 |
| VIX > 30 AND DD > 10% | 4.7% | 82.5% | 14.6% | 0.25 |
| **v5 hysteresis** | **26.4%** | **70.7%** | **69.9%** | **0.70** |

**v5 is the only rule with F1 > 0.65 and a balanced precision/recall profile.** It beats every naive baseline on F1.

Event-level metrics on blind test:
- Event precision: 75% (3 of 4 alarms overlapped a real crash)
- Event detection: 100% (both crashes triggered v5)
- **Lead from peak: –73 days** (median) → v5 is a **fast regime detector, not a forecaster**. Any use-case must accept that.

## Limitations (don't remove from the docs)

1. **2 crashes in the blind test window** = metrics have wide uncertainty bands. Do not treat 75% event precision as a point estimate.
2. **Lead is negative** — the model is reactive. If you need early warning, this tool is insufficient on its own.
3. **March 2026 correction (-8.7%)** was NOT a crash by the ≥15% / ≥30TD definition, but v5's signal peaked at 45.6% on March 27 — one day shy of triggering a full alarm (entry=0.45, min_dur=20d). Borderline false positive avoided only by the duration filter.
4. The **StatV3 half of the blended signal is a manual scoring function**; changes to its thresholds are an untracked overfitting vector. A future v6 should retire StatV3 or fold its logic into the learned model.
5. **Lookahead-bias check passed** (features recomputed on truncated history = features on full history for 2005 era). But the crash *label* is a lookahead function by design — this is fine for training, invalidating for live day-by-day signal, which is why the model's output is the blended probability, not the label.

## Current state (2026-04-23)

```
SP500 = 7108    NASDAQ = 24438    VIX = 19.3    EPU = 481
StatV3 = 9.2%   GBM_v4 = 5.2%   v5 = 12.5%
```

**v5 says: NO CRASH SIGNAL.** Threshold is 45%; reading is 12.5%. Consistent with VIX below 20 and credit/NFCI loose. Elevated EPU (481) is noted in the feature set but outweighed by benign price action and volatility.

## Files

- `scripts/training/train_v5_walkforward.py` — honest training script (reproduces this scorecard)
- `scripts/data/purge_bad_rows.py` — data sanity guard (run before training)
- `scripts/data/fetch_v5_features.py` — pulls NASDAQ/credit/oil/dollar/EPU from FRED
- `models/v5/v5_final.pkl` — final production model
- `data/alarm_config_v5.json` — alarm parameters + honest metrics
