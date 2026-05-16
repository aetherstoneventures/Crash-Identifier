# V6 Crash KPI Engine — Honest Scorecard (v6.0.0-alpha)

> This document reports the **actual** out-of-sample performance of the
> v6 Crash KPI Engine on the pre-declared validation protocol
> (`docs/CRASH_KPI_ENGINE_DESIGN.md`, section 6).
>
> Per the honesty contract: results are reported **as observed**, not
> as wished. Failures of the kill-criteria are stated up front.

## Configuration tested

| Setting | Value |
|---|---|
| Crash threshold `x_pct` | **10.0 %** |
| Horizon `horizon_td` | **63 trading days** |
| Walk-forward folds | `< 1999`, `< 2005`, `< 2012`, `< 2020` |
| BLIND cutoff | `2021-01-01` (single-shot fit < cutoff, score thereafter) |
| Engines | Anomaly + HMM regime + Analog k-NN + Causal connectedness + Bayesian aggregator |
| Gate | L1 z ≥ 1.5 **AND** L2 z ≥ 1.5 **AND** L3 dd ≥ 2 % **AND** posterior ≥ 0.60 **AND** confidence ≥ 0.50 |

Artefact: `data/v6_artifacts/v6_validation_x10_h63.json`.

---

## BLIND ( 2021-01-01 → 2026-04-28 ) — **PRIMARY VERDICT**

| Metric | Value |
|---|---|
| Days scored | 1 251 |
| Gate fires | **0** |
| Crash days (10 % drop within 63 d) | 128 |
| Precision | n/a (no fires) |
| Recall | **0.000** |
| Fire rate | 0.00 % |

### Calibration (BLIND)

| Posterior bucket | Predicted mean | Empirical hit rate | n |
|---|---|---|---|
| (0.4, 0.5] | 0.464 | 5.5 % | 867 |
| (0.5, 0.6] | 0.544 | **20.8 %** | 384 |

### Kill-criteria result: ❌ **FAIL**
- `gate fire pct = 0.00 % < min 0.10 %` (no fires at all)

---

## Walk-forward pooled (1999-01-01 → 2026-04-28)

| Metric | Value |
|---|---|
| Days scored | 6 452 |
| Gate fires | 8 |
| Pooled precision | 0.000 |
| Pooled recall | 0.000 |

| Fold | Cutoff | Days | Fires | Crash days | Notes |
|---|---|---|---|---|---|
| 1 | < 1999 | 1 420 | 0 | 0 | crash labels empty for this window — extractor edge case |
| 2 | < 2005 | 1 673 | 0 | 0 | same as above |
| 3 | < 2012 | 1 870 | 0 | 128 | model never reached gate during GFC under this fold's training set |
| 4 | < 2020 | 1 489 | 8 | 158 | 8 fires were all **March 18-27 2020** — during, not before, COVID crash |

### Calibration (pooled walk-forward)

| Posterior bucket | Predicted | Empirical | n |
|---|---|---|---|
| (0.4, 0.5] | 0.466 | 12.5 % | 2 420 |
| (0.5, 0.6] | 0.550 | **0.6 %** | 3 118 |
| (0.6, 0.7] | 0.611 | 0.0 % | 914 |

The calibration is **inverted** in the (0.5, 0.6] bucket on walk-forward
(predicting 55 % when truth is 0.6 %). The aggregator is over-confident
in the mid-range. This is the central diagnostic.

---

## What this means

The v6 alpha **fails the pre-declared kill criteria on BLIND.** Stating
it plainly: this engine does **not** outperform a null model on the
2021-2026 out-of-sample period. Specifically:

1. **0 % recall on BLIND.** The gate never fired in 5 years, including
   the 2022 bear market (−25 % peak-to-trough).
2. **Mis-calibration.** Posteriors in the 0.5-0.6 band correspond to an
   actual 20 % hit rate on BLIND — the model is *under-confident* in
   that band but the gate threshold (0.60) is just above it, so it never
   triggers.
3. **The 2020 COVID fires happened during the crash, not before** —
   useful for confirmation but not for forecasting.

## Root causes (diagnosed, not yet fixed)

1. **Feature warm-up is too late.** Several v6 features (`vix_term_structure`,
   `skew_z`, `nfci`) are entirely missing in the database; many others
   only start ~2007. The analog engine's training pool is only 705 dates
   after the 252-day horizon trim. The model is data-starved.
2. **Gate threshold is too strict relative to the aggregator's natural
   scale.** The aggregator with equal weights and 4 noisy 0-1 signals
   has posterior support concentrated in [0.4, 0.7]. The 0.60 cutoff is
   reachable only when all four engines simultaneously emit pressure ≥
   ~0.7 — a high bar.
3. **Walk-forward folds before 2010 are not viable** with current
   feature coverage; pool size collapses to 0 and the analog/causal
   engines emit NaN.

## What is **not** broken

- Pipeline orchestration works end-to-end on data ≥ 2007.
- The COVID-March-2020 case study showed all five engines and the gate
  behave as designed when data is available — the gate fired Mar 18-27
  during the COVID crash.
- The validation harness, kill-criteria check, and per-engine pressure
  decomposition are all in place for the next iteration.

## Path forward (post-alpha)

1. **Backfill missing FRED series** (NFCI, SKEW, VIX9D, ICE BofA IG/HY).
   Expected to expand the analog pool from 705 → ~5 000 dates.
2. **Learn per-engine weights** from walk-forward calibration error
   instead of equal weighting.
3. **Re-tune gate threshold** after recalibration (likely 0.50, not 0.60).
4. **Drop folds with insufficient training history** from the official
   protocol and replace with `< 2010, < 2015, < 2020`.
5. **Add a second BLIND on extended data** when 2026-2027 finishes.

These changes are deferred to v6.1; the alpha tag captures the current
honest state.
