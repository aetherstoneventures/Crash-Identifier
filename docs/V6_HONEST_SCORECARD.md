# V6 Crash KPI Engine — Honest Scorecard (v6.1)

> Results are reported **as observed**, not as wished. Failures are stated up
> front. Reproduce with:
>
> ```bash
> python scripts/v6/validate.py both --x 10 --h 63
> ```
>
> Artefact: `data/v6_artifacts/v6_validation_x10_h63.json`

---

## ⚠️ Read this before the numbers: the BLIND set is no longer blind

The protocol (design doc §6) requires a **single-shot** evaluation on data
from 2021-01-01 with **no retuning**.

**That condition has been broken.** During v6.1 development, the 2022 gate
diagnostics were inspected to find out why the gate never fired, and the
archetype decomposition of Layer 1 was designed in response to what that
inspection showed. The 2021–2026 window therefore functioned as a development
set, and its results below are **optimistically biased by an unknown amount**.

Two things partially mitigate this, and neither makes the result clean:

- The archetype split was motivated by a *structural* argument that is
  verifiable independently of the model — 2022 was a rate-led drawdown with
  contained credit spreads and record-low unemployment, which is a fact about
  the macro record, not a fit to the outcome series.
- No parameter was tuned to maximise a BLIND metric. Thresholds are fitted on
  each fold's training window only.

**What a clean verdict would require:** re-running the now-frozen
configuration on either data after 2026-08-19 as it accrues, or a window that
these repairs never touched. Until then, treat the walk-forward folds as the
more trustworthy evidence, and read the BLIND row as an upper bound.

---

## Configuration

| Setting | Value |
|---|---|
| Crash threshold `x_pct` | **10.0%** |
| Horizon `horizon_td` | **63 trading days** |
| Index | Nasdaq Composite (`nasdaq_close`, 1971-02-05 → 2026-08-19) |
| Walk-forward folds | `< 1999`, `< 2005`, `< 2012`, `< 2020` |
| BLIND cutoff | `2021-01-01` (single-shot fit, then score — see caveat above) |
| Engines | Anomaly + HMM regime + Analog k-NN + Causal connectedness + calibrated log-odds aggregator |
| Gate | posterior AND confidence AND L1 AND L2 AND L3, all simultaneously |
| Gate thresholds | fitted on each fold's training window to a ~1% fire rate |
| Backtest | exit on fire, hold 21d, 5bps slippage per switch |

---

## BLIND (2021-01-01 → 2026-08-19)

| Metric | Value |
|---|---|
| Days scored | 1 466 |
| Gate fires | 105 (**7.16%**) |
| Base rate P(maxDD ≥ 10% in 63d) | 0.375 |
| **Precision** | **0.776** |
| Recall | 0.144 |
| **Precision lift vs base rate** | **2.07×** |
| **Median lead to the −10% crossing** | **38.5 trading days** |
| Brier | 0.2398 (skill −0.023) |
| Reliability slope | 0.630 |

### Decision backtest (BLIND)

| | Strategy | Buy & hold |
|---|---|---|
| CAGR | 11.16% | 13.07% |
| **MaxDD** | **−26.5%** | **−36.4%** |
| Sharpe | 0.67 | 0.67 |
| Time in market | 76.7% | 100% |
| Switches | 18 | — |

### Kill criteria — BLIND: **PASS (5/5)**

| # | Criterion | Result |
|---|---|---|
| 1 | Reliability slope ∈ [0.5, 1.5] | 0.630 ✅ |
| 2 | CAGR ≥ B&H − 2pp | −1.91pp ✅ (margin: 0.09pp) |
| 3 | MaxDD ratio ≤ 1.10 | 0.729 ✅ |
| 4 | No engine carries all weight | max 0.400 ✅ |
| 5 | Fire rate ∈ [0.10%, 10%] | 7.16% ✅ |

Criterion 2 passes by **0.09 percentage points**. That is a hair, not a
margin, and it should be read as "roughly matched buy-and-hold returns while
cutting the drawdown by a quarter" — which is what the design document called
the realistic median case (§11) — not as a demonstration of edge.

---

## Walk-forward folds — **the more trustworthy evidence**

| Fold | Window | Days | Fires | Base rate | Precision | CAGR Δ | MaxDD ratio | Kill |
|---|---|---|---|---|---|---|---|---|
| 1 | 1999–2005 | 1 566 | 0 | 0.603 | — | +0.00pp | 1.000 | ❌ |
| 2 | 2005–2012 | 1 825 | 0 | 0.327 | — | +0.00pp | 1.000 | ❌ |
| 3 | 2012–2020 | 2 088 | 10 (0.48%) | 0.176 | 0.000 | −2.68pp | 1.002 | ❌ |
| 4 | 2020–2026 | 1 728 | 61 (3.53%) | 0.394 | **1.000** | **+2.16pp** | **0.827** | ❌ |
| **Pooled** | 1999–2026 | 7 206 | 71 (0.99%) | 0.359 | **0.859** | −0.26pp | 1.000 | ❌ |

### Why each fold fails

- **Folds 1 and 2 — gate never fires (criterion 5).** The engines are fit on
  ≤ 1999 and ≤ 2005 data. Credit and financial-conditions coverage before the
  mid-1980s is thin even after the FRED backfill, so the archetype composites
  rarely clear their training quantiles out-of-sample. Fold 1's base rate of
  **0.603** is also worth noting: across 1999–2005 the Nasdaq spent most days
  within 63 days of a 10% drawdown. A gate is close to meaningless when the
  event is the norm — the useful question there is not "will it fall" but "how
  far", which this configuration does not answer.
- **Fold 3 — CAGR −2.68pp (criterion 2).** Ten fires across 2012–2020, none of
  which preceded a 10% drawdown. This is the cost of a false-alarm regime:
  sitting out a strong bull market.
- **Fold 4 — reliability slope 0.002 (criterion 1).** Precision is a perfect
  1.000 on 61 fires and the backtest is genuinely good (+2.16pp CAGR, drawdown
  cut to 0.827×), but the posterior is nearly flat against outcomes across its
  range, so the *probabilities* are not trustworthy even though the *ranking*
  is. A usable trigger with an untrustworthy probability attached.
- **Pooled — reliability slope −0.415 (criterion 1).** Pooling folds with very
  different base rates (0.176 to 0.603) mixes populations, and the slope
  inverts. Pooled precision is nevertheless **0.859 at 2.39× lift** on 71
  fires.

---

## Calibration

BLIND reliability is monotone and roughly sensible:

| Posterior bucket | Predicted | Empirical | n |
|---|---|---|---|
| (0.1, 0.2] | 0.189 | 0.275 | 167 |
| (0.2, 0.3] | 0.241 | 0.357 | 686 |
| (0.3, 0.4] | 0.330 | 0.372 | 331 |
| (0.4, 0.5] | 0.488 | **0.592** | 157 |
| (0.5, 0.6] | 0.537 | 0.306 | 62 |

The posterior discriminates — the 0.4–0.5 bucket resolves at 59% against a
37.5% base rate — but it is **under-confident in the middle and unreliable in
the top bucket**, where the sample is thin (n = 62). Brier skill is slightly
negative (−0.023): the posterior is *not* better than simply quoting the base
rate as a probability, even though it ranks days usefully. Both facts are
true at once and neither should be dropped when quoting this work.

---

## Engine weights (BLIND fit)

| Engine | Skill | Weight |
|---|---|---|
| Anomaly | 0.037 | 0.400 |
| Regime | 0.000 | 0.400 |
| Analog | 0.638 → 0.100 after embargo | 0.100 |
| Causal | 0.000 | 0.100 |

Before the analog embargo (post-mortem §10b) the analog engine measured 0.638
skill and took 86% of the weight — which would itself have tripped kill
criterion 4. That skill was largely the engine retrieving its own temporal
neighbours. After the embargo its honest contribution is small, and weight
shifts to the anomaly and regime engines. Measured skill for regime and causal
is 0.000, i.e. neither beats the base rate on log-loss; they are held at the
floor weight rather than dropped, and their real contribution is to the
confidence and archetype signals rather than to the posterior.

---

## What is established, and what is not

**Established**

- The architecture works end-to-end on 55 years of data, and the five kill
  criteria are all actually computed (the alpha checked one of five).
- Gate fires carry real information: pooled precision 0.859 at 2.39× lift,
  with a ~38-day median lead.
- Drawdown reduction is consistent where the gate fires at all: 0.827× (fold
  4) and 0.729× (BLIND).
- Every defect in [V6_POSTMORTEM.md](V6_POSTMORTEM.md) is covered by a
  regression test (`tests/test_v6/`, 32 tests).

**Not established**

- **That this beats buy-and-hold.** It does not. BLIND CAGR is 1.91pp *below*
  B&H; pooled is 0.26pp below. The case for it is drawdown reduction at
  comparable return, not excess return.
- **That the probabilities are trustworthy.** Negative Brier skill on BLIND
  and a failing reliability slope on three of four folds say they are not yet.
- **That it generalises to pre-2010 regimes.** Folds 1 and 2 produce no fires
  at all.
- **That the BLIND result is clean.** See the disclosure at the top.

---

## Next steps, in priority order

1. **Re-freeze and re-blind.** Tag the current configuration, then evaluate it
   only on data after 2026-08-19 as it accrues. This is the single most
   valuable outstanding item; everything below is secondary to it.
2. **Fix calibration, not ranking.** The ranking works and the probabilities
   do not. Fit isotonic regression on walk-forward residuals rather than
   adding features.
3. **Give the early folds usable macro history.** Credit and
   financial-conditions coverage before 1986 is the binding constraint on
   folds 1–2.
4. **Report by archetype.** With the type tag now emitted, precision and lead
   time should be broken out per archetype; a detector that is excellent on
   credit-led and useless on shock-led should say so.
5. **Re-examine the fold-1 regime.** A 0.603 base rate means the framing
   ("will a 10% drawdown occur") carries little information in that period.
   Severity or timing is the better question there.
