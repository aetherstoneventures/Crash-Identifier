# V6 Crash KPI Engine — Honest Scorecard (v6.1)

> Results are reported **as observed**, not as wished. Reproduce with:
>
> ```bash
> python scripts/v6/validate.py both --x 10 --h 63
> ```
>
> Artefact: `data/v6_artifacts/v6_validation_x10_h63.json`
> Frozen configuration: `data/v6_artifacts/frozen_config_v6.1.0.json`
> (hash `9745c43c2b0d33b7…`, lock date 2026-08-19)

---

## Verdict: every window FAILS its kill criteria

| Window | Fires | Precision | Lift | Lead | Kill | Failing criterion |
|---|---|---|---|---|---|---|
| Fold 1 (1999–2005) | 0 | — | — | — | ❌ | [5] gate never fires |
| Fold 2 (2005–2012) | 0 | — | — | — | ❌ | [5] gate never fires |
| Fold 3 (2012–2020) | 10 | 0.000 | 0.00× | — | ❌ | [2] CAGR −2.68pp vs B&H |
| Fold 4 (2020–2026) | 61 | **1.000** | 2.54× | 40d | ❌ | [1] reliability slope −0.071 |
| **Pooled** | 71 | **0.859** | 2.39× | 40d | ❌ | [1] reliability slope 0.419 |
| **BLIND (2021–2026)** | 116 | 0.750 | 2.00× | 37d | ❌ | [1] reliability slope 0.330 |

**Nothing here passes.** The gate ranks days usefully — pooled precision 0.859
at 2.39× the base rate, with a 40-day median lead — but the posterior is not a
trustworthy probability on any window, and on the two earliest folds the gate
does not fire at all.

### Why this is stricter than the previous draft

An earlier v6.1 build reported **BLIND 5/5 PASS**. That build had no final
recalibration stage, and its BLIND reliability slope happened to land at 0.630.
Adding Platt recalibration — chosen on walk-forward evidence, see
[below](#the-calibration-decision) — improved the pooled slope from an
*inverted* −0.415 to 0.419 and fixed folds 1–2, but moved BLIND to 0.330,
which fails.

Reverting the calibrator would restore the BLIND pass. **That would be
selecting a modelling choice on the holdout**, which is the exact error this
document warns about elsewhere, so it was not done. The decision rule was
fixed on walk-forward evidence and BLIND is reported as it came out.

---

## ⚠️ The BLIND set is not blind

The protocol (design doc §6) requires a single-shot evaluation from
2021-01-01 with **no retuning**. That condition was broken: during v6.1
development the 2022 gate diagnostics were inspected, and the Layer 1
archetype decomposition was designed in response. The 2021–2026 window
functioned as a development set and its numbers are **optimistically biased by
an unknown amount**.

Partial mitigation, which does not make it clean: the archetype split rests on
a structural fact about the macro record (2022 was a rate-led drawdown with
contained credit spreads and record-low unemployment), not on fitting the
outcome series; and no parameter was tuned to maximise a BLIND metric.

**This is now enforced rather than promised.** `src/v6/freeze.py` hashes every
decision-affecting setting, and `scripts/v6/holdout_eval.py` refuses to run if
the live configuration differs from the freeze, or if the evaluation window
does not lie strictly after the lock date:

```bash
python scripts/v6/holdout_eval.py --freeze data/v6_artifacts/frozen_config_v6.1.0.json --check-only
# Data available past the lock date: 0 trading days
# Not yet — need ~126 more trading days.
```

Treat the walk-forward folds as the evidence and the BLIND row as an upper
bound until that command produces a verdict.

---

## Configuration

| Setting | Value |
|---|---|
| Crash threshold `x_pct` | **10.0%** |
| Horizon `horizon_td` | **63 trading days** |
| Index | Nasdaq Composite (`nasdaq_close`, 1971-02-05 → 2026-08-19) |
| Walk-forward folds | `< 1999`, `< 2005`, `< 2012`, `< 2020` |
| Engines | Anomaly + HMM regime + Analog k-NN + Causal connectedness + calibrated log-odds aggregator |
| Posterior recalibration | Platt (2-parameter logistic), cross-fitted on training only |
| Gate | posterior AND confidence AND L1 AND L2 AND L3, simultaneously |
| Gate thresholds | fitted per fold on training data to ~1% fire rate |
| Backtest | exit on fire, hold 21d, 5bps slippage per switch |

---

## Detail by window

### BLIND (2021-01-01 → 2026-08-19)

| Metric | Value |
|---|---|
| Days scored | 1 466 |
| Gate fires | 116 (7.91%) |
| Base rate P(maxDD ≥ 10% in 63d) | 0.375 |
| Precision | 0.750 (**2.00× base rate**) |
| Recall | 0.154 |
| Median lead to the −10% crossing | **37 trading days** |
| Brier | 0.2474 (skill −0.056) |
| Reliability slope | **0.330** ❌ |

| Backtest | Strategy | Buy & hold |
|---|---|---|
| CAGR | 11.50% | 13.07% |
| **MaxDD** | **−25.7%** | **−36.4%** |
| Sharpe | **0.71** | 0.67 |
| Time in market | 72.9% | 100% |

Kill criteria: [1] ❌ slope 0.330 · [2] ✅ −1.57pp · [3] ✅ ratio 0.707 ·
[4] ✅ max weight 0.400 · [5] ✅ 7.91%.

The economics are the best of any window — drawdown cut by 29% with a slightly
better Sharpe — and the probabilities are the least trustworthy. Both are true.

### Walk-forward folds

| Fold | Window | Days | Fires | Base rate | Slope | Brier skill | CAGR Δ | MaxDD ratio |
|---|---|---|---|---|---|---|---|---|
| 1 | 1999–2005 | 1 566 | 0 | 0.603 | 1.156 ✅ | −0.672 | +0.00pp | 1.000 |
| 2 | 2005–2012 | 1 825 | 0 | 0.327 | 0.615 ✅ | +0.026 | +0.00pp | 1.000 |
| 3 | 2012–2020 | 2 088 | 10 | 0.176 | 0.619 ✅ | −0.035 | −2.68pp ❌ | 1.002 |
| 4 | 2020–2026 | 1 728 | 61 | 0.394 | −0.071 ❌ | −0.183 | +2.16pp | 0.827 |
| Pooled | 1999–2026 | 7 206 | 71 | 0.359 | 0.419 ❌ | −0.096 | −0.26pp | 1.000 |

- **Folds 1–2: silent.** The engines see only ≤1999 / ≤2005 data, where credit
  and financial-conditions history is thin even after the FRED backfill, so
  the archetype composites rarely clear their training quantiles. Fold 1's
  base rate of **0.603** is itself notable: across 1999–2005 the Nasdaq spent
  most days within 63 days of a 10% drawdown. A binary "will it fall" gate
  carries little information when falling is the norm.
- **Fold 3: false alarms cost return.** Ten fires, none preceding a −10%
  move, and sitting out part of a bull market costs 2.68pp of CAGR.
- **Fold 4: perfect precision, useless probabilities.** 61 fires, precision
  1.000, +2.16pp CAGR and drawdown cut to 0.827× — with a reliability slope of
  −0.071. The ranking is excellent and the calibration is absent.

---

## By crash archetype

The point of decomposing Layer 1 was to distinguish crash types rather than
average them away. Doing so surfaces a limitation a blended number hides.

**Pooled walk-forward**

| Archetype | Fires | Precision | Median lead |
|---|---|---|---|
| `credit_led` | **0** | — | — |
| `rate_led` | 1 | 1.000 | 63d |
| `shock_led` | 64 | **0.922** | 40d |
| `valuation_led` | 6 | 0.167 | 27d |

**BLIND**

| Archetype | Fires | Precision | Median lead |
|---|---|---|---|
| `credit_led` | **0** | — | — |
| `rate_led` | 2 | 0.500 | 63d |
| `shock_led` | 109 | 0.743 | 38d |
| `valuation_led` | 5 | 1.000 | 12d |

Two findings, both uncomfortable and both worth stating:

1. **`credit_led` has never fired — not once, in any window.** The archetype
   the original design was implicitly built around, and the one that describes
   2008, is the one this system never triggers. The GFC sits in fold 2's
   window, where the gate is silent entirely.
2. **This is, in practice, a shock detector.** 90% of pooled fires and 94% of
   BLIND fires are `shock_led`. Calling it a general crash detector overstates
   what the evidence supports; it is a liquidity/correlation-break detector
   that occasionally tags other regimes.

`valuation_led` precision swings between 0.167 (pooled) and 1.000 (BLIND) on
5–6 fires — too few to mean anything either way.

---

## The calibration decision

Three options for the final recalibration stage were compared **on
walk-forward folds only**:

| | none | **platt** | isotonic |
|---|---|---|---|
| Pooled reliability slope | −0.415 | **0.419** | 0.215 |
| Fold slopes admissible | 2 / 4 | **3 / 4** | 3 / 4 |
| Pooled Brier skill | **−0.068** | −0.096 | −0.120 |
| Pooled gate precision | **0.859** | **0.859** | 0.772 |

Platt was chosen: it removes the sign inversion in the pooled slope, is the
only option that both improves calibration and leaves gate precision exactly
intact (it is monotone in the log-odds, so it cannot reorder days), and it
does not overfit the way isotonic does — isotonic has the worst Brier skill
*and* loses precision, the signature of a map memorising its training window.

`AggregatorConfig.posterior_calibration` selects between them; all three are
covered by tests.

---

## Why calibration keeps failing: the base rate is non-stationary

The reliability slope fails almost everywhere, and the reason is visible in
the fold base rates:

```
fold 1  P(maxDD >= 10% in 63d) = 0.603
fold 2                          = 0.327
fold 3                          = 0.176
fold 4                          = 0.394
```

The unconditional probability of the target event varies by a factor of
**3.4×** across decades. The aggregator anchors its prior on the *training*
base rate, and any calibration map is fitted on training outcomes — so when
the scoring window's base rate differs sharply, the absolute probability level
is wrong by construction, no matter which map is used. Nothing observable at
fit time reveals the shift.

**Ranking transfers across regimes; absolute probability does not.** That is
the honest characterisation of this system, and it is why gate precision
(0.859 pooled) looks strong while Brier skill is negative. Fixing it needs a
different approach — a drifting or hierarchical prior, or reporting calibrated
*relative* risk rather than absolute probability — not another calibration map
fitted the same way.

---

## What is established, and what is not

**Established**

- Gate fires carry real information: pooled precision 0.859 at 2.39× lift,
  ~40-day median lead.
- Drawdown reduction where the gate fires: 0.827× (fold 4), 0.707× (BLIND).
- The architecture runs end-to-end on 55 years of data; all five kill criteria
  are actually computed (the alpha checked one of five).
- Every defect in [V6_POSTMORTEM.md](V6_POSTMORTEM.md) is covered by a
  regression test (`tests/test_v6/`, 46 tests).
- The configuration is frozen and hash-verified, so the next evaluation can be
  *proven* single-shot rather than asserted to be.

**Not established**

- **That it passes its own kill criteria.** No window does.
- **That the probabilities mean anything in absolute terms.** Brier skill is
  negative on four of five windows.
- **That it beats buy-and-hold.** BLIND CAGR is 1.57pp below; pooled 0.26pp
  below. The case is drawdown reduction at comparable return.
- **That it detects credit-led crashes.** Zero fires of that archetype, ever.
- **That it works pre-2010.** Folds 1 and 2 produce no fires.
- **That the BLIND result is clean.** See the disclosure above.

---

## Next steps, in priority order

1. **Run the frozen holdout when data accrues.** `holdout_eval.py` is ready
   and currently refuses (0 trading days past the lock date; ~126 needed).
   This is worth more than any further modelling.
2. **Address base-rate non-stationarity directly** — a drifting prior, or
   reframing the output as relative risk. Another calibration map fitted the
   same way will not help; that has now been measured three ways.
3. **Give folds 1–2 usable macro history.** Pre-1986 credit and
   financial-conditions coverage is the binding constraint on the silent folds.
4. **Investigate why `credit_led` never fires.** Either its composite is
   mis-specified or its threshold is unreachable; both are testable against
   2007–2009 directly.
5. **Re-frame the fold-1 regime.** At a 0.603 base rate the binary question
   carries little information; severity or timing is the better target there.
