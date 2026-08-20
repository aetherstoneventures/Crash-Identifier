# V6 Crash KPI Engine — Honest Scorecard (v6.1)

> Results are reported **as observed**, not as wished. Reproduce with:
>
> ```bash
> python scripts/v6/validate.py both --x 10 --h 63
> ```
>
> Artefact: `data/v6_artifacts/v6_validation_x10_h63.json`
> Frozen configuration: `data/v6_artifacts/frozen_config_v6.2.0.json`
> (hash `14d8980fa007e3ce…`, lock date 2026-08-19)
> Iteration history and rejected alternatives:
> [DECISION_LEDGER.md](DECISION_LEDGER.md)

---

## Verdict

| Window | Fires | Precision | Lift | Lead | Slope | CAGR Δ | MaxDD ratio | Kill |
|---|---|---|---|---|---|---|---|---|
| Fold 1 (1999–2005) | 0 | — | — | — | 0.984 | +0.00pp | 1.000 | ❌ [5] |
| Fold 2 (2005–2012) | 0 | — | — | — | 0.578 | +0.00pp | 1.000 | ❌ [5] |
| Fold 3 (2012–2020) | 37 | 0.297 | 1.69× | 28d | 0.668 | +0.58pp | 0.772 | ✅ **PASS** |
| Fold 4 (2020–2026) | 40 | 1.000 | 2.54× | 23d | −0.168 | +1.57pp | 0.832 | ❌ [1] |
| Pooled *(diagnostic)* | 77 | 0.662 | 1.85× | 24d | 0.263 | +0.51pp | 1.000 | — |
| **BLIND (2021–2026)** | 12 | **1.000** | **2.67×** | 22d | 0.525 | **+2.43pp** | **0.770** | ✅ **PASS** |

**Two of six windows pass. Read the next section before quoting the BLIND
row** — it is not a validated result, and the reasons are quantified rather
than hedged.

---

## ⚠️ Why the BLIND pass is not evidence

Three independent reasons, each measurable:

**1. The window is contaminated.** The Layer 1 archetype split was designed
after inspecting 2022 gate diagnostics. BLIND has functioned as a development
set since then.

**2. It sits at the end of a nine-configuration search.** Across 8
configurations × 4 folds = 32 fold-evaluations, the observed marginal pass
rates were 0.562 / 0.844 / 0.938 / 1.000 / 0.438 for criteria 1–5. Treating
them as independent, P(one window passes all five) ≈ 0.195, so

> **P(at least one of 32 fold-evaluations passes by chance) = 0.999**

A passing window emerging from this search was near-certain *a priori*. That
is why model iteration was stopped (see
[DECISION_LEDGER.md](DECISION_LEDGER.md) ITER-010) rather than continued until
more windows passed.

**3. The margin is thin.** The BLIND reliability slope is **0.525** against a
floor of 0.500. A 5% perturbation flips it. Criterion 1 is barely met, not
comfortably met.

### What can fairly be claimed

On the configuration selected by walk-forward evidence, the 2021–2026 window
produced 12 gate fires at **1.000 precision** and **2.67× lift**, a 22-day
median lead, **+2.43pp CAGR** against buy-and-hold, and a **23% drawdown
reduction**. Whether that survives contact with unseen data is precisely the
question `scripts/v6/holdout_eval.py` exists to answer, and it currently
reports *"0 trading days past the lock date, need ~126 more."*

### The two remaining failures, with causes

- **Folds 1–2 never fire (criterion 5).** All four engines measure **zero
  skill** on those training windows, so weights fall back to 0.25 each and the
  tuned gate demands 2.0–3.0× lift that never materialises. Pre-1999 the
  feature vector is thin: VIX and credit begin 1986, STLFSI 1994, real rates
  2003, VIX term structure 2007. This is a data-availability limit.
- **Fold 4's reliability slope (−0.168).** Diagnosed: the model stays alarmed
  through recoveries. Its top posterior decile hits 0.310 against a 0.394 base
  rate, and 51 of those days fall in 2023. A remedy exists and works on
  calibration (three recovery-phase features move the slope to +0.198 and the
  pooled slope to 0.488) but suppresses firing, so it is shelved rather than
  adopted — see ledger ITER-007.

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
| Days scored | 1,466 |
| Gate fires | 12 (0.82%) |
| Base rate P(maxDD ≥ 10% in 63d) | 0.375 |
| Precision | **1.000** (2.67× base rate) |
| Recall | 0.023 |
| Median lead to the −10% crossing | **22 trading days** |
| Brier | 0.2458 (skill -0.049) |
| Reliability slope | **0.525** ✅ (floor 0.500) |

| Backtest | Strategy | Buy & hold |
|---|---|---|
| CAGR | **15.49%** | 13.07% |
| **MaxDD** | **-28.0%** | -36.4% |
| Sharpe | 0.80 | 0.67 |
| Time in market | 94.7% | 100% |

All five criteria pass — with the three caveats above. Note the recall of
0.023: the gate is deliberately rare and precise, catching a small
fraction of qualifying days rather than most of them.

### Walk-forward folds

| Fold | Window | Days | Fires | Base rate | Slope | Brier skill | CAGR Δ | MaxDD ratio |
|---|---|---|---|---|---|---|---|---|
| 1 | 1999–2005 | 1,566 | 0 | 0.603 | 0.984 ✅ | -0.675 | +0.00pp | 1.000 |
| 2 | 2005–2012 | 1,825 | 0 | 0.327 | 0.578 ✅ | +0.037 | +0.00pp | 1.000 |
| 3 | 2012–2020 | 2,088 | 37 | 0.176 | 0.668 ✅ | -0.036 | +0.58pp | 0.772 |
| 4 | 2020–2026 | 1,728 | 40 | 0.394 | -0.168 ❌ | -0.163 | +1.57pp | 0.832 |
| Pooled | 1999–2026 | 7,206 | 77 | 0.359 | 0.263 ❌ | -0.089 | +0.51pp | 1.000 |

- **Folds 1–2: silent.** Zero measured engine skill on those training windows
  (weights fall back to 0.25 each). Fold 1's base rate of **0.603** is itself
  notable: across 1999–2005 the Nasdaq spent most days within 63 days of a 10%
  drawdown, and a binary "will it fall" gate carries little information when
  falling is the norm.
- **Fold 3: passes all five.** 37 fires at 1.69× lift, +0.58pp CAGR and
  drawdown cut to 0.772×.
- **Fold 4: perfect precision, inverted probabilities.** 40 fires, precision
  1.000, +1.57pp CAGR, drawdown 0.832× — with a reliability slope of −0.168.
  The ranking is excellent; the calibration is absent. Cause diagnosed above.

---

## By crash archetype

The point of decomposing Layer 1 was to distinguish crash types rather than
average them away. Doing so surfaces a limitation a blended number hides.

**Pooled walk-forward**

| Archetype | Fires | Precision | Median lead |
|---|---|---|---|
| `credit_led` | **0** | — | — |
| `rate_led` | **0** | — | — |
| `shock_led` | 60 | 0.650 | 23d |
| `valuation_led` | 17 | 0.706 | 28d |

**BLIND**

| Archetype | Fires | Precision | Median lead |
|---|---|---|---|
| `rate_led` | **0** | — | — |
| `shock_led` | 12 | 1.000 | 22d |
| `valuation_led` | **0** | — | — |

Two findings, both uncomfortable and both worth stating:

1. **`credit_led` has never fired — not once, in any window.** The archetype
   the original design was implicitly built around, and the one that describes
   2008, is the one this system never triggers. The GFC sits in fold 2's
   window, where the gate is silent entirely.
2. **This is, in practice, a shock detector.** The overwhelming majority of
   fires in every window are `shock_led`. Calling it a general crash detector
   overstates what the evidence supports; it is a liquidity/correlation-break
   detector that occasionally tags other regimes.

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

- Gate fires carry real information: pooled precision 0.662 at 1.85× lift,
  ~24-day median lead; on BLIND, 1.000 precision at 2.67× lift.
- Drawdown reduction wherever the gate fires: 0.772× (fold 3), 0.832×
  (fold 4), 0.770× (BLIND).
- The architecture runs end-to-end on 55 years of data, and all five kill
  criteria are actually computed (the alpha checked one of five).
- Every defect in [V6_POSTMORTEM.md](V6_POSTMORTEM.md) is covered by a
  regression test (`tests/test_v6/`, 57 tests; 120 in the full suite).
- The configuration is frozen and hash-verified, so the next evaluation can be
  *proven* single-shot rather than asserted to be.
- Nine configurations were evaluated and their results — including six
  refutations — are recorded in [DECISION_LEDGER.md](DECISION_LEDGER.md)
  rather than discarded.

**Not established**

- **That the passing windows mean anything.** P(a chance pass across this
  search) = 0.999. Two of six windows pass; that is what chance predicts.
- **That the probabilities are trustworthy.** Brier skill is negative on five
  of six windows, and fold 4's reliability slope is still inverted (−0.168).
- **That it works pre-2010.** Folds 1 and 2 produce no fires at all, because
  every engine measures zero skill on those training windows.
- **That it detects credit-led crashes.** Zero fires of that archetype, ever,
  in any window — including the fold containing 2008.
- **That the BLIND result is clean.** Contaminated, thin-margin, and
  post-search. See the disclosure above.

---

## Next steps, in priority order

1. **Run the frozen holdout when data accrues.** `holdout_eval.py` is ready
   and currently refuses (0 trading days past the lock date; ~126 needed).
   This is worth more than any further modelling — and per ITER-010, further
   modelling on this data cannot produce valid evidence at all.
2. **Do not iterate further on the walk-forward folds.** Nine configurations
   have been tried; the multiple-comparisons arithmetic is in the ledger.
   Additional search will produce passing windows without producing skill.
3. **Fold 4's calibration has a known lever.** Three recovery-phase features
   move its slope from −0.168 to +0.198 and the pooled slope from 0.263 to
   0.488, at the cost of firing. Shelved, reproducible, ledger ITER-007.
4. **The gate tuner is unstable.** Many operating points hit the target
   training fire rate and generalise very differently (40 fires vs 0 for the
   same fold). Selecting on training precision was tried and refuted
   (ITER-008); selecting on training *economics* has not been.
5. **Give folds 1–2 usable macro history**, or drop them from the protocol
   with that reason stated. Pre-1999 the feature vector is too thin for any
   engine to measure skill.
6. **Investigate why `credit_led` never fires.** Either its composite is
   mis-specified or its threshold is unreachable; testable directly against
   2007–2009.
