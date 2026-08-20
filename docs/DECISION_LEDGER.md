# Decision Ledger — living project memory

> **Purpose.** This file is the project's working memory. It records what was
> tried, what was measured, what was decided, and *why* — in enough detail
> that someone (or a future session with no context) can pick the work up
> without re-deriving the reasoning.
>
> **Rules for this file.**
> 1. Append one entry per iteration. Never rewrite history; if a conclusion
>    turns out wrong, add a correcting entry and link back.
> 2. Every entry records the **measurement** that drove the decision, not just
>    the decision.
> 3. Decisions must be made on **walk-forward evidence only**. If a choice was
>    influenced by the holdout, say so explicitly — that is contamination and
>    it must be visible.
> 4. Record failures and dead ends with the same weight as successes. A
>    documented negative result is what stops the next person repeating it.

---

## The objective

Reach a state where merging `v6-kpi-engine` into `main` is objectively
justified — i.e. the pre-declared kill criteria (design doc §7) are met on
evidence, not by moving the goalposts.

| # | Kill criterion | Threshold |
|---|---|---|
| 1 | Reliability slope | within [0.5, 1.5] |
| 2 | CAGR vs buy & hold | ≥ B&H − 2pp |
| 3 | MaxDD ratio vs B&H | ≤ 1.10 |
| 4 | No engine dominates | max weight < 0.90 |
| 5 | Gate fire rate | within [0.10%, 10%] |

**Merge gate** (design doc §10): merge to `main` only after a BLIND verdict
and explicit owner approval.

**Definition of "achieved" used here:** criteria 1–5 pass on the **pooled
walk-forward** window (the evidence set), and the failure modes on individual
folds are understood and documented rather than unexplained. BLIND is
reported but is *not* a decision input — it is contaminated (see ITER-000) and
is now treated as a read-only observation until a clean holdout exists.

---

## ITER-000 — Baseline after the v6.1 repair

**Date:** 2026-08-20 · **Commits:** `01a9c35`, `314b14d`

### What was done
Repaired nine mechanical defects in v6.0.0-alpha (see
[V6_POSTMORTEM.md](V6_POSTMORTEM.md)), then added cross-fitted Platt
recalibration, per-archetype reporting, and configuration freezing.

### Measurement

| Window | Fires | Precision | Lift | Slope | Brier skill | CAGR Δ | MaxDD ratio | Kill |
|---|---|---|---|---|---|---|---|---|
| Fold 1 (1999–2005) | 0 | — | — | 1.156 | −0.672 | +0.00pp | 1.000 | ❌ [5] |
| Fold 2 (2005–2012) | 0 | — | — | 0.615 | +0.026 | +0.00pp | 1.000 | ❌ [5] |
| Fold 3 (2012–2020) | 10 | 0.000 | 0.00× | 0.619 | −0.035 | −2.68pp | 1.002 | ❌ [2] |
| Fold 4 (2020–2026) | 61 | 1.000 | 2.54× | −0.071 | −0.183 | +2.16pp | 0.827 | ❌ [1] |
| **Pooled** | 71 | 0.859 | 2.39× | **0.419** | −0.096 | −0.26pp | 1.000 | ❌ [1] |
| BLIND (contaminated) | 116 | 0.750 | 2.00× | 0.330 | −0.056 | −1.57pp | 0.707 | ❌ [1] |

### Critical evaluation
- **What works:** detection. Pooled precision 0.859 at 2.39× base rate with a
  ~40-day median lead is a real signal, not noise.
- **What fails:** calibration (criterion 1) on the pooled window, and the gate
  is silent on folds 1–2 (criterion 5).
- **Diagnosed cause of the calibration failure:** the base rate of the target
  event is non-stationary — 0.603 / 0.327 / 0.176 / 0.394 across folds, a
  **3.4× swing**. The aggregator anchors its prior on the *training* base rate
  and its calibration map is fitted on training outcomes, so both are wrong by
  construction whenever the scoring window's prevalence differs.
- **Ruled out:** three calibration maps (none / Platt / isotonic) were
  compared; all fitted statically on the training window. None fixed it. The
  problem is not which map — it is that *any* static map encodes a stale
  prevalence.

### Contamination disclosure
The 2021+ BLIND window was inspected while diagnosing why the gate never
fired, and the Layer 1 archetype split was designed in response. BLIND is
therefore **not** a clean holdout and is excluded from decision-making from
this point forward. `src/v6/freeze.py` + `scripts/v6/holdout_eval.py` now make
the "no retuning" rule checkable; the frozen config is `v6.1.0`
(`9745c43c2b0d33b7`, lock date 2026-08-19).

### Decision
Attack the diagnosed root cause — stale prevalence — rather than trying a
fourth static calibration map. See ITER-001.

### Integration status
`v6-kpi-engine` is 12 commits ahead of `main`; **not merged**. Correct per
design doc §10, since the verdict fails.

---

## ITER-001 — Online recalibration ❌ **REFUTED**

**Date:** 2026-08-20 · **Hypothesis:** the calibration failure is caused by
stale prevalence, so recalibrating online — as outcomes resolve — will fix it.

### Rationale
A forward-looking label becomes knowable with a lag: the outcome of date
`t − h` is fully determined by prices at date `t`. So a system sitting at `t`
may legitimately look back at resolved outcomes and update itself. That makes
recalibration an *online* problem rather than a train-once problem, and lets
the map track the prevalence actually being observed.

### What was built
`AdaptiveCalibrator` — refits the Platt map and the prior every 21 trading
days on a trailing 1260-day window of **resolved** labels only (positions
`≤ i − h`). Causality is enforced in one place and covered by
`tests/test_v6/test_adaptive.py` (11 tests).

### Causality: verified ✅
The mechanism is sound and provably non-leaking:
- **Prefix stability** — scoring a truncated history reproduces its dates exactly.
- **Future-label poisoning** — flipping every label after date *k* changes
  nothing at or before *k*.
- **Control** — flipping already-resolved labels *does* change later output,
  proving the channel is real and not merely inert.

A latent bug was caught by these tests: at early positions the window returned
`hi = −63` with `lo = 0`, which would have sliced `values[0:-63]` — nearly the
whole array — had the size guard ever been removed. Fixed to return `(0, 0)`.

### Measurement — walk-forward (the decision evidence)

| Fold | Slope before | Slope after | Fires before | Fires after |
|---|---|---|---|---|
| 1 | 1.156 ✅ | **−0.745** ❌ | 0 | 0 |
| 2 | 0.615 ✅ | **−0.269** ❌ | 0 | 0 |
| 3 | 0.619 ✅ | **−0.626** ❌ | 10 | 8 |
| 4 | −0.071 ❌ | **−1.470** ❌ | 61 | 3 |
| Pooled | 0.419 | 0.396 | 71 | 11 |

Pooled precision **0.859 → 0.273**. Pooled lift **2.39× → 0.76×** — i.e.
worse than firing at random.

### Why it failed (diagnosed, not guessed)

| | raw posterior | after map | fires |
|---|---|---|---|
| Static Platt | std 0.0832 | **std 0.1735** | 61 |
| Adaptive | std 0.0832 | **std 0.0982** | 3 |

A static map fitted on the *full* training window has enough data to find the
relationship, so it is **steep** — it expands discrimination (0.083 → 0.174).
A map refitted monthly on five noisy years is **flat**, and a flat map
collapses every prediction toward the local base rate (0.083 → 0.098). With
the spread gone, few days clear any threshold and the residual variation is
mostly noise — which is exactly what a negative reliability slope reports.

**The hypothesis was wrong about the remedy, not the diagnosis.** Prevalence
does drift; refitting the map on a trailing window is simply not a viable way
to track it, because the window that is short enough to be current is too
short to estimate a map.

### Decision
- `adaptive_calibration` (posterior remapping) → **shipped OFF**. Kept
  selectable so the negative result is reproducible rather than folklore.
- `adaptive_base_rate` (trailing prevalence tracking) → **shipped ON**. It is
  cheap, it is causal, and it does not touch the posterior. It feeds the
  gate's new *lift* requirement (below).

### Carried forward from this iteration
The gate's posterior condition became a **lift requirement** —
`posterior ≥ lift × live base rate` — instead of an absolute cut. An absolute
threshold cannot hold its meaning while prevalence moves: with the live base
rate ranging 0.22–0.49, a fixed 0.35 cut means "1.6× lift" in a calm regime
and "0.7×" — less likely than average — in a stressed one. Requiring lift
keeps the statement constant: *materially more likely than usual, right now.*

### Correction to ITER-000's framing
ITER-000 said the pooled reliability slope was the headline calibration
failure. That framing under-reported an important fact: with static Platt,
**folds 1, 2 and 3 individually PASS criterion 1** (1.156 / 0.615 / 0.619).
Only fold 4 fails (−0.071), and the *pooled* number fails. Pooling four
windows whose base rates span 0.176–0.603 conflates populations, so the
pooled reliability curve mixes regimes. Whether the pooled slope is even the
right statistic is now an open question — see ITER-002.

---

## ITER-002 — Lift-based gate ✅ **ADOPTED**

**Hypothesis:** an absolute posterior threshold cannot hold its meaning while
prevalence drifts, so the gate should require *lift over the live base rate*.

**Change:** gate fires when `posterior ≥ lift × live_base_rate` (lift tuned on
training over a grid), with the live base rate tracked causally by the
surviving half of ITER-001.

### Measurement (walk-forward)

| Fold | CAGR Δ before → after | MaxDD ratio before → after | Fires | Verdict |
|---|---|---|---|---|
| 3 | −2.68pp → **+0.58pp** | 1.002 → **0.772** | 10 → 37 | ❌ → ✅ **PASS** |
| 4 | +2.16pp → +0.37pp | 0.827 → 0.962 | 61 → 32 | still ❌ [1] |
| Pooled | −0.26pp → **+0.24pp** | 1.000 | 71 → 69 | still ❌ [1] |

**Fold 3 became the first window to pass all five criteria.** Pooled CAGR
turned positive. Adopted.

**Cost:** pooled precision fell 0.859 → 0.623. The gate fires more often and
at better times — worse precision, materially better economics. For an action
gate, the economics are the thing being optimised.

---

## ITER-003 — Variance tempering ⚠️ **ADOPTED (mixed)**

**Hypothesis:** the cross-fitted reliability slope (computed since ITER-000
but never consumed) says how far the map will over-disperse out of sample.
Shrinking log-odds deviations by that factor should pull the slope toward 1.
Strictly monotone, so ranking and precision are untouched.

### Measurement

| Fold | Slope before → after |
|---|---|
| 1 | 1.156 → **0.984** (closer to 1) |
| 2 | 0.615 → 0.578 |
| 3 | 0.619 → **0.668** |
| 4 | −0.071 → **−0.168** (worse) |
| Pooled | 0.419 → **0.263** (worse) |

Economics improved: pooled CAGR +0.24 → **+0.51pp**; fold 4 +0.37 → **+1.57pp**
with MaxDD ratio 0.962 → 0.832.

**Decision: adopted, on a tie-break.** Folds passing criterion 1 is unchanged
(3 of 4 either way), so the criterion count does not separate them; the
economics do, clearly. Recorded as *mixed* rather than a win, because the
pooled slope regressed and the stated hypothesis only half held.

---

## ITER-004 — Recovery-phase features ❌ **REFUTED**

**Hypothesis:** fold 4's negative slope comes from the model staying alarmed
during recoveries. Level features cannot distinguish "falling into a crash"
from "climbing out of one" — both show a deep drawdown and high volatility.

**Evidence for the diagnosis (this part held up):** in fold 4 the top decile
of posterior days had an empirical hit rate of **0.310 against a 0.394 base
rate** — inverted — and 51 of those days fell in **2023**, during the recovery
from the 2022 bear market.

**Change:** added `dd_change_21d`, `rv_21_trend`, `price_vs_63d_low` to the
feature vector, and `dd_change_21d` to the HMM's emission set.

### Measurement — clearly worse

| | before | after |
|---|---|---|
| Pooled slope | 0.263 | **−0.567** |
| Pooled precision | 0.662 | **0.276** |
| Pooled lift | 1.85× | **0.77×** (worse than random) |
| Pooled CAGR Δ | +0.51pp | −1.09pp |
| Fold 3 | **PASS** | ❌ fails 3 criteria |
| Fold 4 fires | 40 | 0 |

**Reverted.** Note the diagnosis (recovery-phase confusion is real and
measured) is *not* refuted — only this remedy is. A likely confound: the
change also added a sixth dimension to a 4-state full-covariance HMM that was
already emitting non-convergence warnings. The features and the HMM change
were not ablated separately. See ITER-006.

---

## ITER-005 — Prior re-anchoring ❌ **REFUTED**

**Hypothesis:** shift the posterior's *centre* onto the live base rate while
keeping the evidence deviation — estimating only a mean (which a short window
can do) rather than a slope (which it cannot, per ITER-001).

**Result: 0 fires in every fold**, all slopes negative. Catastrophic.

**Cause — an interaction bug between two of my own changes.** Re-anchoring
makes `posterior ≈ live_base_rate × evidence_factor`, and the ITER-002 gate
requires `posterior ≥ lift × live_base_rate`. The base rate appears on both
sides and cancels, so the fire condition degenerates to
`evidence_factor ≥ lift` with lift tuned to 1.8–3.0 — a bar the evidence term
essentially never clears. Re-anchoring is only coherent with an *absolute*
posterior threshold, never with the lift gate.

**Reverted** (`reanchor_prior=False`, kept selectable).

### A hypothesis of mine that the data refuted
While diagnosing this I asserted that trailing prevalence is *negatively*
correlated with forward prevalence (mean reversion). **That is false.**
Measured directly:

| trailing window | corr(trailing, forward outcome) | top-quintile → forward rate |
|---|---|---|
| 252d | **+0.191** | 0.510 vs 0.211 bottom (overall 0.304) |
| 756d | +0.115 | 0.398 vs 0.342 |
| 1260d | **+0.067** | 0.362 vs 0.270 |

Trailing prevalence is *positively* predictive — but the 1260-day window used
in ITER-001/005 carries almost none of that signal (+0.067). **If prevalence
tracking is revisited, use ~252 days, not 1260.** Recorded as an open lead.

---

## ITER-006 — Reading the kill criteria as written 📋 **PROTOCOL CORRECTION**

Three consecutive refutations were all aimed at the *pooled* reliability
slope. Re-reading the source specification, that target was my own invention:

> Design doc §7, criterion 1: *"**BLIND** aggregator calibration error > 10
> percentage points (reliability slope < 0.5 or > 1.5)."*

The pre-declared criteria are defined on **a single evaluation window**, not
on a pooled ensemble of four separately-fitted models. Pooling forecasts from
four different models, trained on different eras and centred on different
training base rates (0.219 / 0.348 / 0.338 / 0.300) against test base rates
of 0.603 / 0.327 / 0.176 / 0.394, produces a mixture distribution whose
reliability curve is not any single forecaster's calibration.

**Correction to the objective stated at the top of this ledger:** criteria are
assessed **per window**, as designed. Pooled figures continue to be reported
as a diagnostic and are explicitly *not* a pass/fail target.

This is a reading of the original spec, not a relaxation — and it is recorded
here precisely so a reader can check that claim rather than take it on trust.
An earlier sub-hypothesis, that a larger train/test prevalence gap predicts
worse calibration, is also **refuted**: the measured correlation between
|gap| and fold slope is **+0.708**, and fold 1 has both the largest gap
(+0.384) and the best slope (0.984).

### Status under the corrected reading

| Fold | Criterion 1 | Criterion 5 | Others | Verdict |
|---|---|---|---|---|
| 1 | 0.984 ✅ | 0 fires ❌ | ✅ | ❌ |
| 2 | 0.578 ✅ | 0 fires ❌ | ✅ | ❌ |
| 3 | 0.668 ✅ | 1.77% ✅ | ✅ | ✅ **PASS** |
| 4 | −0.168 ❌ | 2.31% ✅ | ✅ | ❌ |

Two concrete blockers remain, each with a measured cause:
- **Folds 1–2 never fire.** All four engines measured **zero skill** on those
  training windows (weights fall back to 0.25 each), and the tuned gate then
  demands 2.5–3.0× lift that never materialises. Pre-1999 the feature vector
  is thin — VIX and credit start 1986, STLFSI 1994, VIX term structure 2007,
  real rates 2003. This looks like a data-availability limit, not a tuning one.
- **Fold 4's slope.** Diagnosed (recovery-phase alarms, ITER-004) but not yet
  remedied.
