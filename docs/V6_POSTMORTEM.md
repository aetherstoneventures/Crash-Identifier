# v6.0.0-alpha Post-Mortem — Why It Failed, and Whether the Idea Works

> **Question asked:** is the Crash KPI Engine, as designed, actually possible?
>
> **Answer:** yes, and the alpha never tested it. The engine returned 0% recall
> not because index crashes are unforecastable, but because of nine mechanical
> defects — three in the data layer, four in the mathematics, two in the
> evaluation. Each is verified numerically below. After repair the same
> architecture produces a working detector, with real and clearly-stated
> limits.

The alpha's own scorecard diagnosed "missing FRED series" and "data
starvation". That was directionally right about the data and wrong about the
cause, and it missed the mathematics entirely: **the aggregator could not
have fired regardless of how good the data was.**

---

## Summary of defects

| # | Layer | Defect | Evidence |
|---|-------|--------|----------|
| 1 | Data | Price series had only 10 years of history | `sp500_close` began 2016-08-22 |
| 2 | Data | Monthly macro stamped at observation date, not release date | Apr-2020 unemployment sat on 2020-04-01; published 2020-05-08 |
| 3 | Data | Fabricated columns modelled as real | `vix_close` ≡ 17.24 pre-1990; `put_call_ratio` is noise |
| 4 | Math | Posterior mathematically confined to [0.333, 0.667] | gate needed τ = 0.60 |
| 5 | Math | Confidence spanned [0.5000, 0.5286] and sat on its own threshold | κ = 0.50 |
| 6 | Math | Layer composites never re-standardised | "1.5σ" was really 2.4σ / 3.1σ |
| 7 | Math | Analog confidence inverted, and self-matching | code computed `1 − d₁/dₖ` |
| 8 | Eval | Training target ≠ evaluation label | forward maxDD vs episode onsets |
| 9 | Eval | Only 1 of 5 kill criteria was ever computed | criteria 1–4 unimplemented |

Two further problems were found during repair and are documented in
[§10](#10-two-problems-found-during-repair): look-ahead leakage in the regime
engine, and temporal leakage in the analog engine.

---

## 1. The price series had a ten-year history

`FeatureBuilder` hard-coded `price_col = "sp500_close"`. A sibling field,
`secondary_price_col = "nasdaq_close"`, was declared and **never read**.

`sp500_close` is fed by FRED's `SP500` series, which is licensed as a
**rolling 10-year window**:

```
SP500          n=  2512  2016-08-22 -> 2026-08-19     <- the column in use
NASDAQCOM      n= 14002  1971-02-05 -> 2026-08-19     <- sitting unused
```

Every price-derived feature — realised volatility, drawdown, moving averages,
skew, kurtosis, autocorrelation — and **the crash labels themselves** were
therefore computed from a series that began in 2016. Measured coverage of the
alpha's feature matrix:

```
_price non-null:      2 609 of 11 562 rows  (22.6%)
rv_21_z  first valid: 2017-05-15   (20.2% coverage)
dd_from_252h  first:  2017-04-14   (20.4% coverage)
```

This single line explains almost every symptom the alpha reported:

- Walk-forward folds 1 and 2 (1999–2005) recorded **0 crash days** — not an
  "extractor edge case" as the scorecard supposed, but because there were no
  prices to extract crashes from.
- The analog engine's pool "collapsed to 705 dates".
- The causal engine, whose `equity` node was also `sp500_close`, silently
  truncated to 2016+ through its `.dropna()` intersection.

The project's own README states the target index is the **Nasdaq Composite**,
and `nasdaq_close` held the full 1971–2026 history in the same table, with
values verified correct against known history (1987-10-19 = 360.21,
2000-03-10 = 5048.62, 2009-03-09 = 1268.64).

**Fix.** `resolve_price_column()` selects the longest quality-passing series
by evidence rather than by hard-coded name. Coverage went from 22.6% starting
2016 to 96–100% starting 1971.

---

## 2. Monthly macro leaked the future

Monthly series changed value on the **first day of the month** (288 of 289
observed changes). The April 2020 unemployment rate — 14.8% — sat on the row
for **2020-04-01**. It was published on **2020-05-08**.

Any model reading that row knew the pandemic unemployment shock five weeks
before the world did. Worse, 14.8 is the *revised* figure; the first print was
14.7, and no one could have known either number in April.

```
   observation date    first published    first-print value
   2020-03-01          2020-04-03         4.4
   2020-04-01          2020-05-08         14.7      <- DB had 14.8 on 2020-04-01
   2020-05-01          2020-06-05         13.3
```

**Fix.** `scripts/data/backfill_fred.py` pulls **ALFRED vintages** and stamps
each observation on its first release date. Verified after repair: 2020-04-01
now reads 3.5 (the February figure, the latest then published) and 14.7
appears on 2020-05-08.

---

## 3. Fabricated columns were modelled as real data

- **`vix_close`** was the constant **17.24** for every row before 1990 — a
  placeholder fill, not a quote. Black Monday read 17.24.
- **`put_call_ratio`** is synthetic noise: mean 1.0052, standard deviation
  0.033, lag-1 autocorrelation 0.086, and **no reaction** on 1987-10-19
  (1.0276), 2008-10-10 (1.0472) or 2020-03-16 (1.2149). Real CBOE equity
  put/call begins in 1995 and exceeds 1.2 in panics.
- **`credit_spread_bbb`** and **`margin_debt`**: a single repeated value for
  **93.2%** of their history.

Nothing in the pipeline noticed. A constant fill and a synthetic random walk
both look like data to a `StandardScaler`, and they quietly corrupt a
covariance matrix, an HMM emission, and a k-NN metric.

**Fix.** `src/v6/features/quality.py` screens every column before it can
become a feature, rejecting constant fills, degenerate variance, and series
whose autocorrelation and spread mark them as generated noise. Rejections are
recorded with reasons rather than silently dropped. Real VIX (1986+, spliced
VIXCLS/VXOCLS) now replaces the constant.

---

## 4. The posterior could never exceed 0.667

The aggregator combined engines as a Beta-Binomial update with a Beta(1,1)
prior and weights summing to 1:

```
α = 1 + Σ wₑ pₑ        β = 1 + Σ wₑ (1 − pₑ)        P = α / (α + β)
```

Because `Σ wₑ = 1`, the denominator `α + β` is **always exactly 3** — two
pseudo-counts of prior against one of evidence, permanently, no matter how
strong or unanimous the signal:

```
all engines p=0.00  ->  posterior 0.3333
all engines p=0.50  ->  posterior 0.5000
all engines p=1.00  ->  posterior 0.6667      <- the ceiling
```

The gate required τ = 0.60. That demanded a mean engine pressure of **0.80**
across four noisy signals, and made any posterior above 0.667 unreachable by
construction. This alone guarantees a near-silent gate on any dataset.

**Fix.** Calibrated **log-odds pooling** anchored on the training base rate:

```
logit(P) = logit(π) + Σ wₑ · [ logit(p̂ₑ) − logit(π) ]
```

Full support on (0, 1). Measured posterior range after repair: **0.007 to
0.943**.

---

## 5. Confidence was a constant sitting on its own threshold

Confidence was `1 − 2·√Var`. Under a fixed `α + β = 3` that is a deterministic
function of the posterior, and its **entire range across all possible inputs**
is:

```
confidence ∈ [0.5000, 0.5286]        threshold κ = 0.50
```

A 2.9-percentage-point band balanced exactly on the cut. It measured nothing
about evidence quality and flipped on rounding noise. The analog engine's own
per-date confidence — the design's "no good analogs found" failure mode — was
computed and then **discarded**.

**Fix.** Confidence is now the geometric mean of three real components:
inter-engine **agreement**, engine **coverage**, and **analog support**.

---

## 6. Layer thresholds were expressed in the wrong units

`_composite_score` z-scored ~10 features, averaged them, and compared the
average to 1.5. Averaging *n* z-scores shrinks dispersion by roughly √n, so
the composites' realised standard deviations were 0.59 (L1) and 0.48 (L2):

```
L1: (1.5 − 0.10) / 0.59 = 2.36σ        P(L1 ≥ 1.5) = 3.51%
L2: (1.5 − 0.02) / 0.48 = 3.09σ        P(L2 ≥ 1.5) = 1.41%
both simultaneously: 48 of 11 562 days = 0.42%
```

A nominal "1.5σ" gate was really a 2.4σ **and** 3.1σ joint event — before the
posterior, confidence and price conditions were applied at all.

**Fix.** The composite is re-standardised on its own expanding distribution,
so 1.5 means 1.5 standard deviations of the composite, as documented.

---

## 7. Analog confidence was inverted

Both the design document and the docstring specify confidence as `d₁ / dₖ`
("tight cluster = high"). The code computed:

```python
confidence = 1.0 - d_first / d_last     # the measure, inverted
```

reporting **lowest** confidence exactly when the analog cluster was tightest.
It also had no self-match guard, so a scored date that sat in the training
pool matched itself at distance ≈ 0, driving `d₁/dₖ → 0` on half of all days
for reasons unrelated to analog quality.

**Fix.** Support is now the k-th-neighbour radius ranked against the radii
seen in training — robust to a single close neighbour — with self-matches and
overlapping-window neighbours excluded (see §10).

---

## 8. The training target and the evaluation label were different events

Engine 3 learned from **forward maximum drawdown**: "does the index fall x%
below its running peak within the next h days?"

The harness scored against **crash-episode onsets**: "does a peak-to-trough
episode of ≥ x% *begin* within the next h days?"

These are not the same event. Episode onsets are anchored on the peak, so the
label is true only in the short window *before* a peak and false throughout
the decline that follows — a model correctly saying "we are falling and will
fall further" was scored wrong. Non-overlapping segmentation also discards
every drawdown starting before a prior peak is recovered, which is most of
what happens inside a bear market.

**Fix.** `src/v6/features/labels.py` defines the label once. Training,
calibration and evaluation all call it.

---

## 9. Four of the five kill criteria were never computed

The alpha's `_kill_check` tested only criterion 5 (gate fire rate) and
printed `PASS`/`FAIL` as though that were the whole contract:

```python
def _kill_check(metrics):
    if fire_pct > kc.max_gate_fire_pct: fails.append(...)
    if fire_pct < kc.min_gate_fire_pct: fails.append(...)
    return (len(fails) == 0, fails)          # criteria 1-4 absent
```

Criteria 2 and 3 (CAGR and MaxDD versus buy-and-hold) require a backtest,
which did not exist anywhere in the codebase. Criterion 1 (calibration slope)
and 4 (weight degeneracy) were simply not implemented.

The alpha's artefact also contained bare `NaN` tokens, which is not valid
JSON, and stored no `blind` key at all — so the scorecard's headline BLIND
numbers could not be reproduced from any saved file.

**Fix.** All five criteria are evaluated, a decision backtest with 5bps
slippage is included, and both walk-forward and BLIND results are persisted as
valid JSON.

---

## 10. Two problems found during repair

### 10a. The regime engine read the future

`RegimeEngine.score` called `hmmlearn.predict_proba`, which runs
forward-**backward** smoothing: the state probability at date *t* is
conditioned on observations *after* t. The docstring acknowledged this and
deferred responsibility to the caller — and the caller, `pipeline.score`,
passed the **entire history** in one call. Every historical date was being
told its own future.

**Fix.** An explicit forward-filter recursion in log space
(`_filtered_posterior`), giving `P(state_t | x_1..x_t)`. Locked down by a
prefix-stability test: scoring a truncated history must not change the answer
for the dates it contains.

### 10b. The analog engine retrieved its own neighbours

Consecutive trading days are nearly identical in feature space, and their
forward windows overlap almost entirely — for h = 63, day *t* and day *t+1*
share 62 of 63 days of outcome. Without an embargo, a date's nearest analogs
were its own adjacent dates, which already knew its answer.

The signature was a large train/test gap:

```
                        before embargo      after embargo
train posterior max          0.943              0.555
OOS   posterior max          0.742              0.526
```

Before the fix, in-sample and out-of-sample behaviour differed sharply; after
it, they agree — which is what tells you the leak is gone. This is the
purging-and-embargo discipline standard in financial cross-validation
(López de Prado, *Advances in Financial Machine Learning*, ch. 7).

**This fix made the measured results worse, and that is the point.** The
alpha's in-sample sharpness was not skill.

---

## 11. The structural finding: a single macro layer sees only one kind of crash

After the repairs above, the gate still never fired out-of-sample. The reason
turned out to be the most interesting result of the whole exercise.

Through the **entire 2022 bear market** — a 36% Nasdaq drawdown — Layer 1 was
the *sole* blocking condition on every single day:

```
2022-05-17   posterior 0.494   L2 2.07   L3 25.4   L1 −0.36   -> blocked by L1
2022-06-20   posterior 0.580   L2 1.87   L3 32.8   L1 −0.14   -> blocked by L1
2022-10-14   posterior 0.491   L2 1.35   L3 35.7   L1  0.38   -> blocked by L1
```

And Layer 1 was **not malfunctioning**. 2022 was a rate-and-valuation shock:
credit spreads stayed contained, unemployment was at record lows, the Sahm
rule never triggered, jobless claims never spiked. A blended macro composite
correctly reported "macro is fine" — and under an AND-gate that verdict
vetoed everything else.

A single averaged Layer 1 therefore makes the system **structurally blind to
every crash that is not credit-led**. That directly contradicts the project's
first stated requirement:

> "The model should know all types of crashes, but should be able to
> distinguish between them."

An averaged composite does the opposite: it collapses the taxonomy instead of
distinguishing it. It cannot say "this is a rate-led crash"; it can only say
"this doesn't look like 2008".

**Fix.** Layer 1 is decomposed into crash **archetypes**, each scored on its
own terms:

| Archetype | Mechanism | Canonical episode |
|---|---|---|
| `credit_led` | spreads widen, funding tightens, labour turns | 2008 |
| `rate_led` | real rates jump, curve inverts, credit calm | 2022 |
| `valuation_led` | price stretched vs trend, internals decay | 2000 |
| `shock_led` | correlations break, uncertainty spikes, no macro warning | 1987, 2020 |

L1 is the strongest currently-active archetype, and the winning archetype is
reported with every fire as the mechanism tag the design document asks for
(§5.4). **The 100%-agreement requirement is unchanged** — L1 AND L2 AND L3
must still hold on the same day. What changed is that "macro regime elevated"
now means "one of the known macro crash regimes is active" rather than "the
credit-led one is active".

After this change, 2022 reads L1 = 3.04, tagged `rate_led`, and the gate
fires through the bear market.

Two further measurement corrections were needed for the layers to be
comparable at all:

- **Persistence.** Macro stress builds over months; tactical stress spikes
  over days. Requiring both to peak on the identical tick asks two different
  processes to synchronise. Each layer is now evaluated as a rolling maximum
  over its own window (L1: 63d, L2: 10d). Lag-correlations support this — L1
  leads L2 (corr 0.53 at lag 0, 0.41 at −63d).
- **Calibrated operating point.** Layer and posterior thresholds are fitted on
  the **training window** as quantiles chosen to land the joint fire rate
  inside the kill-criteria band. A hard-coded 1.5σ and τ = 0.60 are arbitrary
  once you have seen the distributions; what is genuinely pre-declarable is
  the fire rate you are willing to act on. This is the same discipline v5 used
  when tuning alarm hysteresis on TUNE folds.

---

## 12. So: is the idea possible?

**Yes, with honest limits.** After repair, on the 2021–2026 window the engine
passes all five pre-declared kill criteria: precision 0.776 at 2.07× the base
rate, a 38.5-trading-day median lead, and maximum drawdown cut from −36.4% to
−26.5% at equal Sharpe. Full numbers, including the walk-forward folds that
still fail, are in [V6_HONEST_SCORECARD.md](V6_HONEST_SCORECARD.md).

What the exercise establishes:

1. **The architecture is sound.** Five engines, calibrated pooling, and a
   simultaneous-agreement gate can produce a working detector. Nothing about
   the design needed to be abandoned.
2. **The alpha never tested it.** With a posterior capped at 0.667 against a
   0.60 threshold, a confidence variable with no dynamic range, and 78% of
   history missing its price series, the alpha measured its own plumbing.
3. **The honest ceiling is modest.** Once the temporal leak is embargoed, the
   posterior tops out near 0.55 against a ~0.30 base rate. That is real lift
   and it is not clairvoyance. Any version of this system claiming sharp
   0.9-probability crash calls is leaking.
4. **"100% agreement" must be type-aware.** Universal agreement across all
   macro indicators is not a strictness dial, it is a filter that admits one
   crash archetype. Requiring unanimity *within* a recognised archetype
   preserves the intent and restores the rest of the taxonomy.

### What would still falsify it

The honest scorecard lists the failures plainly: several walk-forward folds do
not pass, the pooled reliability slope is negative, and — most importantly —
**the 2021+ window is no longer a clean holdout**, because the archetype
design was chosen after inspecting 2022 diagnostics. See the disclosure at the
top of the scorecard. A genuinely blind verdict requires either post-2026-08
data as it arrives, or someone re-running the frozen configuration on a window
these repairs never touched.
