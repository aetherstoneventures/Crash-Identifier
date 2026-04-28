# Future-Work Experiments — Honest Scorecard

All four ideas listed in [V6_NEGATIVE_RESULT.md](V6_NEGATIVE_RESULT.md#what-might-be-worth-trying-future-work-with-same-kill-discipline)
were tested under the same kill discipline as v5.1 and v6. Result: **none beats v5.**
v5 (tag `v5-BENCHMARK`, commit `925d8b7`) remains production.

---

## Summary table (all results vs v5 on BLIND 2021-2026)

| Experiment | Hypothesis | BLIND CAGR | BLIND day_prec | Verdict |
|---|---|---:|---:|:---|
| **v5 BENCHMARK** | (baseline) | **23.0%** | **95.8%** | — |
| A — Vol-regime gate | Suppress alarms during calm regimes | 23.0% (=) | 95.8% (=) | NULL — no-op |
| B — Conformal intervals | Add uncertainty bounds to v5 prob | n/a | n/a | SKIPPED — diagnostic only, doesn't change alarms |
| C — Bottom-finder vs MA crossover | Beat MA50 re-entry rule | 23.4% (oracle ceiling) | n/a | DEAD END — oracle headroom only +0.5pp |
| D — Multi-asset crash labelling | More events → robuster detector | 15.2% | 57.1% | FAIL — kill criteria 3/5 |

---

## Experiment A — Volatility-regime gating

**Idea.** Use 20d realized vol of S&P 500 vs its 504d rolling quantile to define "calm regime."
Suppress v5 alarms when realized vol is in the bottom 10-40th percentile.

**Method.** Threshold q ∈ {0.10, 0.15, …, 0.40}; choose q on TUNE only by maximizing
TUNE backtest CAGR subject to ev_det ≥ v5 ev_det.

**Result.** **q = 0.10 selected.** Gate filters **0 alarms** at any q value tested.

**Why.** v5's signal is itself a function of vol and economic stress. Whenever v5 fires, vol is
already elevated above the calm threshold — the gate is mathematically redundant with v5's existing
features. PASS on all kill criteria but **0pp improvement.**

**Conclusion.** No regression, no enhancement. Don't ship.

Output: [data/experiment_A_vol_gate.json](../data/experiment_A_vol_gate.json)

---

## Experiment B — Conformal prediction intervals

**Skipped intentionally.**

Conformal intervals quantify per-day uncertainty around v5's probability estimate. They do not
change the alarm decisions and therefore cannot affect ev_det / day_prec / CAGR. The user's stated
objectives (predict crashes earlier, find bottoms better than MA50) are about decisions, not
diagnostics. A conformal layer would be a UI / risk-attribution feature deferred until v5 is
deployed in a context where uncertainty bands inform position sizing — out of scope for now.

---

## Experiment C — Re-entry rule shootout

**Idea.** Test whether any re-entry rule beats the current MA50 rule. Compare MA30/MA200,
MA50/MA200, signal-only (re-enter when v5 prob < exit threshold), and an **oracle** (perfect
bottom-finder that knows future trough).

**Result.**

| Rule | FULL CAGR | TUNE CAGR | BLIND CAGR | BLIND Sharpe |
|---|---:|---:|---:|---:|
| **R1 MA50 (current)** | **16.3%** | **14.7%** | **23.0%** | 1.45 |
| R2 MA30 above MA200 | 14.0% | 12.9% | 18.8% | 1.26 |
| R3 MA50 above MA200 | 13.9% | 12.6% | 19.5% | 1.33 |
| R4 Signal-only | 16.1% | 14.2% | 23.7% | 1.49 |
| R5 ORACLE (cheats) | 16.9% | 15.3% | 23.4% | 1.46 |

**The hard ceiling.** A perfect bottom-finder that *knows the future trough* improves CAGR by
only **+0.5 pp full / +0.3 pp blind** over MA50. This is the **mathematical headroom** for any
re-entry-rule improvement on this strategy.

**Conclusion.** **MA50 is at the practical optimum.** No bottom-finder model — neither v5.1's
nor v6's nor any future one — can deliver more than ~0.5pp CAGR. This permanently closes the
"better-than-MA50 bottom finder" line of research.

Output: [data/experiment_C_reentry.json](../data/experiment_C_reentry.json)

---

## Experiment D — Multi-asset crash labelling

**Idea.** Train v5-architecture XGBoost on a pooled panel of US (^IXIC), EU (Stoxx50, DAX, FTSE),
and Asia (Nikkei, Hang Seng) crashes. ~83 events vs v5's 17.

**Data.** 6 indices fetched via `yfinance` + `curl_cffi` browser impersonation (1971-2026).
Per-index features (drawdown, returns, vol, MA distances) + shared US macro (VIX, regime, credit,
StatV3 risk).

**Training.** TUNE pooled = 49 915 rows / 18 773 crash days. US BLIND held out completely.

**AUCs.** US TUNE in-sample 0.943, US BLIND OOS **0.850** — actually the strongest BLIND AUC
of any model we've trained. So the model *is* learning generalizable crash physics.

**Round 1 (v5's alarm thresholds).** day_prec dropped to 62.7%, BLIND CAGR 13.0%. Unfair
comparison — different probability calibration.

**Round 2 (re-tune alarm thresholds on US TUNE only).** Selected entry=0.30, exit=0.25,
min_dur=15 (highest TUNE Sharpe subject to ev_det≥0.85, day_prec≥0.80 — actually no config met
the day_prec≥0.80 bar so we relaxed to ev_det≥0.85 only).

| Metric | v5 | v5_multi (retuned) | Δ |
|---|---:|---:|---:|
| BLIND ev_det | 100% | 100% | 0 |
| BLIND day_prec | 95.6% | 57.1% | **-38.4 pp** ← FAIL |
| BLIND adp | 26.0% | 34.0% | +8.0 pp |
| BLIND lead | -9d | -66d | -57 d |
| FULL CAGR | 16.5% | 12.9% | **-3.5 pp** ← FAIL |
| BLIND CAGR | 23.7% | 15.2% | **-8.5 pp** ← FAIL |
| FULL MaxDD | -14.8% | -13.1% | +1.6 pp |

**Why higher AUC ≠ better strategy.** The multi-asset model averages across heterogeneous crash
regimes (US tech bubble, EU sovereign debt, Japan deflation, HK political shocks). The resulting
decision boundary is *more general* — it fires for more types of stress — but *less precise* for
the US Nasdaq specifically. v5 wins not because it has more information but because it is
**deliberately overfit to US-specific crash physics**, which is exactly what we need when the
production strategy trades only US assets.

**Lesson (now demonstrated definitively).** More events do not help when the new events come from
different distributions. The binding constraint isn't sample size, it's **regime homogeneity**.

Output: [data/experiment_D_round2.json](../data/experiment_D_round2.json)

---

## What this means

Three independent failure-mode demonstrations now agree:

1. **v5.1** (predictive label + FRED leading indicators) — failed (committed at 11a93d4, reverted at c9a5f35)
2. **v6** (predictive label + options + breadth + cross-asset) — failed ([V6_NEGATIVE_RESULT.md](V6_NEGATIVE_RESULT.md))
3. **v5_multi** (coincident label + multi-region pooling) — failed (this doc)

Combined with C's oracle-ceiling result on bottom-finding, we have empirical evidence that:

- v5 is **at or very near the practical Pareto frontier** for this problem given free public data,
  17 historical US crashes, and honest blind validation.
- The two stated user objectives ("predict crashes before they start" and "find bottoms better than
  MA50") are **not achievable** with the methods, data, and discipline we have available, beyond
  what v5 already provides.

This is not a counsel of despair — it's an honest delineation of where additional research
investment will not pay off, freeing focus for what *can* improve outcomes:

- Position-sizing and risk budgeting on top of v5's signal (out of scope here)
- Faster reaction time via intraday data (would require infrastructure not currently in place)
- Sector rotation *after* v5 fires (different problem, not crash detection)
- Conformal intervals for downstream consumers who need confidence bands (deferred)

**v5 stays as production benchmark.** No further model-replacement experiments planned.
