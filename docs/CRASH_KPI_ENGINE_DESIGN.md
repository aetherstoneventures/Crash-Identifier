# Crash KPI Engine — v6 Design Document

> **Status:** APPROVED design, implementation in progress on branch `v6-kpi-engine`.
> **Supersedes:** v5 production model, statistical_v3, forward_risk (all archived at tag `pre-v6-archive` and on branch `v5-benchmark-protected`).
> **Author:** Designed jointly with the project owner, May 2026.

---

## 1. Problem Statement (verbatim from the owner)

> Training a model on 10 historical crashes is like training a cancer detector on 10 patients with 5 different diseases. The model should know all types of crashes, but should be able to distinguish between them using proper mathematical and statistical KPIs. This is the part where math, statistics, sentiment, economical data and political and geopolitical dynamics play a role! These are patterns that model should learn but distinguish between. The first thing the model should do for "present" or "future" is to recognize "something could be wrong". Then, it should analyze it based on the similarities with previous crashes and see if there are overwhelmingly similar mathematical, statistical, economical, sentimental and political KPIs. But most importantly: when we say crash, this should be a tunable parameter! i.e. we should be able to extract the same x% reductions in the market from the historical data. x could be 2% or 20%, does not matter!!! For the data or "crash" part, it is like a matching algorithm in the historical data based on several indicators of the price reduction.

> [Layers 1, 2, 3] are good as long as they all take effect SIMULTANEOUSLY and with 100% AGREEMENT to my point 1, i.e. they should be added to OTHER SOPHISTICATED MATHEMATICAL AND STATISTICAL learning parameters of the model in 1!

## 2. Goals and Non-Goals

### Goals
- **Tunable crash threshold x% at INFERENCE time** — not baked into training.
- **Tunable horizon h ∈ {21, 63, 126, 252} trading days** at inference.
- **Anomaly-first detection:** the system first answers "is something wrong now?" before answering "what kind of crash."
- **Historical analog matching** by KPI similarity (math, statistics, economic, options-implied sentiment, geopolitical).
- **Simultaneous-agreement gating** of Layers 1/2/3 as a final precision filter.
- **Honest probabilistic output** with explicit uncertainty and visible failure modes ("no good analogs found" / "low confidence").
- **Falsifiable validation:** strict walk-forward + single-shot BLIND ≥ 2021-01-01 + pre-declared kill criteria.

### Non-Goals
- Predicting the exact date of a crash. Output is a *probability distribution and a list of historical analogs*, not a calendar.
- Beating buy-and-hold by a large margin. Realistic success = equal CAGR with materially lower MaxDD, or +1–2pp CAGR at comparable MaxDD.
- Using quantum methods. Current quantum ML is not production-ready for this problem; including it would be cargo-cult sophistication.
- Reusing social-media or news-NLP sentiment for index crashes (literature shows it is coincident-to-lagging; the causal arrow runs price → sentiment).

## 3. Architecture Overview — Five Engines + Bayesian Aggregator + Layer Gate

```
                        ┌──────────────────────────────────────────────┐
                        │             FEATURE VECTOR x(t)              │
                        │  ~40 features, point-in-time, expanding-z    │
                        │  Math/Stats | Macro | Options-Implied        │
                        │  Sentiment | Breadth | Geopolitical          │
                        └──────────────────────────────────────────────┘
                                          │
        ┌─────────────────┬───────────────┼───────────────┬─────────────────┐
        ▼                 ▼               ▼               ▼                 ▼
  ┌───────────┐    ┌───────────┐   ┌───────────┐   ┌───────────┐    (features
  │ ENGINE 1  │    │ ENGINE 2  │   │ ENGINE 3  │   │ ENGINE 4  │     also feed
  │ Density   │    │ Regime    │   │ Analog    │   │ Causal /  │     the gate)
  │ anomaly   │    │ switching │   │ matcher   │   │ structural│
  │ Mahal +   │    │ HMM 3–4   │   │ LMNN-kNN  │   │ Dyn. fac. │
  │ IsoForest │    │ states    │   │ + GP reg. │   │ + Granger │
  │           │    │           │   │ (tunable  │   │ / TE      │
  │ "wrong?"  │    │ "regime?" │   │  x%, h)   │   │ "why?"    │
  └─────┬─────┘    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
        │                │               │               │
        └────────────────┴───────┬───────┴───────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │   ENGINE 5 — BAYESIAN        │
                  │   AGGREGATOR                 │
                  │   P(maxDD ≥ x% in [t,t+h])   │
                  │   + credible interval        │
                  │   + analog list              │
                  │   + mechanism narrative      │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │   LAYER 1/2/3 GATE           │
                  │   ACTION only if ALL agree:  │
                  │   L1 macro regime elevated   │
                  │   L2 tactical stress elevated│
                  │   L3 price/vol confirms      │
                  │   Aggregator prob > τ        │
                  │   Aggregator conf > κ        │
                  └──────────────────────────────┘
```

## 4. Feature Vector Specification

Point-in-time daily vector x(t) ∈ ℝ^d, d ≈ 40. All features computed with strict no-leakage discipline (expanding-window means/stds for z-scores; lookback windows entirely in the past).

### 4.1 Mathematical / Statistical (~12)
- Realized vol z-score (21, 63, 252 d)
- Realized skew (63 d)
- Realized kurtosis (63 d)
- Drawdown from 252 d high
- Days since 252 d high
- Return autocorrelation (10, 20 d)
- Hurst exponent (252 d)
- ATR-style range expansion z-score
- Realized–implied vol gap (when VIX available)

### 4.2 Macro / Economic (~10)
- Yield curve slopes: 10Y–3M, 10Y–2Y, plus Δ
- HY OAS credit spread + Δ
- Investment-grade OAS spread + Δ
- Chicago Fed NFCI level + Δ
- ISM Manufacturing PMI level + Δ
- Sahm rule indicator
- CAPE z-score vs 30 y history
- M2 YoY z-score
- Margin debt (FINRA) z-score
- Real rates (10Y TIPS)

### 4.3 Options-Implied Sentiment (~6) — KEPT
- VIX level + z-score
- VIX term structure (1m / 3m)
- 25-delta skew (puts vs calls IV)
- Put/Call ratio (with regime caveat)
- EPU index z-score
- GPR index z-score

### 4.4 Breadth / Technical (~6)
- % stocks above 50d MA
- % stocks above 200d MA
- Advance/decline cumulative z-score
- New highs minus new lows
- Cross-asset correlation regime indicator
- DXY z-score

### 4.5 Geopolitical / Event (~3)
- GPR threats vs acts decomposition
- Fed meeting / election proximity (binary windows)
- Earnings season indicator

### 4.6 EXPLICITLY DROPPED
- Reddit sentiment (coincident-to-lagging for indices)
- Twitter/X sentiment (same)
- News NLP composite for index crashes (Tetlock-style works for single stocks, not for index timing)
- Google Trends FEARS (weak contrarian only)

**Sentiment drop is reversible:** the architecture supports re-adding any feature; the prior is "literature + causal-arrow argument says drop," not "we proved it on this dataset." If a future ablation shows lift, we re-add.

## 5. Engine Specifications

### 5.1 Engine 1 — Density Anomaly Detector
- **Method:** Mahalanobis distance + Isolation Forest, ensembled by rank averaging.
- **Training set:** "normal" days = days outside ±20 trading days of any historical drawdown ≥ x_train% (x_train = 10% conservative default; this is a *training-set definition*, NOT the inference threshold).
- **Mahalanobis:** μ, Σ estimated on expanding window through t−1. Score ~ χ²_d under null.
- **Isolation Forest:** 200 trees, contamination = 0.05, expanding-window re-fit annually.
- **Output:** `anomaly_score ∈ [0, 1]` (rank-averaged), interpretable threshold 0.99 = alert.
- **Failure mode:** if both methods diverge by > 0.3 rank, flag as "ambiguous regime" (low confidence).

### 5.2 Engine 2 — Regime-Switching Model
- **Method:** Hidden Markov Model, K = 3 or 4 states, Gaussian emissions on a reduced feature set (vol, drawdown, credit spread, yield curve, NFCI).
- **Estimation:** Baum-Welch / EM on expanding window, re-fit quarterly.
- **States interpreted post-hoc** as: calm-bull / normal / stress / crisis (we name, not impose).
- **Output:** P(state_t = k) for each k; transition matrix → P(state_{t+h} = crisis | state_t).
- **References:** Hamilton (1989), Ang & Bekaert (2002), Kim & Nelson (1999).

### 5.3 Engine 3 — Historical Analog Matcher
- **Method:** k-NN with learned Mahalanobis metric via LMNN (Weinberger & Saul 2009), k=50. Alternative head: Gaussian Process regression with RBF kernel for forward-outcome posterior.
- **Tunable x% lives here:** retrieval is by KPI similarity; forward-outcome labeling by x% happens at inference. One trained model serves all x ∈ [2%, 50%] and all h ∈ {21, 63, 126, 252}.
- **Training:** metric learned on expanding window with target = forward maxDD bucket (no leakage); re-fit annually.
- **Output for user query (x, h):**
  - $\hat{F}_t(x, h) = \frac{1}{k}\sum_{i=1}^{k} \mathbf{1}[\text{maxDD}_{(t_i, t_i+h]} \geq x]$
  - Confidence: ratio of distance-to-1st-NN over distance-to-50th-NN (tight cluster = high)
  - Top-10 analog dates returned with their forward paths.

### 5.4 Engine 4 — Causal / Structural Factor Model
- **Method (a):** Dynamic factor model with regime-dependent loadings, K=5 latent factors (growth, inflation, liquidity, risk-appetite, dollar). Kalman filter; loadings switch with Engine 2 regime.
- **Method (b):** Pairwise Granger causality + transfer entropy (Schreiber 2000) on a 6–8 node network (equity, credit, vol, dollar, rates, oil). Rolling 252 d windows.
- **Output:**
  - Factor connectedness index (Diebold–Yilmaz 2012 spillover).
  - Causal-structure-break score: KL divergence between current connectedness matrix and trailing 252 d median.
  - Dominant-driver tag for current period ("credit-led", "vol-led", "exogenous").
- **References:** Billio, Getmansky, Lo, Pelizzon (2012); Ang & Bekaert (2002); Diebold & Yilmaz (2012).

### 5.5 Engine 5 — Bayesian Aggregator
- **Method:** Bayesian model averaging with per-engine likelihood weights learned on past walk-forward folds; Beta-Binomial conjugate prior on each engine's calibration.
- **Output:**
  - Posterior $P(\text{maxDD}_{(t,t+h]} \geq x\%)$ with 90% credible interval.
  - Per-engine contribution decomposition (transparency).
  - Confidence score = inverse of posterior interval width.
- **No black-box stacking.** Weights and contributions are inspectable.

### 5.6 Layer 1 / 2 / 3 Gate (your "100% agreement" constraint)
- **L1 Macro regime score:** weighted sum of macro features → z-score → indicator above 1.5σ.
- **L2 Tactical stress score:** weighted sum of vol / breadth / credit-accel features → indicator above 1.5σ.
- **L3 Price confirmation:** drawdown from 252 d high ≥ user-defined x_confirm (default 2%) OR MA50 break.
- **ACTION gate fires only when:**
  - Aggregator $P(\text{maxDD} \geq x\%) > \tau$ (default 0.6) AND
  - Aggregator confidence > κ (default 0.5) AND
  - L1 elevated AND L2 elevated AND L3 confirms.
- Operationalized as a multiplicative prior in the aggregator step + a hard AND at the action step.

## 6. Validation Protocol

- **Walk-forward folds** matching v5 boundaries for direct comparison: <1999, <2005, <2012, <2020.
- **BLIND set:** all data ≥ 2021-01-01. **Single-shot evaluation. No retuning permitted.** Pre-commit alarm config and feature set before touching BLIND.
- **Calibration metrics:**
  - Reliability diagrams for $\hat{P}(\text{maxDD} \geq x)$ at x ∈ {5, 10, 15, 20}% × h ∈ {21, 63, 126, 252} d
  - Brier score, log loss
  - Coverage of credible intervals
- **Decision metrics (when gate converted to exit/re-entry rule):**
  - CAGR, MaxDD, Sharpe, Sortino vs buy-and-hold
  - 5 bps slippage per switch
  - Hit rate / false-alarm rate at the gate level
- **Anomaly metrics:**
  - Engine 1 AUC for "is in pre-crash window" (informational only — NOT optimization target).

## 7. Kill Criteria (pre-declared)

The v6 KPI engine is declared a FAILURE and the project pivots if any of:

1. BLIND aggregator calibration error > 10 percentage points (reliability slope < 0.5 or > 1.5).
2. BLIND backtest CAGR < buy-and-hold CAGR − 2 pp.
3. BLIND MaxDD > buy-and-hold MaxDD × 1.1.
4. Engine disagreement structurally degenerate (one engine carries all weight).
5. Layer gate never fires OR fires > 10% of days (degenerate behavior).

Failure is acceptable and will be documented in `docs/V6_HONEST_SCORECARD.md` either way.

## 8. Implementation Plan

All work on branch `v6-kpi-engine`. Each milestone is independently committable and testable.

| Step | Module | Deliverable | Commit prefix |
|---|---|---|---|
| 0 | (this doc) | Design approved | `docs(v6)` |
| 1 | repo cleanup | Legacy v5/v3/forward_risk archived locally, working tree v6-only | `chore(v6)` |
| 2 | `src/v6/features/` | Feature vector builder (~40 features, leakage-safe) | `feat(v6)` |
| 3 | `src/v6/engines/anomaly.py` | Engine 1 (Mahal + IsoForest) | `feat(v6)` |
| 4 | `src/v6/engines/regime.py` | Engine 2 (HMM) | `feat(v6)` |
| 5 | `src/v6/engines/analog.py` | Engine 3 (LMNN k-NN + GP) | `feat(v6)` |
| 6 | `src/v6/engines/causal.py` | Engine 4 (dynamic factor + Granger/TE) | `feat(v6)` |
| 7 | `src/v6/engines/aggregator.py` | Engine 5 (Bayesian aggregator) | `feat(v6)` |
| 8 | `src/v6/gate.py` | L1/L2/L3 simultaneous gate | `feat(v6)` |
| 9 | `scripts/v6/walkforward.py` | Walk-forward training + validation harness | `feat(v6)` |
| 10 | `scripts/v6/blind_eval.py` | Single-shot BLIND evaluator (locked) | `feat(v6)` |
| 11 | `src/dashboard/pages/v6_kpi_engine.py` | New dashboard surface | `feat(v6)` |
| 12 | `docs/V6_HONEST_SCORECARD.md` | Honest results doc, kill-criterion verdict | `docs(v6)` |
| 13 | Tag `v6.0.0` | Versioned release | — |

## 9. What is being removed/archived

Preserved at git tag `pre-v6-archive` and branch `v5-benchmark-protected` (both pushed to origin). Removed from working tree on `v6-kpi-engine`:

- `scripts/training/train_*_v5*`, `train_statistical_*`, `train_advanced_*`, `train_crash_detector_*`, `train_bottom_predictor.py`, `train_hybrid_ensemble_walkforward.py`
- `scripts/utils/v5_backtest.py`, `scripts/utils/generate_predictions_v5.py`, `scripts/utils/generate_bottom_predictions.py`
- `scripts/forward_risk/*`
- `scripts/evaluation/evaluate_*`
- `models/crash_detector_advanced/`, `models/statistical_v3/`, `catboost_info/`
- Pre-v5 docs already bannered stale: `ARCHITECTURE.md`, `METHODOLOGY.md`, `REPRODUCIBILITY_GUIDE.md`, `BOTTOM_PREDICTOR_FIX.md`, `DASHBOARD_IMPORT_FIX.md`, etc.
- Dashboard pages: `v5_production.py`, `forward_risk.py`, `statistical_v3.py` (and any related)
- Test files for the above

Anything ambiguous gets moved to `legacy/` rather than deleted, then deleted once v6 is green.

## 10. Versioning

- Branch: `v6-kpi-engine` (working).
- Archive tag: `pre-v6-archive` (immutable snapshot of pre-v6 main).
- Protected branch: `v5-benchmark-protected` (frozen).
- Future tags: `v6.0.0-alpha` after Engine 5 wired; `v6.0.0` after BLIND verdict (pass or fail).
- Merge to `main` only after BLIND verdict and explicit owner approval.

## 11. Honest Expected Outcomes

- **Realistic best case:** equal CAGR vs B&H, MaxDD reduced by 30–50%, Sharpe up by 0.2–0.4.
- **Realistic median case:** ±1pp CAGR vs B&H, MaxDD reduced by 10–25%, Sharpe up by 0.0–0.2.
- **Realistic worst case:** kill criterion triggered, project declared dead, lessons documented.

All three outcomes are valuable. The honest verdict matters more than the result.
