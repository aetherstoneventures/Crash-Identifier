# Forward Risk — 1-month NASDAQ probabilistic forecast

**Status:** Partial-ship. Only the 21-trading-day (1-month) horizon is operational;
3 / 6 / 12-month horizons were **shelved** after BLIND audit.
**Frozen on:** 2026-04-29.
**Single-shot BLIND evaluation date:** 2026-04-29 (predictions span 2021-01-04 → 2026-04-27).

---

## 1. What this module is (and isn't)

**Is.** A calibrated 21-day-ahead probabilistic forecast for the NASDAQ Composite,
producing a 5-quantile fan ($q_{0.05}, q_{0.25}, q_{0.50}, q_{0.75}, q_{0.95}$) for
two targets:
- forward 21-day log-return $\log(P_{t+21}/P_t)$
- forward 21-day worst peak-to-trough log-drawdown $\min_{k \in (0, 21]} \log(P_{t+k} / \max_{j \le k} P_{t+j})$

**Isn't.** A long-horizon predictor (3/6/12-month horizons failed BLIND — see §5).
Not a crash-timing oracle. Not coupled to v5; the two answer different questions.

**Difference vs v5:** v5 detects crashes that are already underway (near-coincident).
Forward Risk gives a 1-month forward distribution that includes both directions.

---

## 2. Data

- **Asset:** NASDAQ Composite (FRED `NASDAQCOM`), 1982-01-04 → 2026-04-27, 11,169 daily rows.
- **Features:** 32 disciplined features built from the existing `indicators` table —
  yield curve (10y-3m, 10y-2y, fed funds), credit (Baa-10y), equity momentum
  (5/21/63/252d), realized vol (21d, 63d), VIX (level + z + Δ), drawdown state,
  macro Δ (unemployment, IP, LEI, CPI, M2), sentiment z, EPU, dollar TWI Δ, WTI oil Δ + vol.
  See [build_features.py](../scripts/forward_risk/build_features.py).
- **Free-data limitation.** Yahoo Finance returns HTTP 401 anonymously since 2024;
  Stooq returns dead 342-byte stubs for indices/ETFs; FRED has no Russell or
  international indices. Cross-asset / multi-equity breadth (originally specced
  with 5 assets) was **infeasible without a paid feed**, so this module is
  NASDAQ-only.

---

## 3. Method

### 3.1 Quantile regression
LightGBM with `objective="quantile"`, fixed hyperparameters chosen from priors
(no search touching BLIND): `n_estimators=400`, `lr=0.03`, `num_leaves=15`,
`max_depth=4`, `min_data_in_leaf=50`, `feature_fraction=0.8`, `bagging_fraction=0.8`,
`reg_alpha=0.1`, `reg_lambda=0.1`. Five quantiles × four horizons × two targets =
40 base models.

### 3.2 Walk-forward folds
Same boundaries as v5 (no regime peeking):

| Fold | Train | Eval |
|------|-------|------|
| 1 | < 1999-01-01 | [1999-01-01, 2005-01-01) |
| 2 | < 2005-01-01 | [2005-01-01, 2012-01-01) |
| 3 | < 2012-01-01 | [2012-01-01, 2020-01-01) |
| 4 | < 2020-01-01 | [2020-01-01, 2026-12-31] (contains BLIND ≥ 2021-01-01) |

Target leakage guard: when training with cutoff $e_{train}$, rows with $t > e_{train} - h$
are dropped from the training set (since $y_{t,h}$ uses prices through $t+h$).

### 3.3 Conformal recalibration
Split-conformal CQR (Romano, Patterson, Candès, 2019). Conformity score per
calibration row:
$$ s_i = \max(q_{lo}(x_i) - y_i,\; y_i - q_{hi}(x_i)) $$
At target miscoverage $\alpha$, $\tau = \lceil (1-\alpha)(n+1) \rceil$-th order
statistic of $\{s_i\}$. Adjusted interval: $[q_{lo} - \tau,\; q_{hi} + \tau]$.

**Calibration set:** folds 1–3 OOF predictions (1999-01-01 → 2019-12-31), n=5,284
per cell. These are true out-of-fold residuals — never seen by the model that
produced them, never overlap with BLIND.

---

## 4. BLIND results (single shot, evaluated 2026-04-29)

| Horizon | Target | n_blind | cov₉₀ pre | cov₉₀ post | width pre | width post | Verdict |
|--------:|:------:|--------:|----------:|-----------:|----------:|-----------:|:-------:|
|  21d | ret   | 1,313 | 0.691 | **0.932** | 0.114 | 0.194 | **PASS** |
|  21d | maxdd | 1,313 | 0.816 | **0.938** | 0.083 | 0.151 | **PASS** |
|  63d | ret   | 1,271 | 0.673 | 0.980 | 0.188 | 0.425 | FAIL (over) |
|  63d | maxdd | 1,271 | 0.645 | 0.964 | 0.111 | 0.308 | FAIL (over) |
| 126d | ret   | 1,208 | 0.667 | 0.972 | 0.241 | 0.692 | FAIL (over) |
| 126d | maxdd | 1,208 | 0.792 | 1.000 | 0.201 | 0.624 | FAIL (over) |
| 252d | ret   | 1,082 | 0.528 | 1.000 | 0.373 | 1.128 | FAIL (over) |
| 252d | maxdd | 1,082 | 0.739 | 1.000 | 0.271 | 0.815 | FAIL (over) |

**Kill criterion:** BLIND CI₉₀ coverage in $[0.85, 0.95]$.
**Pass rate:** 2 / 8 cells.

---

## 5. Why long horizons were shelved

The calibration period (1999-2020) contains both the dotcom crash
(NASDAQ −78%) and the GFC (NASDAQ −55%). The BLIND test period (2021-2026)
contains the 2022 inflation-shock bear at −33% — a real bear, but tamer than
the cal-set tails. Conformal sized $\tau$ to cover the heavier-tailed cal
distribution, so on BLIND the inflated intervals are systematically *too wide*.

The 12-month post-conformal CI₉₀ width is **1.13 in log-return ≈ ±113%
in simple return**. That is technically calibrated against historical tails,
but functionally it tells a user "12-month NASDAQ return is between −68% and
+209% with 90% confidence" — an honest answer, but a vacuous one.

Per the honesty contract this constitutes a kill-criterion failure (coverage
outside $[0.85, 0.95]$). Shipping these intervals would be either (a) a lie
about precision (if we shrunk them) or (b) useless to a decision-maker (if we
kept them at +113% width). Shelved.

---

## 6. Reproducibility

```bash
venv/bin/python3 -W ignore scripts/forward_risk/build_features.py
venv/bin/python3 -W ignore scripts/forward_risk/build_targets.py
venv/bin/python3 -W ignore scripts/forward_risk/train_walkforward.py
venv/bin/python3 -W ignore scripts/forward_risk/diagnose_blind.py
venv/bin/python3 -W ignore scripts/forward_risk/conformal_recalibrate.py
```

Shipped artifacts (committed to repo):
- `data/processed/forward_risk_predictions_conformal.parquet` — long-format predictions w/ original + conformal-adjusted quantiles, every date 2020-01-04 → 2026-04-27, every horizon × target. The dashboard filters to h=21.
- `data/processed/forward_risk_conformal_summary.csv` — per-cell pre/post coverage and CI widths.

Reproducible artifacts (gitignored):
- `forward_risk_features.parquet`, `forward_risk_targets.parquet`,
  `forward_risk_predictions.parquet`, `forward_risk_diagnostics/*`.

---

## 7. What this is NOT a license to do

- **Don't** re-tune the trainer on BLIND outcomes, then re-evaluate on BLIND.
- **Don't** pick a different calibration window because 1999-2020 was "too tough."
- **Don't** market this as "predicts the next crash." It predicts the next-month
  NASDAQ return distribution.
- **Don't** combine it with v5 into a single signal without a fresh,
  pre-registered evaluation protocol.
