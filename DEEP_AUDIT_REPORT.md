# DEEP MATHEMATICAL & CODE AUDIT REPORT
## Market Crash Predictor — Comprehensive Review

**Audit Date**: March 29, 2026  
**Auditor**: Deep quantitative & software engineering review  
**Severity Scale**: 🔴 CRITICAL | 🟠 SERIOUS | 🟡 MODERATE | ⚪ MINOR

---

## EXECUTIVE SUMMARY

This audit uncovered **6 critical**, **5 serious**, and **7 moderate** issues. The system's reported 81.8% recall figure is unreliable due to fundamental mathematical errors in feature construction, look-ahead bias in labeling, and circular/fabricated indicators. Several "indicators" are not real data but synthetic proxies derived from other features already in the model, creating **multicollinearity illusions** that artificially inflate model confidence.

**The system, in its current state, would NOT survive scrutiny at any serious quantitative fund.**

---

## 🔴 CRITICAL ISSUES

### C1. Yield Spread Double-Counting / Wrong Calculation
**File**: `src/feature_engineering/crash_indicators.py` lines 89-100  

The FRED series `T10Y3M` already **IS** the 10-Year minus 3-Month spread. The FRED series `T10Y2Y` already **IS** the 10-Year minus 2-Year spread. But the code computes:

```python
def yield_spread_10y_3m(df):
    return df['yield_10y'] - df['yield_10y_3m']  # WRONG!
    # This computes: DGS10 - T10Y3M = 10Y_yield - (10Y - 3M spread)
    # = 10Y - 10Y + 3M = 3M yield (NOT the spread!)

def yield_spread_10y_2y(df):
    return df['yield_10y'] - df['yield_10y_2y']  # WRONG!
    # This computes: DGS10 - T10Y2Y = 10Y_yield - (10Y - 2Y spread)
    # = 2Y yield (NOT the spread!)
```

**What the model thinks it has**: Yield curve spreads (recession predictors)  
**What it actually has**: Short-rate levels (completely different signal)  
**Impact**: The #1 ranked recession predictor (yield curve inversion) is **not present** in the model. The statistical models check `yield_10y_2y < 0` to detect inversion, but the feature column contains the 2Y yield level (always positive), so **inversion is never detected**.

---

### C2. Shiller PE (CAPE) Fabricated from VIX
**File**: `src/feature_engineering/crash_indicators.py` lines 234-243  

```python
def shiller_pe(df):
    vix = df['vix_close']
    return 25 * (1 - (vix - vix.min()) / (vix.max() - vix.min() + 0.001))
```

This is **not** the Shiller PE ratio. It's an inverted, min-max scaled VIX. The Shiller CAPE is the ratio of the S&P 500 price to a 10-year average of inflation-adjusted earnings — a fundamentally different economic concept (valuation vs. volatility).

**Problems**:
- Uses future information: `vix.min()` and `vix.max()` look over the **entire** dataset, including future data. This is a **look-ahead bias**.
- Creates a **circular feature**: VIX already exists as its own feature. This "Shiller PE" is just a mathematical transformation of VIX, adding redundant information and inflating the apparent feature count.
- The statistical model checks `shiller_pe > 35` for overvaluation, but this fabricated version ranges 0–25 and can never exceed 25, so the rule **never fires**.

---

### C3. At Least 10 of 28 "Indicators" Are Fabricated Proxies
**File**: `src/feature_engineering/crash_indicators.py`  

The system claims "28 indicators" but many are synthetic proxies derived from features already in the model:

| Claimed Indicator | Actually Is | Real Data? |
|---|---|---|
| `shiller_pe` | Inverted VIX (min-max scaled) | ❌ Fabricated |
| `buffett_indicator` | SP500 × 1000 / GDP | ❌ Crude proxy |
| `sp500_pb_ratio` | 10-year SP500 momentum × 3 | ❌ Fabricated |
| `earnings_yield_spread` | 5/(VIX/20+0.1) - 10Y yield | ❌ Fabricated |
| `put_call_ratio` | 1.0 + VIX_change × 0.5 | ❌ Fabricated from VIX |
| `margin_debt` | 100 / (credit_spread + 1) | ❌ Fabricated from credit spread |
| `margin_debt_growth` | -credit_spread.pct_change() | ❌ Fabricated from credit spread |
| `debt_service_ratio` | 0.6×credit_spread + 0.4×unemployment | ❌ Fabricated combo |
| `corporate_debt_growth` | -credit_spread.pct_change() | ❌ Fabricated from credit spread |
| `household_debt_growth` | consumer_sentiment.pct_change() | ❌ Fabricated from sentiment |
| `market_breadth` | 50 + SP500 50-day momentum | ❌ Fabricated from SP500 |
| `credit_gap` | M2/GDP deviation from 40-period mean | 🟡 Proxy |

**Impact**: The model effectively has ~15 real features, not 28. Multiple features are mathematical transformations of the same underlying data (VIX or credit spread), creating severe **multicollinearity**. ML models will learn these correlations and overfit to noise. The "39 engineered features from 20 raw indicators" claim is misleading — it's more like 15 real signals with 24 mathematical echoes.

---

### C4. Look-Ahead Bias in Feature Engineering
**File**: `src/feature_engineering/crash_indicators.py`  

Several feature computations use statistics of the **entire** dataset:

```python
# shiller_pe uses vix.min() and vix.max() — FULL dataset min/max
return 25 * (1 - (vix - vix.min()) / (vix.max() - vix.min() + 0.001))

# sp500_pb_ratio clips using absolute bounds
return 3 * (1 + momentum_10y / 100).clip(0.5, 5)

# debt_service_ratio normalizes using full-dataset min/max
cs_norm = (credit_spread - cs_min) / (cs_max - cs_min + 0.001) * 100
un_norm = (unemployment - un_min) / (un_max - un_min + 0.001) * 100
```

These calculations use `min()` and `max()` across the entire time series, which means the feature values at any given date incorporate information from the future. In walk-forward validation, features computed on the training set still contain future information baked into the normalization.

---

### C5. Crash Labeling Look-Ahead in CrashLabeler
**File**: `src/models/crash_prediction/crash_labeler.py` lines 49-58  

```python
for i in range(len(prices) - self.lookforward_window):
    current_price = prices.iloc[i]
    future_prices = prices.iloc[i:i + self.lookforward_window]
    max_price = future_prices.max()  # BUG: includes future prices
    min_price = future_prices.min()
    drawdown = (min_price - current_price) / current_price
```

The drawdown is calculated as `(min_future - current) / current`, but the `max_price` variable (which uses future data) is computed but never used. The actual drawdown logic checks if the price drops X% from `current_price`, but `current_price` may not be the peak — it could already be mid-crash. The correct approach for crash labeling is to check if price drops X% from its **rolling maximum** (peak), not from the current price.

**Example**: If the market is already down 15% from peak and currently at price 100, a further drop to 85 (15% from current, 28% from peak) would be labeled crash, but a drop to 81 from a peak of 100 (19% from peak, 19% from current) would NOT be labeled — even though it's clearly a significant crash from peak.

---

### C6. Inconsistent Labeling Across Training Scripts
**Files**: `scripts/training/train_crash_detector_v5.py` vs `scripts/training/train_advanced_models.py` vs `src/models/crash_prediction/crash_labeler.py`

Three completely different labeling schemes exist:

1. **CrashLabeler** (`crash_labeler.py`): Forward-looking drawdown from current price over 60 days
2. **V5 script** (`train_crash_detector_v5.py`): Labels 90 days **before** known crash start dates as positive (pure look-ahead — you need to know crash dates in advance)
3. **Advanced script** (`train_advanced_models.py`): Labels `lookforward` days before each crash event start as positive

The V5 script's approach (labeling based on known future crash dates) is technically valid for **backtesting** historical crashes but is **not a generalizable prediction model** — it requires knowing crashes in advance to create labels, making it impossible to generate labels for new, unseen data outside known crash periods.

---

## 🟠 SERIOUS ISSUES

### S1. Feature Pipeline Fits Scaler on Full Dataset
**File**: `src/feature_engineering/feature_pipeline.py` line 95  

```python
features_normalized[cols_to_scale] = self.scaler.fit_transform(features_normalized[cols_to_scale])
```

The `StandardScaler` is fit on the entire dataset before any train/test split occurs. This means the mean and standard deviation used for normalization include information from the test set — a form of **data leakage**. In walk-forward validation, this partially invalidates the "no temporal leakage" claim.

---

### S2. Redundant Feature Removal Is Non-Deterministic
**File**: `src/feature_engineering/feature_pipeline.py` lines 130-148  

When two features are correlated above 0.95, the code always drops the second one (`col_j`). But the correlation matrix iteration order is arbitrary. This means:
- The feature set can change across runs
- Real important features may be dropped in favor of fabricated ones
- A fabricated proxy (e.g., margin_debt from credit spread) might be kept while the real credit spread is dropped

---

### S3. Statistical Models Reference Missing Features
**File**: `src/models/crash_prediction/improved_statistical_model.py`, `statistical_model_v3.py`  

The statistical models check features like `vix_change_pct`, `credit_spread_change`, `sp500_drawdown`, `unemployment_change`, `industrial_production_change`, `sp500_return_5d`, `sp500_return_20d`, `vix_change_5d` — but these features are **not created** in `crash_indicators.py`. The 28-indicator pipeline creates `vix_change_rate` (not `vix_change_pct`), `sp500_drawdown` (in percent, not decimal), etc. Field name mismatches mean many statistical rules **silently never fire** (because `pd.notna` on a missing key returns False).

---

### S4. LSTM Prediction Padding Introduces Zeros
**File**: `src/models/crash_prediction/lstm_crash_model.py` line 262-264  

```python
padded_predictions = np.zeros(len(X))
padded_predictions[self.sequence_length:] = predictions.flatten()
```

The first `sequence_length` (60) predictions are set to 0.0, meaning the model always predicts "no crash" for the first 60 days of any prediction window. In walk-forward validation, this systematically biases metrics.

---

### S5. Bry-Boschan Implementation Has Off-by-One Logic
**File**: `src/feature_engineering/regime_detection.py` lines 128-145  

The peak/trough filtering uses `troughs[i-1]` to find the "previous trough before peak i", but the peaks and troughs lists are independent — peak index `i` doesn't necessarily correspond to trough index `i-1`. This can produce incorrect regime classifications.

---

## 🟡 MODERATE ISSUES

### M1. Buffett Indicator Proxy Is Economically Meaningless
**File**: `src/feature_engineering/crash_indicators.py` lines 247-259  

```python
market_cap_proxy = sp500 * 1000
return market_cap_proxy / gdp * 100
```

The real Buffett Indicator is total US market capitalization / GDP. Multiplying the S&P 500 **index level** by 1000 is not a valid proxy — the index is a price-weighted/cap-weighted number, not a market cap. The ratio of (SP500 × arbitrary constant) / GDP has no economic meaning and will produce wildly different signals from the real Buffett Indicator.

---

### M2. SP500 P/B Ratio Proxy Is Nonsensical
```python
momentum_10y = sp500.pct_change(periods=252*10) * 100
return 3 * (1 + momentum_10y / 100).clip(0.5, 5)
```

Price momentum over 10 years has no mathematical relationship to price-to-book ratio. Book value is an accounting metric. This feature is pure noise dressed up as a valuation signal.

---

### M3. Earnings Yield Spread Proxy Is Circular
```python
earnings_yield_proxy = 5 / (vix / 20 + 0.1)
return earnings_yield_proxy - yield_10y
```

Earnings yield = E/P (inverse of P/E). Using VIX as a proxy for this is economically unfounded. High VIX ≠ low earnings. This creates another VIX derivative feature.

---

### M4. Walk-Forward Validator Is Minimal
**File**: `src/utils/walk_forward_validation.py`  

The entire walk-forward validator is 20 lines. It generates index splits but:
- Doesn't apply SMOTE within each fold (the V5 script does this manually)
- Doesn't re-fit the scaler per fold
- Doesn't track per-fold metrics for variance analysis
- The `expanding_window_split` generates many overlapping test windows, inflating the sample count

---

### M5. `consumer_sentiment` Mean-Fill Creates Look-Ahead
**File**: `src/feature_engineering/crash_indicators.py` line 291  

```python
return df['consumer_sentiment'].fillna(df['consumer_sentiment'].mean())
```

Using the mean of the entire series (including future values) to fill missing data is look-ahead bias.

---

### M6. GDP Growth Uses Wrong Period Count
**File**: `src/feature_engineering/crash_indicators.py` line 368  

```python
def gdp_growth(df, periods=4):
    return (df['real_gdp'] / df['real_gdp'].shift(periods) - 1) * 100
```

With daily data, `shift(4)` computes 4-day growth, not quarterly/annual. For quarterly GDP data resampled to daily, `shift(4)` is meaningless — you need `shift(~63)` for quarterly or `shift(~252)` for annual.

---

### M7. Sahm Rule Period Window Is Wrong for Daily Data
```python
def sahm_rule(df):
    u3ma = df['unemployment_rate'].rolling(3).mean()
    min_12m = u3ma.rolling(12, min_periods=1).min()
    return u3ma - min_12m
```

The Sahm Rule uses a 3-**month** moving average minus the 12-**month** minimum. With daily data, `rolling(3)` is 3 days, not 3 months. Should be `rolling(63)` for 3 months and `rolling(252)` for 12 months.

---

## ⚪ MINOR ISSUES

1. **ImprovedStatisticalModel.__init__** calls `super().__init__()` without `name` argument but `BaseCrashModel.__init__` requires it → will crash at runtime.
2. XGBoostCrashModel.predict returns probabilities, not binary predictions, contradicting the `BaseCrashModel.predict` contract.
3. `conftest.py`, `fix_all_session_bugs.py`, `test_stat_model_simple.py` are dev artifacts that shouldn't be in the repo root.
4. Config contains `LSTM_LEARNING_RATE` reference in `advanced_lstm_bottom_model.py` but it's not defined in `config.py`.
5. `AlternativeCollector.fetch_shiller_pe` always returns an empty series — the actual Shiller PE data is never fetched.

---

## SUMMARY TABLE

| # | Severity | Issue | Impact |
|---|----------|-------|--------|
| C1 | 🔴 CRITICAL | Yield spread calculated wrong (computes short rate, not spread) | Yield curve inversion — the strongest recession signal — is broken |
| C2 | 🔴 CRITICAL | Shiller PE fabricated from VIX with look-ahead | Fake feature with future data leakage |
| C3 | 🔴 CRITICAL | 10+ of 28 indicators are fabricated proxies | Multicollinearity, inflated feature count |
| C4 | 🔴 CRITICAL | Look-ahead bias in feature normalization | Future info leaks into training data |
| C5 | 🔴 CRITICAL | CrashLabeler uses wrong reference point for drawdown | Mislabeled crash periods |
| C6 | 🔴 CRITICAL | Three inconsistent labeling schemes | Can't reproduce or trust results |
| S1 | 🟠 SERIOUS | Scaler fit on full dataset | Data leakage in normalization |
| S2 | 🟠 SERIOUS | Non-deterministic feature removal | Unreproducible feature sets |
| S3 | 🟠 SERIOUS | Statistical models reference missing features | Rules silently never fire |
| S4 | 🟠 SERIOUS | LSTM pads first 60 predictions with zero | Biased metrics |
| S5 | 🟠 SERIOUS | Bry-Boschan peak/trough matching bug | Wrong regime detection |
| M1–M7 | 🟡 MODERATE | Various proxy/math errors | Degraded signal quality |

---

## RECOMMENDATIONS

### Immediate (Must-Fix)
1. **Fix yield spread**: Use `T10Y3M` and `T10Y2Y` directly as the spreads they are — do NOT subtract them from `DGS10`
2. **Remove all fabricated indicators**: Replace with real data or remove entirely. Be honest about having ~15 real features
3. **Fix normalization**: Use rolling/expanding window stats, never full-dataset min/max/mean
4. **Standardize labeling**: Pick ONE labeling scheme, document it, and use it everywhere
5. **Fix Sahm Rule and GDP growth**: Use correct period windows for daily data

### Short-Term
6. Fetch real Shiller PE data from Robert Shiller's website or Quandl
7. Fetch real FINRA margin debt data (the code to do this exists but falls back to proxy)
8. Re-fit scaler within each walk-forward fold
9. Add purging/embargo to walk-forward splits to prevent leakage at boundaries
10. Fix statistical model feature name mismatches

### Medium-Term
11. Honest re-evaluation of model performance with corrected features
12. Consider a simpler model with fewer, real features over a complex model with fabricated ones
13. Add proper backtesting framework with transaction costs and slippage
14. Implement proper out-of-sample testing on held-out years never seen during development

---

*"In God we trust. All others must bring data." — W. Edwards Deming*  
*The data must be REAL data, not VIX wearing a Shiller PE costume.*
