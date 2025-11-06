# AUDIT DETAILED EVIDENCE & CODE REFERENCES

## EVIDENCE 1: Synthetic Put/Call Ratio

### Source Code
**File**: `scripts/data/collect_data.py` (lines 121-126)
```python
# Synthetic put/call ratio: Use VIX change as proxy
# Put/call ratio increases when VIX spikes (fear)
if 'vix_close' in combined_df.columns:
    vix_change = combined_df['vix_close'].pct_change()
    combined_df['put_call_ratio'] = 1.0 + (vix_change * 0.5).clip(-0.5, 0.5)
    logger.info("✅ Calculated put_call_ratio from vix_close")
```

### Database Evidence
**Query Result** (Last 5 rows from indicators table):
```
Date: 2025-11-04, Put/Call Ratio: 1.0
Date: 2025-11-03, Put/Call Ratio: 0.9922591614250125
Date: 2025-10-31, Put/Call Ratio: 1.015671220917445
Date: 2025-10-30, Put/Call Ratio: 0.9997044849634532
Date: 2025-10-29, Put/Call Ratio: 1.015225334886626
```

### Why This is Wrong
- Real CBOE put/call ratio ranges 0.5-2.5 typically
- Synthetic values cluster around 1.0 (mathematical artifact)
- Formula: 1.0 + (VIX_change × 0.5) is NOT how put/call ratio works
- User sees "put_call_ratio" column and assumes real CBOE data

### Impact on Model
- Model trained on synthetic proxy, not real market sentiment
- Predictions based on VIX changes, not actual put/call activity
- Misleads users about data quality

---

## EVIDENCE 2: Synthetic Margin Debt

### Source Code
**File**: `scripts/data/collect_data.py` (lines 114-116)
```python
# Synthetic margin debt: Use credit spread as inverse proxy
# Margin debt increases when credit spreads are tight (easy credit)
if 'credit_spread_bbb' in combined_df.columns:
    combined_df['margin_debt'] = 100 / (combined_df['credit_spread_bbb'] + 1)
    logger.info("✅ Calculated margin_debt from credit_spread_bbb")
```

### Database Evidence
**Query Result** (Sample margin_debt values):
```
2025-11-04: 49.01960784313725
2025-11-03: 49.01960784313725
2025-10-31: 49.504950495049506
2025-10-30: 50.505050505050505
2025-10-29: 51.02040816326531
```

### Why This is Wrong
- Real FINRA margin debt is in billions of dollars (e.g., $800B)
- Synthetic values are 49-51 (mathematical artifact)
- Formula: 100 / (credit_spread + 1) is NOT how margin debt works
- User sees "margin_debt" column and assumes real FINRA data

### Impact on Model
- Model trained on synthetic proxy, not real leverage data
- Predictions based on credit spreads, not actual margin activity
- Misleads users about data quality

---

## EVIDENCE 3: Temporal Leakage in Cross-Validation

### Source Code
**File**: `scripts/training/train_crash_detector_v5.py` (lines 162-180)
```python
# K-Fold Cross-Validation
logger.info("\n" + "=" * 80)
logger.info("K-FOLD CROSS-VALIDATION (5 folds)")
logger.info("=" * 80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Gradient Boosting with regularization
logger.info("\nTraining Gradient Boosting with regularization...")
gb = GradientBoostingClassifier(...)

gb_cv_scores = cross_validate(
    gb, X_train_scaled, y_train, cv=skf,
    scoring=['roc_auc', 'recall', 'precision', 'f1'],
    return_train_score=True
)
```

### The Problem
- `StratifiedKFold(shuffle=True)` randomly shuffles data
- For time-series, this means training on future data to predict past crashes
- Example: Fold 1 trains on 2020-2025 data, tests on 1980-2000 data
- This is TEMPORAL LEAKAGE - a fundamental methodological error

### Correct Approach
Should use `TimeSeriesSplit` instead:
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
# This ensures: train on past, test on future (chronologically)
```

### Impact on Metrics
- Reported AUC 0.7323 is INFLATED
- Reported Recall 81.8% is INFLATED
- Real out-of-sample performance is likely MUCH LOWER
- Model appears better than it actually is

---

## EVIDENCE 4: Outdated Documentation

### METHODOLOGY.md Claims vs Reality

**Claim** (Line 90):
```
## 3. The 28 Financial Indicators
```

**Reality**: System uses 20 indicators, not 28

**Claim** (Lines 42-69):
```
The system uses **5 individual ML models** combined via weighted voting:
#### 2.1 Support Vector Machine (SVM)
#### 2.2 Random Forest
#### 2.3 Gradient Boosting
#### 2.4 Neural Network (MLP)
#### 2.5 Ensemble Voting
```

**Reality**: System uses 2 models (Gradient Boosting + Random Forest), not 5

**Claim** (Line 21):
```
| 3 | Shiller PE Ratio | > 35 | 15% | Valuation extremes; >35 indicates overvaluation |
```

**Reality**: Shiller PE is synthetic (100 / (VIX + 1)), not real data

---

## EVIDENCE 5: Bottom Predictor Training Data

### Source Code
**File**: `scripts/training/train_bottom_predictor.py` (lines 130-175)

```python
def train_bottom_predictor():
    """Train model to predict days to bottom after crash."""
    
    # Load data
    logger.info("\nLoading historical crash events...")
    crashes_df = load_crash_events()
    logger.info(f"✅ Loaded {len(crashes_df)} historical crashes")
    
    # Create training data
    logger.info("\nCreating training features...")
    X_data = []
    y_days_to_bottom = []
    
    for idx, crash in crashes_df.iterrows():
        # ... create features ...
        X_data.append(features)
        y_days_to_bottom.append(days_to_bottom)
    
    X = pd.DataFrame(X_data)
    y_bottom = np.array(y_days_to_bottom)
    
    logger.info(f"\n✅ Created {len(X)} training samples with {len(X.columns)} features")
```

### Database Evidence
**Query Result**: 11 crash events in database (1980-2023)

### The Problem
- Only 11 training samples for a regression model
- 8 features × 11 samples = severely underfitted
- Model is essentially memorizing 11 historical patterns
- High risk of poor generalization to future crashes

### Reported Metrics
- MAE: ~X days (exact value not shown in audit)
- R²: ~X (exact value not shown in audit)
- These metrics are likely unreliable with only 11 samples

---

## EVIDENCE 6: Data Transformations

### Source Code
**File**: `scripts/data/collect_data.py` (lines 138-145)

```python
# Fill NaN values using forward fill then backward fill
combined_df = combined_df.ffill().bfill()

# For any remaining NaN values, fill with column mean
for col in combined_df.columns:
    if combined_df[col].isna().any():
        combined_df[col] = combined_df[col].fillna(combined_df[col].mean())
```

### Issues
1. **Forward fill**: Propagates old values forward (can introduce bias)
2. **Backward fill**: Propagates future values backward (temporal leakage!)
3. **Mean imputation**: Reduces variance, can distort relationships

### No Documentation
- METHODOLOGY.md doesn't explain these transformations
- No justification for why these methods are appropriate
- No discussion of alternatives

---

## EVIDENCE 7: Feature Engineering

### Source Code
**File**: `scripts/training/train_crash_detector_v5.py` (lines 37-99)

Creates 39 features from 20 raw indicators:
- Yield curve features (6)
- Credit stress features (5)
- Volatility features (5)
- Economic features (4)
- Market momentum features (5)
- Money & debt features (4)
- Sentiment features (3)
- Composite features (3)

### Transformations Applied
- Rolling averages (5-day, 20-day, 60-day)
- Z-scores (standardized deviations)
- Percentage changes
- Binary indicators (inversion, spike, elevation)
- Differences (5-day changes)

### Issue
- No documentation of which features are used for which indicators
- No justification for specific window sizes (5, 20, 60 days)
- No discussion of feature importance or selection

---

## EVIDENCE 8: Historical Crash Verification

### Database Records
```
1. 1980 Recession: 1980-11-28 to 1983-01-15 (-27.1%)
2. Black Monday: 1987-08-25 to 1989-07-15 (-33.5%)
3. 1990 Recession: 1990-07-16 to 1991-06-30 (-19.9%)
4. Russian Crisis: 1998-07-17 to 1999-03-15 (-19.3%)
5. Dot-Com Bubble: 2000-03-24 to 2007-10-09 (-49.1%)
6. Financial Crisis: 2007-10-09 to 2013-03-28 (-56.8%)
7. Debt Crisis: 2011-05-02 to 2012-03-15 (-19.4%)
8. Commodity Crash: 2015-06-23 to 2016-08-15 (-20.5%)
9. Volatility Spike: 2018-09-21 to 2019-09-15 (-19.8%)
10. COVID Pandemic: 2020-02-19 to 2020-08-18 (-33.9%)
11. Fed Rate Hike: 2022-01-03 to 2023-01-24 (-27.5%)
```

### Verification Status
- ✅ Dates are historically accurate
- ✅ Drawdowns are reasonable approximations
- ✅ Recovery periods are plausible
- ⚠️ Some dates are approximate (e.g., Dot-Com peak vs trough)

---

## CONCLUSION

The audit found **3 HIGH severity issues** that fundamentally undermine system credibility:

1. **Synthetic indicators not disclosed** - Users think they're using real CBOE/FINRA data
2. **Temporal leakage in cross-validation** - Performance metrics are inflated
3. **Severely outdated documentation** - References non-existent models and indicators

These issues must be addressed before the system can be trusted for real trading decisions.

