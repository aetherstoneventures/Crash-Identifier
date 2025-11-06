# AUDIT - CODE LOCATIONS REFERENCE

Quick reference for all issues found in the audit.

---

## HIGH SEVERITY ISSUES

### Issue 1: Synthetic Put/Call Ratio

**File**: `scripts/data/collect_data.py`  
**Lines**: 121-126

```python
# Synthetic put/call ratio: Use VIX change as proxy
if 'vix_close' in combined_df.columns:
    vix_change = combined_df['vix_close'].pct_change()
    combined_df['put_call_ratio'] = 1.0 + (vix_change * 0.5).clip(-0.5, 0.5)
    logger.info("✅ Calculated put_call_ratio from vix_close")
```

**Also referenced in**:
- `src/data_collection/alternative_collector.py` lines 138-142 (original implementation)
- `scripts/training/train_crash_detector_v5.py` line 87 (used in features)
- `src/dashboard/app.py` (displayed to users)

**Database**: `data/market_crash.db` → `indicators` table → `put_call_ratio` column

---

### Issue 2: Synthetic Margin Debt

**File**: `scripts/data/collect_data.py`  
**Lines**: 114-116

```python
# Synthetic margin debt: Use credit spread as inverse proxy
if 'credit_spread_bbb' in combined_df.columns:
    combined_df['margin_debt'] = 100 / (combined_df['credit_spread_bbb'] + 1)
    logger.info("✅ Calculated margin_debt from credit_spread_bbb")
```

**Also referenced in**:
- `src/data_collection/alternative_collector.py` lines 130-134 (original implementation)
- `scripts/training/train_crash_detector_v5.py` line 83 (used in features)
- `src/dashboard/app.py` (displayed to users)

**Database**: `data/market_crash.db` → `indicators` table → `margin_debt` column

---

### Issue 3: Temporal Leakage in Cross-Validation

**File**: `scripts/training/train_crash_detector_v5.py`  
**Lines**: 157-180

```python
# K-Fold Cross-Validation
logger.info("\n" + "=" * 80)
logger.info("K-FOLD CROSS-VALIDATION (5 folds)")
logger.info("=" * 80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # ← PROBLEM: shuffle=True

# Gradient Boosting with regularization
logger.info("\nTraining Gradient Boosting with regularization...")
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

gb_cv_scores = cross_validate(
    gb, X_train_scaled, y_train, cv=skf,
    scoring=['roc_auc', 'recall', 'precision', 'f1'],
    return_train_score=True
)
```

**Impact on metrics** (lines 182-189):
```python
logger.info(f"GB Train AUC: {gb_cv_scores['train_roc_auc'].mean():.4f}")
logger.info(f"GB Val AUC:   {gb_cv_scores['test_roc_auc'].mean():.4f}")
logger.info(f"GB Recall:    {gb_cv_scores['test_recall'].mean():.4f}")
# These metrics are INFLATED due to temporal leakage
```

---

## MEDIUM SEVERITY ISSUES

### Issue 4: Outdated Documentation - 28 Indicators

**File**: `docs/METHODOLOGY.md`  
**Line**: 90

```markdown
## 3. The 28 Financial Indicators
```

**Reality**: System uses 20 indicators, not 28

**Should be**: "## 3. The 20 Financial Indicators"

---

### Issue 5: Outdated Documentation - 5 ML Models

**File**: `docs/METHODOLOGY.md`  
**Lines**: 42-69

```markdown
### Ensemble Architecture
The system uses **5 individual ML models** combined via weighted voting:

#### 2.1 Support Vector Machine (SVM)
#### 2.2 Random Forest
#### 2.3 Gradient Boosting
#### 2.4 Neural Network (MLP)
#### 2.5 Ensemble Voting
```

**Reality**: System uses 2 models (Gradient Boosting + Random Forest)

**Actual implementation**: `scripts/training/train_crash_detector_v5.py` lines 164-224

---

### Issue 6: Synthetic Shiller PE Presented as Real

**File**: `docs/METHODOLOGY.md`  
**Line**: 21

```markdown
| 3 | Shiller PE Ratio | > 35 | 15% | Valuation extremes; >35 indicates overvaluation |
```

**Reality**: Shiller PE is synthetic (100 / (VIX + 1))

**Actual implementation**: `src/data_collection/alternative_collector.py` lines 125-128

---

### Issue 7: Bottom Predictor - Only 11 Training Samples

**File**: `scripts/training/train_bottom_predictor.py`  
**Lines**: 130-175

```python
def train_bottom_predictor():
    """Train model to predict days to bottom after crash."""
    
    # Load data
    logger.info("\nLoading historical crash events...")
    crashes_df = load_crash_events()
    logger.info(f"✅ Loaded {len(crashes_df)} historical crashes")  # ← Only 11
    
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
    # ↑ Only 11 samples for regression model
```

**Database verification**: `data/market_crash.db` → `crash_events` table → 11 rows

---

## LOW SEVERITY ISSUES

### Issue 8: Data Transformations Not Documented

**File**: `scripts/data/collect_data.py`  
**Lines**: 138-145

```python
# Fill NaN values using forward fill then backward fill
combined_df = combined_df.ffill().bfill()

# For any remaining NaN values, fill with column mean
for col in combined_df.columns:
    if combined_df[col].isna().any():
        combined_df[col] = combined_df[col].fillna(combined_df[col].mean())
```

**Problem**: No documentation of why these methods are used

---

### Issue 9: Feature Engineering Not Documented

**File**: `scripts/training/train_crash_detector_v5.py`  
**Lines**: 37-99

Creates 39 features from 20 indicators with:
- Rolling averages (5, 20, 60 day windows)
- Z-scores
- Percentage changes
- Binary indicators

**Problem**: No documentation of feature selection or importance

---

### Issue 10: README Claims Don't Match Implementation

**File**: `README.md`  
**Lines**: 10-15

```markdown
A machine learning-based market crash detection system that predicts stock market crashes 
with 81.8% accuracy using 20 economic and market indicators. The system combines:

- **ML Models**: Gradient Boosting (70%) + Random Forest (30%) with K-Fold cross-validation
- **Features**: 39 engineered features from 20 raw indicators
- **Data**: 11,431 daily records (1982-2025) from FRED and Yahoo Finance
- **Validation**: Anti-overfitting measures (overfitting gap < 0.002)
```

**Problem**: K-Fold cross-validation is invalid for time-series (temporal leakage)

---

### Issue 11: Crash Events Definition Inconsistency

**File**: `scripts/data/populate_crash_events.py`  
**Lines**: 11-23

```python
HISTORICAL_CRASHES = [
    ('1980-11-28', '1982-08-12', '1982-08-12', '1983-01-15', -27.1, 5, '1980 Recession', ...),
    ('1987-08-25', '1987-12-04', '1987-10-19', '1989-07-15', -33.5, 21, 'Black Monday', ...),
    # ...
]
```

**Schema**: (start_date, end_date, trough_date, recovery_date, max_drawdown, recovery_months, crash_type, notes)

**Problem**: Some dates are approximate (e.g., Dot-Com peak vs trough)

---

## VERIFICATION COMMANDS

To verify these issues yourself:

```bash
# Check synthetic indicators in database
sqlite3 data/market_crash.db "SELECT date, put_call_ratio, margin_debt FROM indicators LIMIT 5;"

# Check crash events
sqlite3 data/market_crash.db "SELECT crash_type, start_date, max_drawdown FROM crash_events;"

# Check cross-validation implementation
grep -n "StratifiedKFold" scripts/training/train_crash_detector_v5.py

# Check synthetic indicator creation
grep -n "put_call_ratio\|margin_debt" scripts/data/collect_data.py
```

---

## SUMMARY TABLE

| Issue | File | Lines | Severity | Type |
|-------|------|-------|----------|------|
| Synthetic put/call ratio | `scripts/data/collect_data.py` | 121-126 | HIGH | Data |
| Synthetic margin debt | `scripts/data/collect_data.py` | 114-116 | HIGH | Data |
| Temporal leakage | `scripts/training/train_crash_detector_v5.py` | 162 | HIGH | ML |
| 28 indicators claim | `docs/METHODOLOGY.md` | 90 | MEDIUM | Docs |
| 5 models claim | `docs/METHODOLOGY.md` | 42-69 | MEDIUM | Docs |
| Synthetic Shiller PE | `docs/METHODOLOGY.md` | 21 | MEDIUM | Docs |
| 11 samples | `scripts/training/train_bottom_predictor.py` | 130-175 | MEDIUM | ML |
| Transformations | `scripts/data/collect_data.py` | 138-145 | LOW | Docs |
| Features | `scripts/training/train_crash_detector_v5.py` | 37-99 | LOW | Docs |
| README claims | `README.md` | 10-15 | LOW | Docs |
| Crash dates | `scripts/data/populate_crash_events.py` | 11-23 | LOW | Data |

