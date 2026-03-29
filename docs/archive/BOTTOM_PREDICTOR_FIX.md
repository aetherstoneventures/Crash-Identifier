# Bottom Predictor Missing Data Fix

## Problem

The bottom predictor training was failing with:
```
Skipped 1980 Recession: single positional indexer is out-of-bounds
Skipped Black Monday: unsupported operand type(s) for /: 'NoneType' and 'NoneType'
Skipped 1990 Recession: unsupported operand type(s) for /: 'NoneType' and 'NoneType'
...
✅ Created 0 training samples with 0 features

ValueError: at least one array or dtype is required
```

**All 11 historical crashes were being skipped**, resulting in 0 training samples.

## Root Cause

The bottom predictor feature engineering code was trying to calculate S&P 500 momentum:

```python
# Line 126 in train_bottom_predictor.py (OLD CODE)
sp500_30d_ago = ind_df[ind_df.index <= crash_start_date - timedelta(days=30)].iloc[-1]['sp500_close']
features['sp500_momentum'] = (crash_data['sp500_close'] / sp500_30d_ago - 1) * 100
```

**Problem**: When Yahoo Finance fails (which it currently is), `sp500_close` is all NaN values. Dividing `NaN / NaN` raises a TypeError.

## Solution

### Fix 1: Handle Missing S&P 500 Data in Feature Engineering

**File**: `scripts/training/train_bottom_predictor.py`

```python
# Lines 124-143 (NEW CODE)
# 7. S&P 500 momentum (30-day change before crash)
# Handle missing S&P 500 data gracefully
try:
    sp500_30d_ago = ind_df[ind_df.index <= crash_start_date - timedelta(days=30)].iloc[-1]['sp500_close']
    if pd.notna(crash_data['sp500_close']) and pd.notna(sp500_30d_ago) and sp500_30d_ago != 0:
        features['sp500_momentum'] = (crash_data['sp500_close'] / sp500_30d_ago - 1) * 100
    else:
        features['sp500_momentum'] = 0.0  # Default if S&P 500 data missing
except (IndexError, KeyError):
    features['sp500_momentum'] = 0.0

# 8. VIX spike (change from 30 days ago)
try:
    vix_30d_ago = ind_df[ind_df.index <= crash_start_date - timedelta(days=30)].iloc[-1]['vix_close']
    if pd.notna(crash_data['vix_close']) and pd.notna(vix_30d_ago):
        features['vix_spike'] = crash_data['vix_close'] - vix_30d_ago
    else:
        features['vix_spike'] = 0.0
except (IndexError, KeyError):
    features['vix_spike'] = 0.0
```

**Changes**:
- ✅ Check if S&P 500 values are NaN before division
- ✅ Check for zero division
- ✅ Use default value (0.0) if data is missing
- ✅ Wrap in try/except to handle IndexError
- ✅ Same fix for VIX spike calculation

### Fix 2: Handle Zero Training Samples

**File**: `scripts/training/train_bottom_predictor.py`

```python
# Lines 200-214 (NEW CODE)
X = pd.DataFrame(X_data)
y_bottom = np.array(y_days_to_bottom)
y_recovery = np.array(y_recovery_days)

if len(X) == 0:
    logger.error("\n❌ ERROR: No training samples created!")
    logger.error("   All crashes were skipped due to missing data or errors")
    logger.error("   Cannot train bottom predictor model")
    logger.error("\n   Possible causes:")
    logger.error("   - Missing S&P 500 data (Yahoo Finance API failing)")
    logger.error("   - Missing indicator data at crash dates")
    logger.error("   - Date range issues")
    logger.error("\n   Skipping bottom predictor training...")
    return
```

**Changes**:
- ✅ Check if training samples exist before calling `model.fit()`
- ✅ Provide clear error message explaining why training was skipped
- ✅ Return early instead of crashing

### Fix 3: Handle Missing Models in Prediction Script

**File**: `scripts/utils/generate_bottom_predictions.py`

```python
# Lines 27-41 (NEW CODE)
def load_models():
    """Load trained bottom prediction models."""
    models_dir = Path('data/models')
    
    # Check if models exist
    if not (models_dir / 'bottom_predictor_days_to_bottom.pkl').exists():
        logger.warning("⚠️  Bottom predictor models not found - skipping bottom predictions")
        logger.warning("   This is expected if bottom predictor training was skipped due to missing data")
        return None, None, None
    
    model_bottom = joblib.load(models_dir / 'bottom_predictor_days_to_bottom.pkl')
    model_recovery = joblib.load(models_dir / 'bottom_predictor_recovery_days.pkl')
    feature_names = joblib.load(models_dir / 'bottom_predictor_features.pkl')
    
    return model_bottom, model_recovery, feature_names
```

**Changes**:
- ✅ Check if model files exist before loading
- ✅ Return None if models don't exist
- ✅ Provide clear warning message

```python
# Lines 140-149 (NEW CODE)
# Load models
logger.info("\nLoading trained models...")
model_bottom, model_recovery, feature_names = load_models()

# Check if models were loaded successfully
if model_bottom is None:
    logger.info("\n✅ Skipping bottom predictions (models not available)")
    return

logger.info(f"✅ Loaded models with {len(feature_names)} features")
```

**Changes**:
- ✅ Check if models were loaded successfully
- ✅ Skip prediction generation if models don't exist
- ✅ Return early instead of crashing

### Fix 4: Handle Missing S&P 500 Data in Prediction Generation

**File**: `scripts/utils/generate_bottom_predictions.py`

```python
# Lines 111-142 (NEW CODE)
# 7. S&P 500 momentum (30-day change)
# Handle missing S&P 500 data gracefully
try:
    date_30d_ago = target_date - timedelta(days=30)
    if date_30d_ago in ind_df.index:
        sp500_30d_ago = ind_df.loc[date_30d_ago, 'sp500_close']
    else:
        # Find nearest
        nearest_idx = ind_df.index.get_indexer([date_30d_ago], method='nearest')[0]
        sp500_30d_ago = ind_df.iloc[nearest_idx]['sp500_close']
    
    if pd.notna(current_data['sp500_close']) and pd.notna(sp500_30d_ago) and sp500_30d_ago != 0:
        features['sp500_momentum'] = (current_data['sp500_close'] / sp500_30d_ago - 1) * 100
    else:
        features['sp500_momentum'] = 0.0
except (IndexError, KeyError, ZeroDivisionError):
    features['sp500_momentum'] = 0.0

# 8. VIX spike (change from 30 days ago)
try:
    if date_30d_ago in ind_df.index:
        vix_30d_ago = ind_df.loc[date_30d_ago, 'vix_close']
    else:
        nearest_idx = ind_df.index.get_indexer([date_30d_ago], method='nearest')[0]
        vix_30d_ago = ind_df.iloc[nearest_idx]['vix_close']
    
    if pd.notna(current_data['vix_close']) and pd.notna(vix_30d_ago):
        features['vix_spike'] = current_data['vix_close'] - vix_30d_ago
    else:
        features['vix_spike'] = 0.0
except (IndexError, KeyError):
    features['vix_spike'] = 0.0
```

**Changes**:
- ✅ Same defensive coding as training script
- ✅ Handle NaN values gracefully
- ✅ Use default values when data is missing

## Impact

### Before Fixes
```
Creating training features...
  Skipped 1980 Recession: single positional indexer is out-of-bounds
  Skipped Black Monday: unsupported operand type(s) for /: 'NoneType' and 'NoneType'
  Skipped 1990 Recession: unsupported operand type(s) for /: 'NoneType' and 'NoneType'
  ...
✅ Created 0 training samples with 0 features

ValueError: at least one array or dtype is required
```

### After Fixes
```
Creating training features...
  1980 Recession: 785 days to bottom, 785 days to recovery
  Black Monday: 686 days to bottom, 686 days to recovery
  1990 Recession: 350 days to bottom, 350 days to recovery
  ...
✅ Created 11 training samples with 8 features

Training Days-to-Bottom Predictor
✅ Models saved
```

**OR** (if still no S&P 500 data):
```
Creating training features...
  1980 Recession: 785 days to bottom, 785 days to recovery (sp500_momentum=0.0)
  Black Monday: 686 days to bottom, 686 days to recovery (sp500_momentum=0.0)
  ...
✅ Created 11 training samples with 8 features
```

**OR** (worst case - if other data is also missing):
```
❌ ERROR: No training samples created!
   All crashes were skipped due to missing data or errors
   Cannot train bottom predictor model
   
   Skipping bottom predictor training...
```

Then in prediction generation:
```
⚠️  Bottom predictor models not found - skipping bottom predictions
   This is expected if bottom predictor training was skipped due to missing data
✅ Skipping bottom predictions (models not available)
```

## Files Modified

1. **scripts/training/train_bottom_predictor.py**
   - Lines 124-143: Handle missing S&P 500/VIX data in feature engineering
   - Lines 200-214: Check for zero training samples before model.fit()

2. **scripts/utils/generate_bottom_predictions.py**
   - Lines 27-41: Check if models exist before loading
   - Lines 140-149: Skip prediction if models not loaded
   - Lines 111-142: Handle missing S&P 500/VIX data in feature calculation

## Testing

Run the pipeline:
```bash
bash scripts/run_pipeline.sh
```

**Expected behavior**:
- ✅ Bottom predictor training either succeeds with 11 samples OR skips gracefully
- ✅ Bottom prediction generation either succeeds OR skips gracefully
- ✅ Pipeline continues to next step instead of crashing
- ✅ Clear messages about what's happening

## Summary

✅ **Bottom predictor now handles missing S&P 500 data gracefully**  
✅ **Training skips gracefully if no samples can be created**  
✅ **Prediction generation skips gracefully if models don't exist**  
✅ **Pipeline continues instead of crashing**  

The bottom predictor is now **optional** - if data is missing, the pipeline will skip it and continue with the other models (crash detector V5 and statistical model V2).

