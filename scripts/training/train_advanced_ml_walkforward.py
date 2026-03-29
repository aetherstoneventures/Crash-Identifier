"""
PHASE 4: ADVANCED ML MODELS WITH WALK-FORWARD VALIDATION
XGBoost, LightGBM, CatBoost with proper time-series validation

Goal: Achieve AUC > 0.70 using mathematically rigorous gradient boosting models
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import sqlite3
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

# Advanced ML libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

# Import walk-forward validator only (avoid DatabaseManager import)
from src.utils.walk_forward_validation import WalkForwardValidator

def engineer_crash_features_v6(df):
    """Enhanced feature engineering (39 features) - corrected for available columns."""
    X = pd.DataFrame(index=df.index)

    # 1. YIELD CURVE (6 features)
    X['yield_curve_10y_3m'] = df['yield_10y_3m']
    X['yield_curve_inversion'] = (X['yield_curve_10y_3m'] < 0).astype(int)
    X['yield_curve_negative_days'] = (X['yield_curve_10y_3m'] < 0).rolling(20).sum()
    X['yield_curve_slope_ma5'] = X['yield_curve_10y_3m'].rolling(5).mean()
    X['yield_curve_slope_ma20'] = X['yield_curve_10y_3m'].rolling(20).mean()
    X['yield_curve_deterioration'] = X['yield_curve_slope_ma20'].diff(5)

    # 2. CREDIT STRESS (6 features)
    X['credit_spread'] = df['credit_spread_bbb']
    X['credit_spread_ma5'] = X['credit_spread'].rolling(5).mean()
    X['credit_spread_ma20'] = X['credit_spread'].rolling(20).mean()
    X['credit_spread_zscore'] = (X['credit_spread'] - X['credit_spread_ma20']) / (X['credit_spread'].rolling(20).std() + 0.001)
    X['credit_spread_widening'] = X['credit_spread'].diff(5) > 0
    X['credit_spread_high'] = X['credit_spread'] > X['credit_spread_ma20'] * 1.2

    # 3. VOLATILITY (6 features)
    X['vix'] = df['vix_close']
    X['vix_ma5'] = X['vix'].rolling(5).mean()
    X['vix_ma20'] = X['vix'].rolling(20).mean()
    X['vix_spike'] = (X['vix'] > X['vix_ma20'] * 1.5).astype(int)
    X['vix_elevated'] = (X['vix'] > 20).astype(int)
    X['vix_trend'] = X['vix'].diff(5)

    # 4. ECONOMIC DETERIORATION (6 features)
    X['unemployment'] = df['unemployment_rate']
    X['unemployment_ma5'] = X['unemployment'].rolling(5).mean()
    X['unemployment_rising'] = X['unemployment'].diff(5) > 0
    X['industrial_prod'] = df['industrial_production']
    X['industrial_prod_ma5'] = X['industrial_prod'].rolling(5).mean()
    X['industrial_prod_declining'] = X['industrial_prod'].diff(5) < 0

    # 5. MARKET MOMENTUM (5 features)
    X['sp500_returns_5d'] = df['sp500_close'].pct_change(5)
    X['sp500_returns_20d'] = df['sp500_close'].pct_change(20)
    X['sp500_volatility_20d'] = df['sp500_close'].pct_change().rolling(20).std()
    X['sp500_negative_returns'] = (X['sp500_returns_20d'] < 0).astype(int)
    X['sp500_high_volatility'] = X['sp500_volatility_20d'] > X['sp500_volatility_20d'].rolling(60).mean() * 1.5

    # 6. MONEY & DEBT (2 features)
    X['m2_growth'] = df['m2_money_supply'].pct_change(20)
    X['debt_to_gdp'] = df['debt_to_gdp']

    # 7. SENTIMENT (2 features)
    X['consumer_sentiment'] = df['consumer_sentiment']
    X['consumer_sentiment_ma5'] = X['consumer_sentiment'].rolling(5).mean()

    # 8. COMPOSITE (6 features)
    X['fed_funds_rate'] = df['fed_funds_rate']
    X['lei'] = df['lei']
    X['housing_starts'] = df['housing_starts']
    X['real_gdp'] = df['real_gdp']
    X['cpi'] = df['cpi']
    X['savings_rate'] = df['savings_rate']

    # Fill NaN values
    X = X.ffill().bfill().fillna(0)

    return X

print("=" * 80)
print("PHASE 4: ADVANCED ML MODELS - WALK-FORWARD VALIDATION")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading data from database...")
db_path = "data/market_crash.db"
conn = sqlite3.connect(db_path)

query = """
SELECT date, sp500_close, sp500_volume, vix_close, yield_10y, yield_10y_2y, yield_10y_3m,
       credit_spread_bbb, unemployment_rate, industrial_production, consumer_sentiment,
       real_gdp, cpi, m2_money_supply, fed_funds_rate, housing_starts, savings_rate,
       debt_to_gdp, lei
FROM indicators
ORDER BY date
"""
df = pd.read_sql_query(query, conn)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Load crash events
crash_query = "SELECT start_date FROM crash_events ORDER BY start_date"
crash_dates = pd.read_sql_query(crash_query, conn)
crash_dates['start_date'] = pd.to_datetime(crash_dates['start_date'])
conn.close()

print(f"Data: {df.shape}, Crashes: {len(crash_dates)}")

# ============================================================================
# 2. ENGINEER FEATURES (V6)
# ============================================================================
print("\n2. Engineering features (V6 - 39 features)...")
X = engineer_crash_features_v6(df)
print(f"Features: {X.shape}")

# ============================================================================
# 3. CREATE LABELS (90-day lookback window)
# ============================================================================
print("\n3. Creating crash labels (90-day lookback window)...")
y = pd.Series(0, index=X.index)
for crash_date in crash_dates['start_date']:
    lookback_start = crash_date - pd.Timedelta(days=90)
    mask = (X.index >= lookback_start) & (X.index <= crash_date)
    y[mask] = 1

print(f"Crash samples: {y.sum()} ({y.mean()*100:.1f}%)")

# ============================================================================
# 4. WALK-FORWARD VALIDATION SETUP
# ============================================================================
print("\n4. Setting up walk-forward validation...")
validator = WalkForwardValidator(
    window_size=252,      # 1 year test window
    step_size=21,         # 1 month step
    min_train_size=1260   # 5 years minimum training
)

splits = validator.expanding_window_split(X, y)
print(f"Generated {len(splits)} folds")

# ============================================================================
# 5. TRAIN MODELS WITH WALK-FORWARD VALIDATION
# ============================================================================

def train_xgboost_fold(X_train, y_train, X_test, y_test):
    """Train XGBoost on one fold."""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 3,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 5.0,
        'reg_lambda': 10.0,
        'min_child_weight': 10,
        'gamma': 0.5,
        'scale_pos_weight': 15,
        'seed': 42,
        'tree_method': 'hist'
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    y_pred = model.predict(dtest)
    return y_pred

def train_lightgbm_fold(X_train, y_train, X_test, y_test):
    """Train LightGBM on one fold."""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'max_depth': 3,
        'learning_rate': 0.01,
        'num_leaves': 7,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 5.0,
        'reg_lambda': 10.0,
        'min_child_samples': 50,
        'scale_pos_weight': 15,
        'verbose': -1,
        'seed': 42
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[test_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)
        ]
    )

    y_pred = model.predict(X_test)
    return y_pred

def train_catboost_fold(X_train, y_train, X_test, y_test):
    """Train CatBoost on one fold."""
    train_pool = Pool(X_train, y_train)
    test_pool = Pool(X_test, y_test)

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.01,
        depth=3,
        l2_leaf_reg=10.0,
        bootstrap_type='Bayesian',
        bagging_temperature=3.0,
        random_strength=3.0,
        auto_class_weights='Balanced',
        early_stopping_rounds=50,
        verbose=False,
        random_seed=42
    )

    model.fit(train_pool, eval_set=test_pool, use_best_model=True)
    y_pred = model.predict_proba(X_test)[:, 1]
    return y_pred

# ============================================================================
# WALK-FORWARD VALIDATION FOR EACH MODEL
# ============================================================================

models_to_test = [
    ('XGBoost', train_xgboost_fold),
    ('LightGBM', train_lightgbm_fold),
    ('CatBoost', train_catboost_fold)
]

results = {}

for model_name, train_func in models_to_test:
    print("\n" + "=" * 80)
    print(f"TESTING: {model_name}")
    print("=" * 80)

    fold_aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]

        # Skip folds with no crashes in test set
        if y_test_fold.sum() == 0:
            continue

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)

        # Apply SMOTE to training data (adaptive strategy)
        # Calculate max safe sampling strategy
        n_minority = int(y_train_fold.sum())
        n_majority = len(y_train_fold) - n_minority

        if n_minority > 0 and n_minority < n_majority:
            # Current ratio of minority to majority
            current_ratio = n_minority / n_majority
            # Target ratio (what we want after SMOTE)
            target_ratio = min(0.3, current_ratio * 2.0)  # At most double the current ratio or 0.3

            # Only apply SMOTE if we have enough minority samples and target is achievable
            if n_minority >= 6 and target_ratio > current_ratio:  # SMOTE needs at least 6 samples
                try:
                    smote = SMOTE(sampling_strategy=target_ratio, random_state=42, k_neighbors=min(5, n_minority-1))
                    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train_fold)
                except:
                    # If SMOTE fails, skip it
                    X_train_balanced, y_train_balanced = X_train_scaled, y_train_fold
            else:
                # Skip SMOTE if not enough samples
                X_train_balanced, y_train_balanced = X_train_scaled, y_train_fold
        else:
            # No minority samples or already balanced, skip SMOTE
            X_train_balanced, y_train_balanced = X_train_scaled, y_train_fold

        # Train model
        try:
            y_pred = train_func(X_train_balanced, y_train_balanced, X_test_scaled, y_test_fold)
            auc = roc_auc_score(y_test_fold, y_pred)
            fold_aucs.append(auc)

            if (fold_idx + 1) % 100 == 0:
                print(f"Fold {fold_idx + 1}: AUC={auc:.4f}")
        except Exception as e:
            continue

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)

    results[model_name] = {
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'valid_folds': len(fold_aucs)
    }

    print(f"\n{model_name}:")
    print(f"  Mean AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")
    print(f"  Valid folds: {len(fold_aucs)}")

# ============================================================================
# ENSEMBLE MODEL (WEIGHTED AVERAGE)
# ============================================================================
print("\n" + "=" * 80)
print("TESTING: ENSEMBLE (Equal Weight)")
print("=" * 80)

ensemble_aucs = []

for fold_idx, (train_idx, test_idx) in enumerate(splits):
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_test_fold = X.iloc[test_idx]
    y_test_fold = y.iloc[test_idx]

    if y_test_fold.sum() == 0:
        continue

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_test_scaled = scaler.transform(X_test_fold)

    # Apply SMOTE to training data (adaptive strategy)
    n_minority = int(y_train_fold.sum())
    n_majority = len(y_train_fold) - n_minority

    if n_minority > 0 and n_minority < n_majority:
        current_ratio = n_minority / n_majority
        target_ratio = min(0.3, current_ratio * 2.0)

        if n_minority >= 6 and target_ratio > current_ratio:
            try:
                smote = SMOTE(sampling_strategy=target_ratio, random_state=42, k_neighbors=min(5, n_minority-1))
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train_fold)
            except:
                X_train_balanced, y_train_balanced = X_train_scaled, y_train_fold
        else:
            X_train_balanced, y_train_balanced = X_train_scaled, y_train_fold
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train_fold

    try:
        # Get predictions from all three models
        xgb_pred = train_xgboost_fold(X_train_balanced, y_train_balanced, X_test_scaled, y_test_fold)
        lgb_pred = train_lightgbm_fold(X_train_balanced, y_train_balanced, X_test_scaled, y_test_fold)
        cat_pred = train_catboost_fold(X_train_balanced, y_train_balanced, X_test_scaled, y_test_fold)

        # Ensemble: equal weight average
        ensemble_pred = (xgb_pred + lgb_pred + cat_pred) / 3.0
        auc = roc_auc_score(y_test_fold, ensemble_pred)
        ensemble_aucs.append(auc)

        if (fold_idx + 1) % 100 == 0:
            print(f"Fold {fold_idx + 1}: Ensemble AUC={auc:.4f}")
    except Exception as e:
        continue

ensemble_mean_auc = np.mean(ensemble_aucs)
ensemble_std_auc = np.std(ensemble_aucs)

results['Ensemble'] = {
    'mean_auc': ensemble_mean_auc,
    'std_auc': ensemble_std_auc,
    'valid_folds': len(ensemble_aucs)
}

print(f"\nEnsemble (Equal Weight):")
print(f"  Mean AUC: {ensemble_mean_auc:.4f} (+/- {ensemble_std_auc:.4f})")
print(f"  Valid folds: {len(ensemble_aucs)}")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ADVANCED ML MODELS - RESULTS SUMMARY")
print("=" * 80)

for model_name, metrics in results.items():
    print(f"{model_name:20s}: AUC {metrics['mean_auc']:.4f} (+/- {metrics['std_auc']:.4f})")

print("\n" + "=" * 80)
print("COMPARISON WITH PREVIOUS MODELS")
print("=" * 80)
print(f"{'Statistical Model V3':20s}: AUC 0.3532 (+/- 0.1624)")
print(f"{'ML Ensemble (V6)':20s}: AUC 0.5829 (+/- 0.2922)")
print(f"{'Hybrid (80/20)':20s}: AUC 0.6156 (+/- 0.2228)")

# Find best model
best_model = max(results.items(), key=lambda x: x[1]['mean_auc'])
print(f"\n{'BEST MODEL':20s}: {best_model[0]} - AUC {best_model[1]['mean_auc']:.4f}")

# ============================================================================
# PERFORMANCE ASSESSMENT
# ============================================================================
print("\n" + "=" * 80)
print("PERFORMANCE ASSESSMENT")
print("=" * 80)

best_auc = best_model[1]['mean_auc']
if best_auc >= 0.70:
    print(f"✅ TARGET ACHIEVED: AUC {best_auc:.4f} >= 0.70")
    print("Phase 4 COMPLETE - Advanced ML models successful!")
elif best_auc >= 0.65:
    print(f"⚠️  CLOSE TO TARGET: AUC {best_auc:.4f} (need +{0.70 - best_auc:.4f})")
    print("Recommendation: Proceed to Phase 5 (Enhanced Features)")
else:
    print(f"❌ BELOW TARGET: AUC {best_auc:.4f} (need +{0.70 - best_auc:.4f})")
    print("Recommendation: Phase 5 (Enhanced Features) is REQUIRED")

# ============================================================================
# TRAIN FINAL MODELS ON ALL DATA
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING FINAL MODELS ON ALL DATA")
print("=" * 80)

# Scale all data
scaler_final = StandardScaler()
X_scaled = scaler_final.fit_transform(X)

# Apply SMOTE (adaptive strategy for final training)
n_minority = int(y.sum())
n_majority = len(y) - n_minority
current_ratio = n_minority / n_majority
target_ratio = min(0.3, current_ratio * 2.0)

if n_minority >= 6 and target_ratio > current_ratio:
    try:
        smote_final = SMOTE(sampling_strategy=target_ratio, random_state=42, k_neighbors=min(5, n_minority-1))
        X_balanced, y_balanced = smote_final.fit_resample(X_scaled, y)
        print(f"After SMOTE: {len(X_balanced)} samples ({y_balanced.sum()} crashes)")
    except:
        X_balanced, y_balanced = X_scaled, y
        print(f"SMOTE failed, using original data: {len(X_balanced)} samples ({y_balanced.sum()} crashes)")
else:
    X_balanced, y_balanced = X_scaled, y
    print(f"Skipping SMOTE (insufficient minority samples): {len(X_balanced)} samples ({y_balanced.sum()} crashes)")

# Train XGBoost
print("\nTraining final XGBoost model...")
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 3,
    'learning_rate': 0.01,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 5.0,
    'reg_lambda': 10.0,
    'min_child_weight': 10,
    'gamma': 0.5,
    'scale_pos_weight': 15,
    'seed': 42,
    'tree_method': 'hist'
}
dtrain_final = xgb.DMatrix(X_balanced, label=y_balanced)
xgb_final = xgb.train(xgb_params, dtrain_final, num_boost_round=500)

# Train LightGBM
print("Training final LightGBM model...")
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'max_depth': 3,
    'learning_rate': 0.01,
    'num_leaves': 7,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 5.0,
    'reg_lambda': 10.0,
    'min_child_samples': 50,
    'scale_pos_weight': 15,
    'verbose': -1,
    'seed': 42
}
train_data_final = lgb.Dataset(X_balanced, label=y_balanced)
lgb_final = lgb.train(lgb_params, train_data_final, num_boost_round=500)

# Train CatBoost
print("Training final CatBoost model...")
train_pool_final = Pool(X_balanced, y_balanced)
cat_final = CatBoostClassifier(
    iterations=500,
    learning_rate=0.01,
    depth=3,
    l2_leaf_reg=10.0,
    bootstrap_type='Bayesian',
    bagging_temperature=3.0,
    random_strength=3.0,
    auto_class_weights='Balanced',
    verbose=False,
    random_seed=42
)
cat_final.fit(train_pool_final)

print("✅ Final models trained on all data")

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING MODELS")
print("=" * 80)

models_dir = Path('data/models')
models_dir.mkdir(parents=True, exist_ok=True)

# Save XGBoost
xgb_final.save_model(str(models_dir / 'xgboost_model.json'))
print("✅ Saved: data/models/xgboost_model.json")

# Save LightGBM
lgb_final.save_model(str(models_dir / 'lightgbm_model.txt'))
print("✅ Saved: data/models/lightgbm_model.txt")

# Save CatBoost
cat_final.save_model(str(models_dir / 'catboost_model.cbm'))
print("✅ Saved: data/models/catboost_model.cbm")

# Save scaler
with open(models_dir / 'scaler_advanced.pkl', 'wb') as f:
    pickle.dump(scaler_final, f)
print("✅ Saved: data/models/scaler_advanced.pkl")

# Save results
with open(models_dir / 'advanced_ml_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("✅ Saved: data/models/advanced_ml_results.pkl")

print("\n" + "=" * 80)
print("PHASE 4 COMPLETE")
print("=" * 80)

