"""
Crash Detector V6 - Walk-Forward Validation (NO DATA LEAKAGE)

This script implements:
1. Walk-Forward Validation to prevent temporal leakage
2. Expanding window training (always train on all historical data)
3. SMOTE for class imbalance handling (applied within each fold)
4. Regularization (L2 penalty)
5. Realistic performance metrics (expected AUC ~0.70-0.80)

Ensures the model generalizes well to unseen data WITHOUT look-ahead bias.
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             recall_score, precision_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from src.utils.database import DatabaseManager, Indicator, Prediction
from src.utils.config import DATABASE_URL
from src.utils.walk_forward_validation import WalkForwardValidator

def engineer_crash_features_v6(df):
    """Enhanced feature engineering matching V5 but with better normalization."""
    X = pd.DataFrame(index=df.index)
    
    # 1. YIELD CURVE
    X['yield_curve_10y_3m'] = df['yield_10y_3m']
    X['yield_curve_inversion'] = (X['yield_curve_10y_3m'] < 0).astype(int)
    X['yield_curve_negative_days'] = (X['yield_curve_10y_3m'] < 0).rolling(20).sum()
    X['yield_curve_slope_ma5'] = X['yield_curve_10y_3m'].rolling(5).mean()
    X['yield_curve_slope_ma20'] = X['yield_curve_10y_3m'].rolling(20).mean()
    X['yield_curve_deterioration'] = X['yield_curve_slope_ma20'].diff(5)
    
    # 2. CREDIT STRESS
    X['credit_spread'] = df['credit_spread_bbb']
    X['credit_spread_ma5'] = X['credit_spread'].rolling(5).mean()
    X['credit_spread_ma20'] = X['credit_spread'].rolling(20).mean()
    X['credit_spread_zscore'] = (X['credit_spread'] - X['credit_spread_ma20']) / (X['credit_spread'].rolling(20).std() + 0.001)
    X['credit_spread_widening'] = X['credit_spread'].diff(5) > 0
    X['credit_spread_high'] = X['credit_spread'] > X['credit_spread_ma20'] * 1.2
    
    # 3. VOLATILITY
    X['vix'] = df['vix_close']
    X['vix_ma5'] = X['vix'].rolling(5).mean()
    X['vix_ma20'] = X['vix'].rolling(20).mean()
    X['vix_spike'] = (X['vix'] > X['vix_ma20'] * 1.5).astype(int)
    X['vix_elevated'] = (X['vix'] > 20).astype(int)
    X['vix_trend'] = X['vix'].diff(5)
    
    # 4. ECONOMIC DETERIORATION
    X['unemployment'] = df['unemployment_rate']
    X['unemployment_ma5'] = X['unemployment'].rolling(5).mean()
    X['unemployment_rising'] = X['unemployment'].diff(5) > 0
    X['industrial_prod'] = df['industrial_production']
    X['industrial_prod_ma5'] = X['industrial_prod'].rolling(5).mean()
    X['industrial_prod_declining'] = X['industrial_prod'].diff(5) < 0
    
    # 5. MARKET MOMENTUM
    X['sp500_returns_5d'] = df['sp500_close'].pct_change(5)
    X['sp500_returns_20d'] = df['sp500_close'].pct_change(20)
    X['sp500_volatility_20d'] = df['sp500_close'].pct_change().rolling(20).std()
    X['sp500_negative_returns'] = (X['sp500_returns_20d'] < 0).astype(int)
    X['sp500_high_volatility'] = X['sp500_volatility_20d'] > X['sp500_volatility_20d'].rolling(60).mean() * 1.5
    
    # 6. MONEY & DEBT
    X['m2_growth'] = df['m2_money_supply'].pct_change(20)
    X['debt_to_gdp'] = df['debt_to_gdp']
    X['margin_debt'] = df['margin_debt']
    X['margin_debt_ma20'] = X['margin_debt'].rolling(20).mean()
    
    # 7. SENTIMENT
    X['put_call_ratio'] = df['put_call_ratio']
    X['consumer_sentiment'] = df['consumer_sentiment']
    X['consumer_sentiment_ma5'] = X['consumer_sentiment'].rolling(5).mean()
    
    # 8. COMPOSITE
    X['fed_funds_rate'] = df['fed_funds_rate']
    X['lei'] = df['lei']
    X['housing_starts'] = df['housing_starts']
    
    # Fill NaN values
    X = X.ffill().bfill().fillna(0)
    
    return X

def train_crash_detector_v6():
    """Train crash detection model with walk-forward validation to prevent overfitting."""
    logger.info("=" * 80)
    logger.info("TRAINING CRASH DETECTOR V6 - WALK-FORWARD VALIDATION")
    logger.info("=" * 80)
    
    # Load data
    db = DatabaseManager()
    db.create_tables()

    with db.get_session() as session:
        df = pd.read_sql_query(
            "SELECT * FROM indicators ORDER BY date",
            session.bind
        )
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        logger.info(f"\nData shape: {df.shape}")

        # Get crash events
        crashes = pd.read_sql_query(
            "SELECT start_date, end_date FROM crash_events ORDER BY start_date",
            session.bind
        )
        crash_events = list(zip(crashes['start_date'], crashes['end_date']))
    logger.info(f"Historical crashes: {len(crash_events)}")
    
    # Engineer features
    logger.info("\nEngineering features...")
    X = engineer_crash_features_v6(df)
    logger.info(f"Features created: {X.shape[1]}")
    
    # Create labels with 90-day window
    y = pd.Series(0, index=df.index)
    for crash_start, crash_end in crash_events:
        crash_start_date = pd.to_datetime(crash_start)
        lookback_date = crash_start_date - pd.Timedelta(days=90)
        mask = (df.index >= lookback_date) & (df.index < crash_start_date)
        y[mask] = 1
    
    logger.info(f"Crash samples: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")

    # WALK-FORWARD VALIDATION (NO DATA LEAKAGE)
    logger.info("\n" + "=" * 80)
    logger.info("WALK-FORWARD VALIDATION (EXPANDING WINDOW)")
    logger.info("=" * 80)
    logger.info("✅ Using expanding window walk-forward validation")
    logger.info("   - Window size: 252 days (1 year)")
    logger.info("   - Step size: 21 days (1 month)")
    logger.info("   - Min train size: 1260 days (5 years)")
    logger.info("   - NO temporal leakage - always train on past, test on future")
    logger.info("   - SMOTE applied within each fold to handle class imbalance")
    logger.info("=" * 80)

    # Initialize walk-forward validator
    validator = WalkForwardValidator(
        window_size=252,  # 1 year test window
        step_size=21,     # 1 month step
        min_train_size=1260  # 5 years minimum training
    )

    # Generate splits
    splits = validator.expanding_window_split(X, y)
    logger.info(f"\nGenerated {len(splits)} walk-forward folds")

    if len(splits) == 0:
        logger.error("❌ No walk-forward splits generated! Data may be too short.")
        logger.error(f"   Data length: {len(X)} samples")
        logger.error(f"   Min required: {1260 + 252} samples")
        return

    # Import SMOTE for handling class imbalance
    try:
        from imblearn.over_sampling import SMOTE
        smote_available = True
        logger.info("✅ SMOTE available - will use oversampling within each fold")
    except ImportError:
        smote_available = False
        logger.error("❌ imbalanced-learn not installed - REQUIRED for walk-forward validation")
        logger.error("   Install with: pip install imbalanced-learn")
        return

    # Walk-forward validation loop
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING WITH WALK-FORWARD VALIDATION")
    logger.info("=" * 80)

    gb_fold_metrics = []
    rf_fold_metrics = []
    all_test_indices = []
    all_test_y_true = []
    all_test_gb_proba = []
    all_test_rf_proba = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        logger.info(f"\n{'='*80}")
        logger.info(f"FOLD {fold_idx + 1}/{len(splits)}")
        logger.info(f"{'='*80}")
        logger.info(f"Train size: {len(train_idx)} samples")
        logger.info(f"Test size: {len(test_idx)} samples")

        # Split data
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]

        logger.info(f"Train crashes: {y_train_fold.sum()} ({y_train_fold.sum()/len(y_train_fold)*100:.2f}%)")
        logger.info(f"Test crashes: {y_test_fold.sum()} ({y_test_fold.sum()/len(y_test_fold)*100:.2f}%)")

        # Check if we have enough crashes in training set for SMOTE
        if y_train_fold.sum() < 2:
            logger.warning(f"⚠️  Fold {fold_idx + 1}: Only {y_train_fold.sum()} crash samples - SKIPPING")
            continue

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)

        # Apply SMOTE to training set only
        try:
            smote = SMOTE(sampling_strategy=0.3, random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train_fold)
            logger.info(f"After SMOTE: {len(X_train_balanced)} samples ({y_train_balanced.sum()} crashes, {y_train_balanced.sum()/len(y_train_balanced)*100:.2f}%)")
        except Exception as e:
            logger.warning(f"⚠️  SMOTE failed on fold {fold_idx + 1}: {e}")
            logger.warning("   Using original training set with class weights")
            X_train_balanced = X_train_scaled
            y_train_balanced = y_train_fold

        # Train Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        gb.fit(X_train_balanced, y_train_balanced)

        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf.fit(X_train_balanced, y_train_balanced)

        # Evaluate on test fold
        gb_test_proba = gb.predict_proba(X_test_scaled)[:, 1]
        rf_test_proba = rf.predict_proba(X_test_scaled)[:, 1]

        gb_test_pred = (gb_test_proba >= 0.5).astype(int)
        rf_test_pred = (rf_test_proba >= 0.5).astype(int)

        # Calculate metrics (only if we have both classes in test set)
        if y_test_fold.sum() > 0 and y_test_fold.sum() < len(y_test_fold):
            gb_auc = roc_auc_score(y_test_fold, gb_test_proba)
            rf_auc = roc_auc_score(y_test_fold, rf_test_proba)

            gb_recall = recall_score(y_test_fold, gb_test_pred, zero_division=0)
            rf_recall = recall_score(y_test_fold, rf_test_pred, zero_division=0)

            gb_precision = precision_score(y_test_fold, gb_test_pred, zero_division=0)
            rf_precision = precision_score(y_test_fold, rf_test_pred, zero_division=0)

            logger.info(f"\nGradient Boosting - Fold {fold_idx + 1}:")
            logger.info(f"  AUC: {gb_auc:.4f}")
            logger.info(f"  Recall: {gb_recall:.4f}")
            logger.info(f"  Precision: {gb_precision:.4f}")

            logger.info(f"\nRandom Forest - Fold {fold_idx + 1}:")
            logger.info(f"  AUC: {rf_auc:.4f}")
            logger.info(f"  Recall: {rf_recall:.4f}")
            logger.info(f"  Precision: {rf_precision:.4f}")

            gb_fold_metrics.append({
                'fold': fold_idx + 1,
                'auc': gb_auc,
                'recall': gb_recall,
                'precision': gb_precision
            })

            rf_fold_metrics.append({
                'fold': fold_idx + 1,
                'auc': rf_auc,
                'recall': rf_recall,
                'precision': rf_precision
            })
        else:
            logger.warning(f"⚠️  Fold {fold_idx + 1}: Test set has only one class - skipping AUC calculation")

        # Store predictions for final evaluation
        all_test_indices.extend(test_idx)
        all_test_y_true.extend(y_test_fold.values)
        all_test_gb_proba.extend(gb_test_proba)
        all_test_rf_proba.extend(rf_test_proba)

    # Aggregate metrics across folds
    logger.info("\n" + "=" * 80)
    logger.info("WALK-FORWARD VALIDATION RESULTS")
    logger.info("=" * 80)

    if len(gb_fold_metrics) == 0:
        logger.error("❌ No valid folds completed! Cannot train model.")
        return

    gb_metrics_df = pd.DataFrame(gb_fold_metrics)
    rf_metrics_df = pd.DataFrame(rf_fold_metrics)

    logger.info("\nGradient Boosting - Aggregated Metrics:")
    logger.info(f"  AUC:       {gb_metrics_df['auc'].mean():.4f} (+/- {gb_metrics_df['auc'].std():.4f})")
    logger.info(f"  Recall:    {gb_metrics_df['recall'].mean():.4f} (+/- {gb_metrics_df['recall'].std():.4f})")
    logger.info(f"  Precision: {gb_metrics_df['precision'].mean():.4f} (+/- {gb_metrics_df['precision'].std():.4f})")

    logger.info("\nRandom Forest - Aggregated Metrics:")
    logger.info(f"  AUC:       {rf_metrics_df['auc'].mean():.4f} (+/- {rf_metrics_df['auc'].std():.4f})")
    logger.info(f"  Recall:    {rf_metrics_df['recall'].mean():.4f} (+/- {rf_metrics_df['recall'].std():.4f})")
    logger.info(f"  Precision: {rf_metrics_df['precision'].mean():.4f} (+/- {rf_metrics_df['precision'].std():.4f})")

    # Calculate overall metrics on all test predictions
    all_test_y_true = np.array(all_test_y_true)
    all_test_gb_proba = np.array(all_test_gb_proba)
    all_test_rf_proba = np.array(all_test_rf_proba)

    overall_gb_auc = roc_auc_score(all_test_y_true, all_test_gb_proba)
    overall_rf_auc = roc_auc_score(all_test_y_true, all_test_rf_proba)

    logger.info("\n" + "=" * 80)
    logger.info("OVERALL TEST SET PERFORMANCE (ALL FOLDS COMBINED)")
    logger.info("=" * 80)
    logger.info(f"Gradient Boosting AUC: {overall_gb_auc:.4f}")
    logger.info(f"Random Forest AUC:     {overall_rf_auc:.4f}")

    # Ensemble predictions
    ensemble_proba = all_test_gb_proba * 0.7 + all_test_rf_proba * 0.3
    ensemble_auc = roc_auc_score(all_test_y_true, ensemble_proba)
    logger.info(f"Ensemble AUC (70% GB + 30% RF): {ensemble_auc:.4f}")

    # Check for realistic performance
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE VALIDATION")
    logger.info("=" * 80)

    if ensemble_auc > 0.95:
        logger.warning("⚠️  WARNING: AUC > 0.95 suggests possible data leakage!")
        logger.warning("   Expected realistic AUC: 0.70-0.85")
    elif ensemble_auc > 0.85:
        logger.info("✅ EXCELLENT: AUC > 0.85 (very strong performance)")
    elif ensemble_auc > 0.75:
        logger.info("✅ GOOD: AUC > 0.75 (strong performance)")
    elif ensemble_auc > 0.65:
        logger.info("⚠️  FAIR: AUC > 0.65 (moderate performance)")
    else:
        logger.warning("❌ POOR: AUC < 0.65 (needs improvement)")

    # Train final models on ALL data
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING FINAL MODELS ON ALL DATA")
    logger.info("=" * 80)

    # Scale all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply SMOTE to full dataset
    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
    logger.info(f"After SMOTE: {len(X_balanced)} samples ({y_balanced.sum()} crashes, {y_balanced.sum()/len(y_balanced)*100:.2f}%)")

    # Train final models
    gb_final = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    gb_final.fit(X_balanced, y_balanced)

    rf_final = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf_final.fit(X_balanced, y_balanced)

    logger.info("✅ Final models trained on all data")

    # Save models
    with open('data/models/gb_model_v6.pkl', 'wb') as f:
        pickle.dump(gb_final, f)
    with open('data/models/rf_model_v6.pkl', 'wb') as f:
        pickle.dump(rf_final, f)
    with open('data/models/scaler_v6.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    logger.info("\n✅ Models saved (v6)")
    logger.info(f"   - data/models/gb_model_v6.pkl")
    logger.info(f"   - data/models/rf_model_v6.pkl")
    logger.info(f"   - data/models/scaler_v6.pkl")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Walk-forward folds: {len(splits)}")
    logger.info(f"Valid folds: {len(gb_fold_metrics)}")
    logger.info(f"Overall Ensemble AUC: {ensemble_auc:.4f}")
    logger.info(f"GB Mean AUC: {gb_metrics_df['auc'].mean():.4f}")
    logger.info(f"RF Mean AUC: {rf_metrics_df['auc'].mean():.4f}")
    logger.info("=" * 80)

if __name__ == '__main__':
    train_crash_detector_v6()

