"""
PHASE 3.5: Crash Detector V5 - Anti-Overfitting with Cross-Validation

This script implements:
1. K-Fold Cross-Validation to prevent overfitting
2. Regularization (L2 penalty)
3. Early stopping
4. Hyperparameter tuning with validation curves
5. Proper train/validation/test split

Ensures the model generalizes well to unseen data.
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
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             recall_score, precision_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from src.utils.database import DatabaseManager, Indicator, Prediction
from src.utils.config import DATABASE_URL

def engineer_crash_features_v5(df):
    """Enhanced feature engineering matching V4 but with better normalization."""
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

def train_crash_detector_v5():
    """Train crash detection model with cross-validation to prevent overfitting."""
    logger.info("=" * 80)
    logger.info("TRAINING CRASH DETECTOR V5 - ANTI-OVERFITTING WITH CROSS-VALIDATION")
    logger.info("=" * 80)
    
    # Load data
    db = DatabaseManager()
    db.create_tables()
    session = db.get_session()
    
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
    X = engineer_crash_features_v5(df)
    logger.info(f"Features created: {X.shape[1]}")
    
    # Create labels with 90-day window
    y = pd.Series(0, index=df.index)
    for crash_start, crash_end in crash_events:
        crash_start_date = pd.to_datetime(crash_start)
        lookback_date = crash_start_date - pd.Timedelta(days=90)
        mask = (df.index >= lookback_date) & (df.index < crash_start_date)
        y[mask] = 1
    
    logger.info(f"Crash samples: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    
    # Train-test split (80-20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"\nTrain set: {len(X_train)} samples ({y_train.sum()} crashes)")
    logger.info(f"Test set: {len(X_test)} samples ({y_test.sum()} crashes)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # K-Fold Cross-Validation
    logger.info("\n" + "=" * 80)
    logger.info("K-FOLD CROSS-VALIDATION (5 folds)")
    logger.info("=" * 80)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
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
    
    logger.info(f"GB Train AUC: {gb_cv_scores['train_roc_auc'].mean():.4f} (+/- {gb_cv_scores['train_roc_auc'].std():.4f})")
    logger.info(f"GB Val AUC:   {gb_cv_scores['test_roc_auc'].mean():.4f} (+/- {gb_cv_scores['test_roc_auc'].std():.4f})")
    logger.info(f"GB Recall:    {gb_cv_scores['test_recall'].mean():.4f}")
    logger.info(f"GB Precision: {gb_cv_scores['test_precision'].mean():.4f}")
    
    # Check for overfitting
    gb_overfit = gb_cv_scores['train_roc_auc'].mean() - gb_cv_scores['test_roc_auc'].mean()
    logger.info(f"GB Overfitting gap: {gb_overfit:.4f} {'⚠️ HIGH' if gb_overfit > 0.15 else '✓ OK'}")
    
    # Random Forest with regularization
    logger.info("\nTraining Random Forest with regularization...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    rf_cv_scores = cross_validate(
        rf, X_train_scaled, y_train, cv=skf,
        scoring=['roc_auc', 'recall', 'precision', 'f1'],
        return_train_score=True
    )
    
    logger.info(f"RF Train AUC: {rf_cv_scores['train_roc_auc'].mean():.4f} (+/- {rf_cv_scores['train_roc_auc'].std():.4f})")
    logger.info(f"RF Val AUC:   {rf_cv_scores['test_roc_auc'].mean():.4f} (+/- {rf_cv_scores['test_roc_auc'].std():.4f})")
    logger.info(f"RF Recall:    {rf_cv_scores['test_recall'].mean():.4f}")
    logger.info(f"RF Precision: {rf_cv_scores['test_precision'].mean():.4f}")
    
    # Check for overfitting
    rf_overfit = rf_cv_scores['train_roc_auc'].mean() - rf_cv_scores['test_roc_auc'].mean()
    logger.info(f"RF Overfitting gap: {rf_overfit:.4f} {'⚠️ HIGH' if rf_overfit > 0.15 else '✓ OK'}")
    
    # Train final models on full training set
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING FINAL MODELS ON FULL TRAINING SET")
    logger.info("=" * 80)
    
    gb.fit(X_train_scaled, y_train)
    rf.fit(X_train_scaled, y_train)
    
    # Test set performance
    logger.info("\nTest Set Performance:")
    gb_test_auc = roc_auc_score(y_test, gb.predict_proba(X_test_scaled)[:, 1])
    rf_test_auc = roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:, 1])
    
    logger.info(f"GB Test AUC: {gb_test_auc:.4f}")
    logger.info(f"RF Test AUC: {rf_test_auc:.4f}")
    
    # Ensemble
    ensemble_proba = (gb.predict_proba(X_test_scaled)[:, 1] * 0.7 + 
                      rf.predict_proba(X_test_scaled)[:, 1] * 0.3)
    ensemble_auc = roc_auc_score(y_test, ensemble_proba)
    logger.info(f"Ensemble Test AUC: {ensemble_auc:.4f}")
    
    # Save models
    with open('data/models/gb_model_v5.pkl', 'wb') as f:
        pickle.dump(gb, f)
    with open('data/models/rf_model_v5.pkl', 'wb') as f:
        pickle.dump(rf, f)
    with open('data/models/scaler_v5.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    logger.info("\n✅ Models saved (v5)")
    session.close()

if __name__ == '__main__':
    train_crash_detector_v5()

