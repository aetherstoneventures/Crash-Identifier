"""
Statistical Model V3 - Walk-Forward Validation & Comparison with ML
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import logging
import pickle
import os
import sqlite3

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from src.utils.walk_forward_validation import WalkForwardValidator
from src.models.crash_prediction.statistical_model_v3 import StatisticalModelV3

def prepare_features_for_statistical(df):
    """Prepare features in the format expected by StatisticalModelV3."""
    X = pd.DataFrame(index=df.index)
    
    # Direct mappings
    X['yield_10y_2y'] = df['yield_10y_2y']
    X['yield_10y_3m'] = df['yield_10y_3m']
    X['vix_close'] = df['vix_close']
    X['credit_spread_bbb'] = df['credit_spread_bbb']
    X['unemployment_rate'] = df['unemployment_rate']
    X['industrial_production'] = df['industrial_production']
    X['consumer_sentiment'] = df['consumer_sentiment']
    
    # Calculate derived features
    X['sp500_drawdown'] = (df['sp500_close'] / df['sp500_close'].rolling(252).max() - 1)
    X['sp500_return_5d'] = df['sp500_close'].pct_change(5)
    X['sp500_return_20d'] = df['sp500_close'].pct_change(20)
    
    X['vix_change_pct'] = df['vix_close'].pct_change()
    X['vix_change_5d'] = df['vix_close'].pct_change(5)
    
    X['credit_spread_change'] = df['credit_spread_bbb'].diff(20)
    X['unemployment_change'] = df['unemployment_rate'].diff(20)
    X['industrial_production_change'] = df['industrial_production'].diff(20)
    
    # Fill NaN
    X = X.ffill().bfill().fillna(0)
    
    return X

def main():
    logger.info("="*80)
    logger.info("STATISTICAL MODEL V3 - WALK-FORWARD VALIDATION")
    logger.info("="*80)

    # Use sqlite3 directly to avoid DatabaseManager mutex issues
    conn = sqlite3.connect('data/market_crash.db')
    df = pd.read_sql_query("SELECT * FROM indicators ORDER BY date", conn)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    crashes = pd.read_sql_query("SELECT start_date, end_date FROM crash_events", conn)
    crash_events = list(zip(crashes['start_date'], crashes['end_date']))
    conn.close()

    logger.info(f"Data: {df.shape}, Crashes: {len(crash_events)}")
    
    X = prepare_features_for_statistical(df)
    logger.info(f"Features: {X.shape[1]}")
    
    y = pd.Series(0, index=df.index)
    for crash_start, _ in crash_events:
        crash_date = pd.to_datetime(crash_start)
        lookback = crash_date - pd.Timedelta(days=90)
        y[(df.index >= lookback) & (df.index < crash_date)] = 1
    
    logger.info(f"Crash samples: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    
    validator = WalkForwardValidator(window_size=252, step_size=21, min_train_size=1260)
    splits = validator.expanding_window_split(X, y)
    logger.info(f"Generated {len(splits)} folds\n")
    
    stat_metrics = []
    
    for i, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        if y_train.sum() < 2 or y_test.sum() == 0 or y_test.sum() == len(y_test):
            continue
        
        # Train statistical model (calibrates thresholds)
        stat_model = StatisticalModelV3()
        stat_model.train(X_train, y_train)
        
        # Predict
        stat_proba = stat_model.predict_proba(X_test)
        
        # Calculate metrics
        if y_test.sum() > 0 and y_test.sum() < len(y_test):
            auc = roc_auc_score(y_test, stat_proba)
            stat_metrics.append(auc)
            
            if (i + 1) % 50 == 0:
                logger.info(f"Fold {i+1}: Statistical AUC={auc:.4f}")
    
    logger.info("\n" + "="*80)
    logger.info("STATISTICAL MODEL V3 - WALK-FORWARD RESULTS")
    logger.info("="*80)
    logger.info(f"Valid folds: {len(stat_metrics)}")
    logger.info(f"\nStatistical Model V3:")
    logger.info(f"  Mean AUC: {np.mean(stat_metrics):.4f} (+/- {np.std(stat_metrics):.4f})")
    
    # Performance assessment
    mean_auc = np.mean(stat_metrics)
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE ASSESSMENT")
    logger.info("="*80)
    if mean_auc > 0.95:
        logger.warning("⚠️  WARNING: AUC > 0.95 suggests possible data leakage!")
    elif mean_auc > 0.80:
        logger.info("✅ EXCELLENT: AUC > 0.80 (very strong performance)")
    elif mean_auc > 0.70:
        logger.info("✅ GOOD: AUC > 0.70 (strong performance)")
    elif mean_auc > 0.60:
        logger.info("⚠️  FAIR: AUC > 0.60 (moderate performance)")
    else:
        logger.warning("❌ POOR: AUC < 0.60 (needs improvement)")
    
    # Train final model on all data
    logger.info("\n" + "="*80)
    logger.info("TRAINING FINAL STATISTICAL MODEL ON ALL DATA")
    logger.info("="*80)
    
    stat_final = StatisticalModelV3()
    stat_final.train(X, y)
    
    # Save model
    os.makedirs('data/models', exist_ok=True)
    with open('data/models/statistical_model_v3.pkl', 'wb') as f:
        pickle.dump(stat_final, f)
    
    logger.info("✅ Statistical model saved:")
    logger.info("   - data/models/statistical_model_v3.pkl")
    logger.info("="*80)

if __name__ == '__main__':
    main()
