"""
Train Enhanced Statistical Model V3 - Foundation for Hybrid Approach

This script:
1. Loads 46 dynamically detected crashes
2. Engineers features with regime indicators
3. Trains Statistical Model V3 with multi-threshold logic
4. Evaluates performance with detailed metrics
5. Saves model for hybrid ensemble
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import logging
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from src.utils.database import DatabaseManager, Indicator, CrashEvent
from src.models.crash_prediction.statistical_model_v3 import StatisticalModelV3


def engineer_features_for_statistical_model(df):
    """Engineer features needed by Statistical Model V3."""
    logger.info("Engineering features for Statistical Model V3...")
    
    # Calculate change features
    df['vix_change_pct'] = df['vix_close'].pct_change()
    df['vix_change_5d'] = df['vix_close'].pct_change(5)
    df['credit_spread_change'] = df['credit_spread_bbb'].diff()
    df['unemployment_change'] = df['unemployment_rate'].diff()
    df['industrial_production_change'] = df['industrial_production'].pct_change()
    
    # Calculate drawdown
    df['sp500_peak'] = df['sp500_close'].expanding().max()
    df['sp500_drawdown'] = (df['sp500_close'] - df['sp500_peak']) / df['sp500_peak']
    
    # Calculate returns
    df['sp500_return_5d'] = df['sp500_close'].pct_change(5)
    df['sp500_return_20d'] = df['sp500_close'].pct_change(20)
    
    logger.info(f"✅ Engineered features: {df.shape[1]} total columns")
    
    return df


def create_crash_labels(df, crashes, prediction_window=60):
    """Create crash labels with prediction window."""
    labels = pd.Series(0, index=df.index)
    
    for crash in crashes:
        start_date = pd.to_datetime(crash.start_date)
        # Label days within prediction_window before crash start
        warning_start = start_date - pd.Timedelta(days=prediction_window)
        warning_period = (df.index >= warning_start) & (df.index < start_date)
        labels[warning_period] = 1
    
    logger.info(f"✅ Created labels: {labels.sum()} positive samples ({labels.sum()/len(labels)*100:.1f}%)")
    
    return labels


def train_statistical_model_v3():
    """Train and evaluate Statistical Model V3."""
    logger.info("=" * 80)
    logger.info("TRAINING STATISTICAL MODEL V3 - HYBRID APPROACH FOUNDATION")
    logger.info("=" * 80)
    
    # Initialize database
    db = DatabaseManager()
    
    # Load data
    logger.info("\nLoading data from database...")
    with db.get_session() as session:
        indicators = session.query(Indicator).order_by(Indicator.date).all()
        crashes = session.query(CrashEvent).all()
        session.expunge_all()
    
    logger.info(f"✅ Loaded {len(indicators)} days of data")
    logger.info(f"✅ Loaded {len(crashes)} crashes")
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'date': ind.date,
        'sp500_close': ind.sp500_close,
        'vix_close': ind.vix_close,
        'yield_10y_3m': ind.yield_10y_3m,
        'yield_10y_2y': ind.yield_10y_2y,
        'credit_spread_bbb': ind.credit_spread_bbb,
        'unemployment_rate': ind.unemployment_rate,
        'industrial_production': ind.industrial_production,
        'consumer_sentiment': ind.consumer_sentiment,
        'm2_money_supply': ind.m2_money_supply,
        'debt_to_gdp': ind.debt_to_gdp,
        'fed_funds_rate': ind.fed_funds_rate,
        'housing_starts': ind.housing_starts,
        'lei': ind.lei
    } for ind in indicators])
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    
    # Engineer features
    df = engineer_features_for_statistical_model(df)
    
    # Create labels
    y = create_crash_labels(df, crashes, prediction_window=60)
    
    # Remove NaN values
    valid_idx = ~df.isna().any(axis=1)
    df_clean = df[valid_idx]
    y_clean = y[valid_idx]
    
    logger.info(f"\n✅ Final dataset: {len(df_clean)} samples")
    logger.info(f"   Positive samples: {y_clean.sum()} ({y_clean.sum()/len(y_clean)*100:.1f}%)")
    logger.info(f"   Negative samples: {(~y_clean.astype(bool)).sum()} ({(~y_clean.astype(bool)).sum()/len(y_clean)*100:.1f}%)")
    
    # Time-series split (80/20)
    split_idx = int(len(df_clean) * 0.8)
    X_train, X_test = df_clean.iloc[:split_idx], df_clean.iloc[split_idx:]
    y_train, y_test = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]
    
    logger.info(f"\n✅ Train set: {len(X_train)} samples ({y_train.sum()} crashes)")
    logger.info(f"✅ Test set:  {len(X_test)} samples ({y_test.sum()} crashes)")
    
    # Train model
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING STATISTICAL MODEL V3")
    logger.info("=" * 80)
    
    model = StatisticalModelV3()
    model.train(X_train, y_train, X_test, y_test)
    
    # Predict with explanations
    logger.info("\nGenerating predictions with explanations...")
    train_proba, train_explanations = model.predict_with_explanation(X_train)
    test_proba, test_explanations = model.predict_with_explanation(X_test)
    
    # Evaluate
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 80)
    
    # AUC scores
    train_auc = roc_auc_score(y_train, train_proba) if len(np.unique(y_train)) >= 2 else float('nan')
    test_auc = roc_auc_score(y_test, test_proba) if len(np.unique(y_test)) >= 2 else float('nan')
    
    logger.info(f"\nAUC Scores:")
    logger.info(f"  Train AUC: {train_auc:.4f}")
    logger.info(f"  Test AUC:  {test_auc:.4f}")
    logger.info(f"  Overfitting gap: {train_auc - test_auc:.4f} {'✓ OK' if train_auc - test_auc < 0.1 else '⚠️ HIGH'}")
    
    # Classification metrics at different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    logger.info(f"\nClassification Metrics at Different Thresholds:")
    logger.info("-" * 80)
    
    for threshold in thresholds:
        test_pred = (test_proba >= threshold).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_test, test_pred, zero_division=0)
        recall = recall_score(y_test, test_pred, zero_division=0)
        f1 = f1_score(y_test, test_pred, zero_division=0)
        
        logger.info(f"\nThreshold = {threshold:.1f}:")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall:    {recall:.3f}")
        logger.info(f"  F1 Score:  {f1:.3f}")
    
    # Detailed classification report at 0.5 threshold
    test_pred_05 = (test_proba >= 0.5).astype(int)
    logger.info(f"\nDetailed Classification Report (threshold=0.5):")
    logger.info("\n" + classification_report(y_test, test_pred_05, target_names=['No Crash', 'Crash']))
    
    # Analyze factor contributions
    logger.info("\n" + "=" * 80)
    logger.info("FACTOR ANALYSIS")
    logger.info("=" * 80)

    # Average factor scores for crash vs non-crash periods
    # Reset index to align with y_test
    test_explanations_reset = test_explanations.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)

    crash_explanations = test_explanations_reset[y_test_reset == 1]
    normal_explanations = test_explanations_reset[y_test_reset == 0]
    
    factors = ['yield_curve_score', 'volatility_score', 'credit_stress_score', 
               'economic_score', 'market_momentum_score', 'sentiment_score']
    
    logger.info(f"\nAverage Factor Scores:")
    logger.info(f"{'Factor':<25} {'Crash Periods':<15} {'Normal Periods':<15} {'Difference':<15}")
    logger.info("-" * 70)
    
    for factor in factors:
        crash_avg = crash_explanations[factor].mean() if len(crash_explanations) > 0 else 0
        normal_avg = normal_explanations[factor].mean() if len(normal_explanations) > 0 else 0
        diff = crash_avg - normal_avg
        
        logger.info(f"{factor:<25} {crash_avg:<15.3f} {normal_avg:<15.3f} {diff:<15.3f}")
    
    # Regime distribution
    logger.info(f"\nVolatility Regime Distribution:")
    regime_counts = test_explanations['regime'].value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(test_explanations) * 100
        logger.info(f"  {regime.capitalize():<10}: {count:>5} ({pct:>5.1f}%)")
    
    # Save model
    logger.info("\n" + "=" * 80)
    logger.info("SAVING MODEL")
    logger.info("=" * 80)
    
    models_dir = Path('models/statistical_v3')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    with open(models_dir / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata
    metadata = {
        'model_type': 'Statistical V3',
        'train_auc': train_auc,
        'test_auc': test_auc,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'num_crashes': len(crashes),
        'crash_threshold': model.crash_threshold,
        'thresholds': model.thresholds,
        'base_weights': model.base_weights
    }
    
    with open(models_dir / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    logger.info(f"✅ Saved Statistical Model V3 to {models_dir}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Test AUC: {test_auc:.4f}")
    logger.info(f"Model ready for hybrid ensemble")
    logger.info("=" * 80)


if __name__ == '__main__':
    train_statistical_model_v3()

