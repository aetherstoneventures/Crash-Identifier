"""
Train bottom prediction model to identify optimal re-entry timing after market crashes.

This model predicts:
1. When the market has bottomed out after a crash
2. Optimal re-entry timing for investors
3. Expected recovery timeline

Based on historical crash patterns and recovery indicators.
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from src.utils.database import DatabaseManager, Indicator, CrashEvent
from src.feature_engineering.crash_indicators import CrashIndicators

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_crash_events():
    """Load historical crash events from database."""
    db = DatabaseManager()
    session = db.get_session()

    crashes = session.query(CrashEvent).order_by(CrashEvent.start_date).all()
    session.close()

    crash_data = []
    for crash in crashes:
        crash_data.append({
            'crash_id': crash.id,
            'start_date': pd.to_datetime(crash.start_date),
            'trough_date': pd.to_datetime(crash.trough_date),
            'end_date': pd.to_datetime(crash.end_date),
            'max_drawdown': crash.max_drawdown,
            'recovery_months': crash.recovery_months,
            'crash_type': crash.crash_type
        })

    return pd.DataFrame(crash_data)


def load_indicators():
    """Load all indicators from database."""
    db = DatabaseManager()
    session = db.get_session()
    
    indicators = session.query(Indicator).order_by(Indicator.date).all()
    session.close()
    
    data = []
    for ind in indicators:
        data.append({
            'date': ind.date,
            'sp500_close': ind.sp500_close,
            'vix_close': ind.vix_close,
            'yield_10y_3m': ind.yield_10y_3m,
            'credit_spread_bbb': ind.credit_spread_bbb,
            'unemployment_rate': ind.unemployment_rate,
            'consumer_sentiment': ind.consumer_sentiment,
            'lei': ind.lei,
        })
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    
    return df


def create_bottom_prediction_features(ind_df, crash_start_date):
    """
    Create features for predicting days to bottom from crash start.
    
    Args:
        ind_df: DataFrame with indicators
        crash_start_date: Date when crash started
    
    Returns:
        Dictionary of features
    """
    # Get data at crash start
    crash_data = ind_df[ind_df.index <= crash_start_date].iloc[-1]
    
    # Calculate features
    features = {}
    
    # 1. VIX level at crash start (higher VIX = more panic = longer to bottom)
    features['vix_at_crash'] = crash_data['vix_close']
    
    # 2. Yield curve inversion severity
    features['yield_curve'] = crash_data['yield_10y_3m']
    
    # 3. Credit spread (wider = more stress = longer recovery)
    features['credit_spread'] = crash_data['credit_spread_bbb']
    
    # 4. Unemployment rate (higher = worse economy = longer recovery)
    features['unemployment'] = crash_data['unemployment_rate']
    
    # 5. Consumer sentiment (lower = more pessimism = longer recovery)
    features['sentiment'] = crash_data['consumer_sentiment']
    
    # 6. Leading economic index
    features['lei'] = crash_data['lei']
    
    # 7. S&P 500 momentum (30-day change before crash)
    sp500_30d_ago = ind_df[ind_df.index <= crash_start_date - timedelta(days=30)].iloc[-1]['sp500_close']
    features['sp500_momentum'] = (crash_data['sp500_close'] / sp500_30d_ago - 1) * 100
    
    # 8. VIX spike (change from 30 days ago)
    vix_30d_ago = ind_df[ind_df.index <= crash_start_date - timedelta(days=30)].iloc[-1]['vix_close']
    features['vix_spike'] = crash_data['vix_close'] - vix_30d_ago
    
    return features


def train_bottom_predictor():
    """Train model to predict days to bottom after crash."""
    logger.info("=" * 80)
    logger.info("TRAINING BOTTOM PREDICTION MODEL")
    logger.info("=" * 80)
    
    # Load data
    logger.info("\nLoading historical crash events...")
    crashes_df = load_crash_events()
    logger.info(f"✅ Loaded {len(crashes_df)} historical crashes")
    
    logger.info("\nLoading indicators...")
    ind_df = load_indicators()
    logger.info(f"✅ Loaded {len(ind_df)} days of indicator data")
    
    # Create training data
    logger.info("\nCreating training features...")
    X_data = []
    y_days_to_bottom = []
    y_recovery_days = []
    
    for idx, crash in crashes_df.iterrows():
        try:
            # Calculate days to bottom
            days_to_bottom = (crash['trough_date'] - crash['start_date']).days
            
            # Calculate total recovery days
            recovery_days = (crash['end_date'] - crash['start_date']).days
            
            # Create features
            features = create_bottom_prediction_features(ind_df, crash['start_date'])
            
            X_data.append(features)
            y_days_to_bottom.append(days_to_bottom)
            y_recovery_days.append(recovery_days)
            
            logger.info(f"  {crash['crash_type']}: {days_to_bottom} days to bottom, {recovery_days} days to recovery")
            
        except Exception as e:
            logger.warning(f"  Skipped {crash['crash_type']}: {e}")
            continue
    
    X = pd.DataFrame(X_data)
    y_bottom = np.array(y_days_to_bottom)
    y_recovery = np.array(y_recovery_days)
    
    logger.info(f"\n✅ Created {len(X)} training samples with {len(X.columns)} features")
    
    # Train model for days to bottom
    logger.info("\n" + "=" * 80)
    logger.info("Training Days-to-Bottom Predictor")
    logger.info("=" * 80)
    
    model_bottom = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    model_bottom.fit(X, y_bottom)
    
    # Evaluate
    predictions_bottom = model_bottom.predict(X)
    mae_bottom = mean_absolute_error(y_bottom, predictions_bottom)
    r2_bottom = r2_score(y_bottom, predictions_bottom)
    
    logger.info(f"\n✅ Days-to-Bottom Model Performance:")
    logger.info(f"   MAE: {mae_bottom:.1f} days")
    logger.info(f"   R²: {r2_bottom:.3f}")
    
    # Train model for recovery days
    logger.info("\n" + "=" * 80)
    logger.info("Training Recovery-Days Predictor")
    logger.info("=" * 80)
    
    model_recovery = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    model_recovery.fit(X, y_recovery)
    
    # Evaluate
    predictions_recovery = model_recovery.predict(X)
    mae_recovery = mean_absolute_error(y_recovery, predictions_recovery)
    r2_recovery = r2_score(y_recovery, predictions_recovery)
    
    logger.info(f"\n✅ Recovery-Days Model Performance:")
    logger.info(f"   MAE: {mae_recovery:.1f} days")
    logger.info(f"   R²: {r2_recovery:.3f}")
    
    # Save models
    logger.info("\n" + "=" * 80)
    logger.info("Saving Models")
    logger.info("=" * 80)
    
    models_dir = Path('data/models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model_bottom, models_dir / 'bottom_predictor_days_to_bottom.pkl')
    joblib.dump(model_recovery, models_dir / 'bottom_predictor_recovery_days.pkl')
    joblib.dump(list(X.columns), models_dir / 'bottom_predictor_features.pkl')
    
    logger.info("✅ Saved models:")
    logger.info("   - bottom_predictor_days_to_bottom.pkl")
    logger.info("   - bottom_predictor_recovery_days.pkl")
    logger.info("   - bottom_predictor_features.pkl")
    
    # Test on historical crashes
    logger.info("\n" + "=" * 80)
    logger.info("TESTING ON HISTORICAL CRASHES")
    logger.info("=" * 80)
    
    for idx, crash in crashes_df.iterrows():
        try:
            features = create_bottom_prediction_features(ind_df, crash['start_date'])
            X_test = pd.DataFrame([features])
            
            pred_days_to_bottom = model_bottom.predict(X_test)[0]
            pred_recovery_days = model_recovery.predict(X_test)[0]
            
            actual_days_to_bottom = (crash['trough_date'] - crash['start_date']).days
            actual_recovery_days = (crash['end_date'] - crash['start_date']).days
            
            error_bottom = abs(pred_days_to_bottom - actual_days_to_bottom)
            error_recovery = abs(pred_recovery_days - actual_recovery_days)
            
            logger.info(f"\n{crash['crash_type']} ({crash['start_date'].strftime('%Y-%m-%d')}):")
            logger.info(f"  Days to Bottom: Predicted {pred_days_to_bottom:.0f}, Actual {actual_days_to_bottom}, Error {error_bottom:.0f}")
            logger.info(f"  Recovery Days: Predicted {pred_recovery_days:.0f}, Actual {actual_recovery_days}, Error {error_recovery:.0f}")
            logger.info(f"  Optimal Re-Entry: {(crash['start_date'] + timedelta(days=int(pred_days_to_bottom))).strftime('%Y-%m-%d')}")
            
        except Exception as e:
            logger.warning(f"  Error testing {crash['crash_type']}: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ BOTTOM PREDICTION MODEL TRAINING COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    train_bottom_predictor()

