"""
Generate bottom predictions for all dates in the database.

This script:
1. Loads the trained bottom prediction models
2. For each date, calculates features and predicts days to bottom
3. Stores predictions in the database
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import joblib
from pathlib import Path

from src.utils.database import DatabaseManager, Indicator, Prediction
from src.feature_engineering.crash_indicators import CrashIndicators

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_models():
    """Load trained bottom prediction models."""
    models_dir = Path('data/models')
    
    model_bottom = joblib.load(models_dir / 'bottom_predictor_days_to_bottom.pkl')
    model_recovery = joblib.load(models_dir / 'bottom_predictor_recovery_days.pkl')
    feature_names = joblib.load(models_dir / 'bottom_predictor_features.pkl')
    
    return model_bottom, model_recovery, feature_names


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


def create_features_for_date(ind_df, target_date):
    """
    Create features for predicting bottom from a given date.
    
    Args:
        ind_df: DataFrame with indicators
        target_date: Date to create features for
    
    Returns:
        Dictionary of features
    """
    # Get data at target date
    if target_date not in ind_df.index:
        # Find nearest date
        nearest_idx = ind_df.index.get_indexer([target_date], method='nearest')[0]
        target_date = ind_df.index[nearest_idx]
    
    current_data = ind_df.loc[target_date]
    
    # Calculate features
    features = {}
    
    # 1. VIX level
    features['vix_at_crash'] = current_data['vix_close']
    
    # 2. Yield curve
    features['yield_curve'] = current_data['yield_10y_3m']
    
    # 3. Credit spread
    features['credit_spread'] = current_data['credit_spread_bbb']
    
    # 4. Unemployment rate
    features['unemployment'] = current_data['unemployment_rate']
    
    # 5. Consumer sentiment
    features['sentiment'] = current_data['consumer_sentiment']
    
    # 6. Leading economic index
    features['lei'] = current_data['lei']
    
    # 7. S&P 500 momentum (30-day change)
    date_30d_ago = target_date - timedelta(days=30)
    if date_30d_ago in ind_df.index:
        sp500_30d_ago = ind_df.loc[date_30d_ago, 'sp500_close']
    else:
        # Find nearest
        nearest_idx = ind_df.index.get_indexer([date_30d_ago], method='nearest')[0]
        sp500_30d_ago = ind_df.iloc[nearest_idx]['sp500_close']
    
    features['sp500_momentum'] = (current_data['sp500_close'] / sp500_30d_ago - 1) * 100
    
    # 8. VIX spike (change from 30 days ago)
    if date_30d_ago in ind_df.index:
        vix_30d_ago = ind_df.loc[date_30d_ago, 'vix_close']
    else:
        nearest_idx = ind_df.index.get_indexer([date_30d_ago], method='nearest')[0]
        vix_30d_ago = ind_df.iloc[nearest_idx]['vix_close']
    
    features['vix_spike'] = current_data['vix_close'] - vix_30d_ago
    
    return features


def generate_bottom_predictions():
    """Generate bottom predictions for all dates."""
    logger.info("=" * 80)
    logger.info("GENERATING BOTTOM PREDICTIONS")
    logger.info("=" * 80)
    
    # Load models
    logger.info("\nLoading trained models...")
    model_bottom, model_recovery, feature_names = load_models()
    logger.info(f"✅ Loaded models with {len(feature_names)} features")
    
    # Load indicators
    logger.info("\nLoading indicators...")
    ind_df = load_indicators()
    logger.info(f"✅ Loaded {len(ind_df)} days of indicator data")
    
    # Load existing predictions
    logger.info("\nLoading existing predictions...")
    db = DatabaseManager()
    session = db.get_session()
    
    predictions = session.query(Prediction).order_by(Prediction.prediction_date).all()
    logger.info(f"✅ Found {len(predictions)} existing predictions to update")
    
    # Generate bottom predictions for each date
    logger.info("\nGenerating bottom predictions...")
    updated_count = 0
    
    for pred in predictions:
        try:
            # Create features for this date
            features = create_features_for_date(ind_df, pred.prediction_date)
            X = pd.DataFrame([features])[feature_names]
            
            # Predict days to bottom and recovery
            days_to_bottom = int(model_bottom.predict(X)[0])
            days_to_recovery = int(model_recovery.predict(X)[0])
            
            # Calculate dates
            bottom_date = pred.prediction_date + timedelta(days=days_to_bottom)
            recovery_date = pred.prediction_date + timedelta(days=days_to_recovery)
            
            # Update prediction
            pred.bottom_prediction_date = bottom_date
            pred.recovery_prediction_date = recovery_date
            
            updated_count += 1
            
            if updated_count % 1000 == 0:
                logger.info(f"  Processed {updated_count}/{len(predictions)} predictions...")
            
        except Exception as e:
            logger.warning(f"  Error processing {pred.prediction_date}: {e}")
            continue
    
    # Commit changes
    logger.info(f"\nSaving {updated_count} updated predictions to database...")
    session.commit()
    session.close()
    
    logger.info(f"✅ Updated {updated_count} predictions with bottom predictions")
    
    # Show sample predictions
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLE BOTTOM PREDICTIONS")
    logger.info("=" * 80)
    
    session = db.get_session()
    recent_predictions = session.query(Prediction).order_by(Prediction.prediction_date.desc()).limit(10).all()
    
    for pred in reversed(recent_predictions):
        if pred.bottom_prediction_date and pred.recovery_prediction_date:
            days_to_bottom = (pred.bottom_prediction_date - pred.prediction_date).days
            days_to_recovery = (pred.recovery_prediction_date - pred.prediction_date).days
            
            logger.info(f"\n{pred.prediction_date.strftime('%Y-%m-%d')}:")
            logger.info(f"  Crash Probability: {pred.crash_probability:.1%}")
            logger.info(f"  Days to Bottom: {days_to_bottom}")
            logger.info(f"  Bottom Date: {pred.bottom_prediction_date.strftime('%Y-%m-%d')}")
            logger.info(f"  Recovery Date: {pred.recovery_prediction_date.strftime('%Y-%m-%d')}")
    
    session.close()
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ BOTTOM PREDICTIONS GENERATION COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    generate_bottom_predictions()

