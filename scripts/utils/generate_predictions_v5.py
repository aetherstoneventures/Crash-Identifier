"""
Generate crash probability predictions using trained Statistical Model V3.

This script loads the trained StatisticalModelV3 and generates crash probability
predictions for all dates in the database, storing them in the predictions table.
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from src.utils.database import DatabaseManager, Prediction


def engineer_features_for_prediction(df):
    """
    Engineer features matching what StatisticalModelV3 expects.

    Maps raw DB column names to the names the model uses internally,
    and computes derived features (drawdown, returns, change rates, etc.).
    """
    out = df.copy()

    # Map raw columns to the names StatisticalModelV3 expects
    out['vix_level'] = out['vix_close']
    out['yield_spread_10y_2y'] = out['yield_10y_2y']
    out['yield_spread_10y_3m'] = out['yield_10y_3m']
    # credit_spread_bbb already matches
    # unemployment_rate already matches
    # consumer_sentiment already matches

    # Derived: VIX 20-day pct change
    out['vix_change_20d'] = out['vix_close'].pct_change(20)

    # Derived: Credit spread 20-day change
    out['credit_spread_change_20d'] = out['credit_spread_bbb'].diff(20)

    # Derived: Sahm Rule approximation (3-month avg minus 12-month low of unemployment)
    unemp = out['unemployment_rate']
    unemp_3m_avg = unemp.rolling(63).mean()   # ~3 months of trading days
    unemp_12m_low = unemp.rolling(252).min()   # ~12 months
    out['sahm_rule'] = unemp_3m_avg - unemp_12m_low

    # Derived: Industrial production YoY growth
    out['industrial_prod_growth_yoy'] = out['industrial_production'].pct_change(252)

    # Derived: S&P 500 drawdown from expanding max
    sp500 = out['sp500_close']
    sp500_peak = sp500.expanding().max()
    out['sp500_drawdown'] = (sp500 - sp500_peak) / sp500_peak

    # Derived: S&P 500 returns
    out['sp500_return_5d'] = sp500.pct_change(5)
    out['sp500_return_20d'] = sp500.pct_change(20)

    # Fill NaN from rolling computations
    out = out.ffill().bfill().fillna(0)

    return out


def generate_predictions_v5():
    """Generate crash probability predictions using Statistical Model V3."""
    logger.info("=" * 80)
    logger.info("GENERATING CRASH PROBABILITY PREDICTIONS (STATISTICAL MODEL V3)")
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

    # Load Statistical Model V3
    logger.info("\nLoading Statistical Model V3...")
    model_paths = [
        Path('models/statistical_v3/model.pkl'),
        Path('data/models/crash_predictor_statistical.pkl'),
    ]

    stat_model = None
    for model_path in model_paths:
        if model_path.exists():
            with open(model_path, 'rb') as f:
                stat_model = pickle.load(f)
            logger.info(f"✓ Loaded StatisticalModelV3 from {model_path}")
            break

    if stat_model is None:
        logger.error("❌ No trained Statistical Model V3 found!")
        logger.error("   Run training first (train_statistical_model_v3.py)")
        return

    # Engineer features with correct column names for the model
    logger.info("\nEngineering features...")
    df_features = engineer_features_for_prediction(df)
    logger.info(f"Features: {df_features.shape[1]} columns")

    # Generate predictions
    logger.info("\nGenerating crash probability predictions...")
    crash_proba = stat_model.predict_proba(df_features)

    # Handle NaN values
    crash_proba = np.nan_to_num(crash_proba, nan=0.0)
    crash_proba = np.clip(crash_proba, 0, 1)

    # Store predictions
    logger.info("\nStoring predictions in database...")

    with db.get_session() as session:
        # Clear old predictions
        session.query(Prediction).delete()
        session.commit()

        for i, date in enumerate(df.index):
            pred = Prediction(
                prediction_date=date,
                crash_probability=float(crash_proba[i]),
                model_version='statistical_v3',
                bottom_prediction_date=None,
                recovery_prediction_date=None,
                confidence_interval_lower=None,
                confidence_interval_upper=None
            )
            session.add(pred)

        session.commit()
        logger.info(f"✓ Stored {len(df)} predictions")

    # Print statistics
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Mean Crash Probability: {crash_proba.mean():.4f}")
    logger.info(f"Std:  {crash_proba.std():.4f}")
    logger.info(f"Min:  {crash_proba.min():.4f}")
    logger.info(f"Max:  {crash_proba.max():.4f}")
    logger.info(f"High risk (>0.5): {(crash_proba > 0.5).sum()} days")
    logger.info(f"Very high risk (>0.7): {(crash_proba > 0.7).sum()} days")

    logger.info("\n✅ Predictions generated successfully")


if __name__ == '__main__':
    generate_predictions_v5()

