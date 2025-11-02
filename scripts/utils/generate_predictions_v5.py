"""
Generate crash probability predictions using V5 ML models and V2 statistical model.
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import pickle

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from src.utils.database import DatabaseManager, Prediction
from src.utils.config import DATABASE_URL

def calculate_statistical_crash_probability_v2(df):
    """
    Calculate crash probability using improved statistical methods.
    """
    proba = pd.Series(0.0, index=df.index)

    # 1. YIELD CURVE INVERSION (Weight: 25%)
    yield_curve = df['yield_10y_3m'].fillna(0)
    yield_inversion = (yield_curve < 0).astype(float)
    yield_inversion_days = (yield_curve < 0).rolling(20).sum() / 20
    yield_signal = (yield_inversion * 0.5 + yield_inversion_days * 0.5).clip(0, 1)
    proba += yield_signal * 0.25

    # 2. CREDIT STRESS (Weight: 20%)
    credit_spread = df['credit_spread_bbb'].fillna(0)
    credit_ma20 = credit_spread.rolling(20).mean()
    credit_std = credit_spread.rolling(20).std()
    credit_zscore = (credit_spread - credit_ma20) / (credit_std + 0.001)
    credit_widening = (credit_spread.diff(5) > 0).astype(float)
    credit_signal = (credit_zscore.clip(0, 3) / 3 * 0.6 + credit_widening * 0.4).clip(0, 1)
    proba += credit_signal * 0.20

    # 3. VOLATILITY SPIKE (Weight: 20%)
    vix = df['vix_close'].fillna(20)
    vix_ma20 = vix.rolling(20).mean()
    vix_spike = (vix > vix_ma20 * 1.5).astype(float)
    vix_elevated = (vix > 25).astype(float)
    vix_signal = (vix_spike * 0.6 + vix_elevated * 0.4).clip(0, 1)
    proba += vix_signal * 0.20

    # 4. ECONOMIC DETERIORATION (Weight: 20%)
    unemployment = df['unemployment_rate'].fillna(0)
    unemployment_rising = (unemployment.diff(5) > 0).astype(float)
    industrial_prod = df['industrial_production'].fillna(0)
    industrial_declining = (industrial_prod.diff(5) < 0).astype(float)
    econ_signal = (unemployment_rising * 0.5 + industrial_declining * 0.5).clip(0, 1)
    proba += econ_signal * 0.20

    # 5. VALUATION EXTREMES (Weight: 10%)
    sp500 = df['sp500_close'].fillna(0)
    sp500_returns_20d = sp500.pct_change(20)
    sp500_volatility = sp500.pct_change().rolling(20).std()
    sp500_high_vol = (sp500_volatility > sp500_volatility.rolling(60).mean() * 1.5).astype(float)
    sp500_negative = (sp500_returns_20d < 0).astype(float)
    valuation_signal = (sp500_high_vol * 0.6 + sp500_negative * 0.4).clip(0, 1)
    proba += valuation_signal * 0.10

    # 6. MOMENTUM REVERSAL (Weight: 5%)
    sp500_returns_5d = sp500.pct_change(5)
    momentum_reversal = (sp500_returns_5d < -0.05).astype(float)
    proba += momentum_reversal * 0.05

    proba = proba.clip(0, 1)
    return proba

def engineer_crash_features_v5(df):
    """Feature engineering matching train_crash_detector_v5.py"""
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

def generate_predictions_v5():
    """Generate crash probability predictions using V5 ML and V2 statistical models."""
    logger.info("=" * 80)
    logger.info("GENERATING CRASH PROBABILITY PREDICTIONS (V5 ML + V2 STATISTICAL)")
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
    
    # Load ML models
    logger.info("\nLoading ML models (V5)...")
    with open('data/models/gb_model_v5.pkl', 'rb') as f:
        gb = pickle.load(f)
    with open('data/models/rf_model_v5.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('data/models/scaler_v5.pkl', 'rb') as f:
        scaler = pickle.load(f)
    logger.info("✓ ML models loaded")
    
    logger.info("✓ Statistical model function ready")
    
    # Engineer features for ML
    logger.info("\nEngineering ML features...")
    X = engineer_crash_features_v5(df)
    logger.info(f"Features: {X.shape[1]}")
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Generate ML predictions
    logger.info("\nGenerating ML predictions...")
    gb_proba = gb.predict_proba(X_scaled)[:, 1]
    rf_proba = rf.predict_proba(X_scaled)[:, 1]
    ml_proba = gb_proba * 0.7 + rf_proba * 0.3
    
    # Use ML model only (V5 achieved 81.8% recall)
    logger.info("\nUsing ML model predictions (V5 with cross-validation)...")
    ensemble_proba = ml_proba

    # Handle NaN values
    ensemble_proba = np.nan_to_num(ensemble_proba, nan=0.0)
    ensemble_proba = np.clip(ensemble_proba, 0, 1)

    # Store predictions
    logger.info("Storing predictions in database...")
    
    # Clear old predictions
    session.query(Prediction).delete()
    session.commit()
    
    for i, date in enumerate(df.index):
        pred = Prediction(
            prediction_date=date,
            crash_probability=float(ensemble_proba[i]),
            model_version='v5_ensemble',
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
    logger.info(f"ML Model Mean: {ml_proba.mean():.4f}")
    logger.info(f"ML Model Std: {ml_proba.std():.4f}")
    logger.info(f"ML Model Min: {ml_proba.min():.4f}")
    logger.info(f"ML Model Max: {ml_proba.max():.4f}")
    logger.info(f"High risk (>0.5): {(ensemble_proba > 0.5).sum()} days")
    logger.info(f"Very high risk (>0.7): {(ensemble_proba > 0.7).sum()} days")
    
    session.close()
    logger.info("\n✅ Predictions generated successfully")

if __name__ == '__main__':
    generate_predictions_v5()

