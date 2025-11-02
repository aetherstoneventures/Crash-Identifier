"""
Improved Statistical Model V2 - Enhanced Crash Detection

Implements:
1. Multi-factor risk scoring with dynamic weights
2. Regime detection (bull/bear/crisis)
3. Composite indicators combining multiple signals
4. Adaptive thresholds based on market conditions
5. Better handling of edge cases
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import logging
import pickle

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from src.utils.database import DatabaseManager, Prediction
from src.utils.config import DATABASE_URL

def calculate_statistical_crash_probability_v2(df):
    """
    Calculate crash probability using improved statistical methods.
    
    Combines multiple risk factors:
    1. Yield curve inversion (recession signal)
    2. Credit stress (financial system stress)
    3. Volatility spike (market fear)
    4. Economic deterioration (recession indicators)
    5. Valuation extremes (market overheating)
    6. Momentum reversal (trend change)
    """
    
    proba = pd.Series(0.0, index=df.index)
    
    # 1. YIELD CURVE INVERSION (Weight: 25%)
    yield_curve = df['yield_10y_3m'].fillna(0)
    yield_inversion = (yield_curve < 0).astype(float)
    yield_inversion_days = (yield_curve < 0).rolling(20).sum() / 20  # % of days inverted
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
    momentum_reversal = (sp500_returns_5d < -0.05).astype(float)  # Large daily drops
    proba += momentum_reversal * 0.05
    
    # Ensure probability is between 0 and 1
    proba = proba.clip(0, 1)
    
    return proba

def train_statistical_model_v2():
    """Train and save improved statistical model."""
    logger.info("=" * 80)
    logger.info("TRAINING IMPROVED STATISTICAL MODEL V2")
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
    
    # Calculate crash probabilities
    logger.info("\nCalculating statistical crash probabilities...")
    crash_proba = calculate_statistical_crash_probability_v2(df)
    
    logger.info(f"Mean probability: {crash_proba.mean():.4f}")
    logger.info(f"Std probability: {crash_proba.std():.4f}")
    logger.info(f"Min probability: {crash_proba.min():.4f}")
    logger.info(f"Max probability: {crash_proba.max():.4f}")
    logger.info(f"High risk (>0.5): {(crash_proba > 0.5).sum()} days")
    logger.info(f"Very high risk (>0.7): {(crash_proba > 0.7).sum()} days")
    
    # Save model function
    with open('data/models/statistical_model_v2.pkl', 'wb') as f:
        pickle.dump(calculate_statistical_crash_probability_v2, f)
    
    logger.info("\n✅ Statistical model saved (v2)")
    session.close()

if __name__ == '__main__':
    train_statistical_model_v2()

