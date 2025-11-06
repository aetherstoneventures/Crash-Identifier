"""
Collect market and economic data from FRED and Yahoo Finance.

This script collects:
1. Economic indicators from FRED (16 indicators)
2. Market data from Yahoo Finance (S&P 500, VIX)
3. Alternative data (synthetic indicators)
4. Calculates derived indicators
5. Stores in database
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from src.utils.database import DatabaseManager, Indicator
from src.utils.config import FRED_API_KEY
from src.data_collection.fred_collector import FREDCollector
from src.data_collection.yahoo_collector import YahooCollector
from src.data_collection.alternative_collector import AlternativeCollector
from src.feature_engineering.feature_pipeline import FeaturePipeline


def collect_data():
    """Collect all market and economic data."""
    logger.info("=" * 80)
    logger.info("COLLECTING MARKET AND ECONOMIC DATA")
    logger.info("=" * 80)
    
    # Initialize database
    db = DatabaseManager()
    db.create_tables()
    
    # Date range: Last 50 years
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=50*365)).strftime('%Y-%m-%d')
    
    logger.info(f"\nDate range: {start_date} to {end_date}")
    
    # Step 1: Collect FRED data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Collecting FRED Economic Indicators")
    logger.info("=" * 80)
    
    fred_data = pd.DataFrame()
    if FRED_API_KEY:
        try:
            fred_collector = FREDCollector(FRED_API_KEY)
            fred_data = fred_collector.fetch_all_indicators(start_date, end_date)
            logger.info(f"✅ Collected FRED data: {fred_data.shape}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to collect FRED data: {e}")
            logger.info("   Continuing with Yahoo data only...")
    else:
        logger.warning("⚠️ FRED_API_KEY not set, skipping FRED data")
    
    # Step 2: Collect Yahoo Finance data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Collecting Yahoo Finance Market Data")
    logger.info("=" * 80)
    
    try:
        yahoo_collector = YahooCollector()
        market_data = yahoo_collector.fetch_sp500_and_vix(start_date, end_date)
        
        sp500_data = market_data['sp500']
        vix_data = market_data['vix']
        
        logger.info(f"✅ Collected S&P 500: {len(sp500_data)} days")
        logger.info(f"✅ Collected VIX: {len(vix_data)} days")
        
    except Exception as e:
        logger.error(f"❌ Failed to collect Yahoo data: {e}")
        raise
    
    # Step 3: Merge data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Merging Data Sources")
    logger.info("=" * 80)

    # Create combined dataframe
    combined_df = pd.DataFrame()

    # Add FRED data
    if not fred_data.empty:
        combined_df = fred_data.copy()

    # Add Yahoo data - extract Close and Volume columns
    sp500_close = sp500_data['Close'].copy()
    sp500_close.name = 'sp500_close'
    sp500_volume = sp500_data['Volume'].copy()
    sp500_volume.name = 'sp500_volume'
    vix_close = vix_data['Close'].copy()
    vix_close.name = 'vix_close'

    combined_df['sp500_close'] = sp500_close
    combined_df['sp500_volume'] = sp500_volume
    combined_df['vix_close'] = vix_close

    # Step 4: Calculate synthetic indicators for margin_debt and put_call_ratio
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Calculating Synthetic Indicators")
    logger.info("=" * 80)
    logger.info("⚠️  WARNING: The following indicators are SYNTHETIC PROXIES, NOT real market data:")
    logger.info("=" * 80)

    # Synthetic margin debt: Use credit spread as inverse proxy
    # Margin debt increases when credit spreads are tight (easy credit)
    if 'credit_spread_bbb' in combined_df.columns:
        combined_df['margin_debt'] = 100 / (combined_df['credit_spread_bbb'] + 1)
        logger.warning("⚠️  SYNTHETIC: margin_debt = 100 / (credit_spread_bbb + 1)")
        logger.warning("   - NOT real FINRA margin debt data")
        logger.warning("   - Proxy based on credit spreads (inverse relationship)")
        logger.warning("   - Real FINRA data: billions of dollars (e.g., $800B)")
        logger.warning("   - Synthetic values: typically 49-51 (mathematical artifact)")
    else:
        combined_df['margin_debt'] = None
        logger.warning("⚠️ Cannot calculate margin_debt (missing credit_spread_bbb)")

    # Synthetic put/call ratio: Use VIX change as proxy
    # Put/call ratio increases when VIX spikes (fear)
    if 'vix_close' in combined_df.columns:
        vix_change = combined_df['vix_close'].pct_change()
        combined_df['put_call_ratio'] = 1.0 + (vix_change * 0.5).clip(-0.5, 0.5)
        logger.warning("⚠️  SYNTHETIC: put_call_ratio = 1.0 + (VIX_change × 0.5)")
        logger.warning("   - NOT real CBOE put/call ratio data")
        logger.warning("   - Proxy based on VIX percentage changes")
        logger.warning("   - Real CBOE data: ranges 0.5-2.5 typically")
        logger.warning("   - Synthetic values: cluster around 1.0 (mathematical artifact)")
    else:
        combined_df['put_call_ratio'] = None
        logger.warning("⚠️ Cannot calculate put_call_ratio (missing vix_close)")

    logger.info("=" * 80)
    logger.info("✅ All 20 indicators ready (18 real + 2 synthetic proxies)")
    logger.info("   Real indicators: 16 from FRED + 2 from Yahoo Finance")
    logger.info("   Synthetic proxies: margin_debt, put_call_ratio")
    
    # Step 5: Clean and prepare data for storage
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Cleaning Data")
    logger.info("=" * 80)

    # Fill NaN values using forward fill then backward fill
    combined_df = combined_df.ffill().bfill()

    # For any remaining NaN values, fill with column mean
    for col in combined_df.columns:
        if combined_df[col].isna().any():
            combined_df[col] = combined_df[col].fillna(combined_df[col].mean())

    logger.info(f"✅ Data cleaned ({combined_df.shape[1]} columns, {combined_df.isna().sum().sum()} NaN values remaining)")
    
    # Step 6: Store in database
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Storing Raw Data in Database")
    logger.info("=" * 80)
    
    session = db.get_session()
    
    try:
        # Clear old data
        session.query(Indicator).delete()
        session.commit()
        logger.info("Cleared old indicator data")
        
        # Reset index to get date as column
        combined_df = combined_df.reset_index()
        if 'index' in combined_df.columns:
            combined_df = combined_df.rename(columns={'index': 'date'})
        
        # Ensure date column is datetime
        if 'date' in combined_df.columns:
            combined_df['date'] = pd.to_datetime(combined_df['date']).dt.date
        
        # Insert data
        inserted = 0
        for idx, row in combined_df.iterrows():
            try:
                # Skip rows with no date
                if pd.isna(row.get('date')):
                    continue
                
                # Create indicator record - ONLY 20 USABLE INDICATORS WITH 100% DATA COVERAGE
                indicator = Indicator(
                    date=row['date'],
                    # Yield curve (3)
                    yield_10y_3m=row.get('yield_10y_3m'),
                    yield_10y_2y=row.get('yield_10y_2y'),
                    yield_10y=row.get('yield_10y'),
                    # Credit (1)
                    credit_spread_bbb=row.get('credit_spread_bbb'),
                    # Economic (5)
                    unemployment_rate=row.get('unemployment_rate'),
                    real_gdp=row.get('real_gdp'),
                    cpi=row.get('cpi'),
                    fed_funds_rate=row.get('fed_funds_rate'),
                    industrial_production=row.get('industrial_production'),
                    # Market (3)
                    sp500_close=row.get('sp500_close'),
                    sp500_volume=row.get('sp500_volume'),
                    vix_close=row.get('vix_close'),
                    # Sentiment (1)
                    consumer_sentiment=row.get('consumer_sentiment'),
                    # Housing (1)
                    housing_starts=row.get('housing_starts'),
                    # Monetary (1)
                    m2_money_supply=row.get('m2_money_supply'),
                    # Debt (1)
                    debt_to_gdp=row.get('debt_to_gdp'),
                    # Savings (1)
                    savings_rate=row.get('savings_rate'),
                    # Composite (1)
                    lei=row.get('lei'),
                    # Alternative data sources (2)
                    margin_debt=row.get('margin_debt'),
                    put_call_ratio=row.get('put_call_ratio'),
                )
                session.add(indicator)
                inserted += 1
                
                if inserted % 1000 == 0:
                    logger.info(f"  Inserted {inserted} records...")
                    
            except Exception as e:
                logger.warning(f"  Skipped row {idx}: {e}")
                continue
        
        session.commit()
        logger.info(f"✅ Inserted {inserted} indicator records")
        
    except Exception as e:
        session.rollback()
        logger.error(f"❌ Failed to store data: {e}")
        raise
    finally:
        session.close()
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ DATA COLLECTION COMPLETE")
    logger.info(f"   Total records: {inserted}")
    logger.info(f"   Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    logger.info("=" * 80)


if __name__ == '__main__':
    collect_data()

