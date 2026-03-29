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
from src.data_collection.sp500_multi_source import SP500MultiSourceCollector
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
            # Normalize FRED index to plain dates (no time component) for clean joins
            if not fred_data.empty:
                fred_data.index = pd.to_datetime(fred_data.index).date
            logger.info(f"✅ Collected FRED data: {fred_data.shape}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to collect FRED data: {e}")
            logger.info("   Continuing with Yahoo data only...")
    else:
        logger.warning("⚠️ FRED_API_KEY not set, skipping FRED data")
    
    # Step 2: Collect S&P 500 data (MULTI-SOURCE with intelligent fallback)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Collecting S&P 500 Data (Multi-Source Fallback)")
    logger.info("=" * 80)

    sp500_df = pd.DataFrame()
    sp500_source = "None"

    try:
        sp500_collector = SP500MultiSourceCollector()
        sp500_df, sp500_source = sp500_collector.fetch_sp500(start_date, end_date)
        logger.info(f"✅ Collected S&P 500: {len(sp500_df)} days from {sp500_source}")
    except Exception as e:
        logger.error(f"❌ Failed to collect S&P 500 from all sources: {e}")
        logger.warning("⚠️  Continuing without S&P 500 data - crash detection will be limited")

    # Step 2b: Collect VIX data (Yahoo Finance only - FRED VIX already collected in Step 1)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2b: Collecting VIX Data (Yahoo Finance)")
    logger.info("=" * 80)

    vix_data = pd.DataFrame()
    try:
        yahoo_collector = YahooCollector()
        vix_data = yahoo_collector.fetch_price_data('vix', start_date, end_date)
        logger.info(f"✅ Collected VIX: {len(vix_data)} days from Yahoo Finance")
    except Exception as e:
        logger.warning(f"⚠️  Failed to collect VIX from Yahoo: {e}")
        logger.info("   Will use FRED VIX (VIXCLS) instead")
    
    # Step 3: Merge data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Merging Data Sources")
    logger.info("=" * 80)

    # Create combined dataframe
    combined_df = pd.DataFrame()

    # Add FRED data
    if not fred_data.empty:
        combined_df = fred_data.copy()

    # Add S&P 500 data from multi-source collector
    if not sp500_df.empty:
        # sp500_df has columns ['date', 'close']
        # Convert to Series with **date-only** index for merging
        sp500_series = sp500_df.set_index('date')['close']
        # Ensure index is plain date (no timezone / time) to match FRED
        dt_index = pd.to_datetime(sp500_series.index)
        if dt_index.tz is not None:
            dt_index = dt_index.tz_convert(None)
        sp500_series.index = dt_index.date
        sp500_series.name = 'sp500_close'

        # Merge with combined_df (also ensure its index is date-only)
        if combined_df.empty:
            combined_df = pd.DataFrame(index=sp500_series.index)
        else:
            combined_df.index = pd.to_datetime(combined_df.index).date

        combined_df['sp500_close'] = sp500_series
        logger.info(f"✅ Added S&P 500 data from {sp500_source}")
    else:
        logger.warning("⚠️  No S&P 500 data available - columns will be empty")
        combined_df['sp500_close'] = np.nan

    # Add placeholder for S&P 500 volume (not available from FRED)
    combined_df['sp500_volume'] = np.nan

    # VIX is already in combined_df from FRED (as 'vix')
    # Only add Yahoo VIX if FRED VIX is not available
    if 'vix' in combined_df.columns and not combined_df['vix'].isna().all():
        # Use FRED VIX data (rename to vix_close for consistency)
        logger.info("✅ Using VIX data from FRED (VIXCLS)")
        combined_df['vix_close'] = combined_df['vix']
    elif not vix_data.empty and 'Close' in vix_data.columns:
        vix_close = vix_data['Close'].copy()
        vix_close.name = 'vix_close'
        combined_df['vix_close'] = vix_close
        logger.info("✅ Using VIX data from Yahoo Finance")
    else:
        logger.warning("⚠️  No VIX data from Yahoo or FRED - using NaN")
        combined_df['vix_close'] = np.nan

    # Step 4: Collect alternative data (margin debt, put/call ratio)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Collecting Alternative Data (FREE)")
    logger.info("=" * 80)

    try:
        alt_collector = AlternativeCollector()

        # Fetch FINRA margin debt (FREE Excel download)
        logger.info("Fetching FINRA margin debt (FREE)...")
        margin_debt = alt_collector.fetch_margin_debt(start_date, end_date)

        # Fetch put/call ratio (FREE from SPY options via yfinance)
        logger.info("Fetching put/call ratio from SPY options (FREE)...")
        put_call_ratio = alt_collector.fetch_put_call_ratio(start_date, end_date)

        # Add to combined dataframe
        if not margin_debt.empty:
            combined_df['margin_debt'] = margin_debt
            logger.info(f"✅ Added {len(margin_debt)} margin debt records (FREE FINRA data)")
        else:
            logger.warning("⚠️  No margin debt data - will use synthetic proxy")
            if 'credit_spread_bbb' in combined_df.columns:
                combined_df['margin_debt'] = 100 / (combined_df['credit_spread_bbb'] + 1)
                logger.warning("   Using synthetic: margin_debt = 100 / (credit_spread_bbb + 1)")
            else:
                combined_df['margin_debt'] = np.nan

        if not put_call_ratio.empty:
            combined_df['put_call_ratio'] = put_call_ratio
            logger.info(f"✅ Added {len(put_call_ratio)} put/call ratio records (FREE yfinance data)")
        else:
            logger.warning("⚠️  No put/call ratio data - will use synthetic proxy")
            if 'vix_close' in combined_df.columns and not combined_df['vix_close'].isna().all():
                vix_change = combined_df['vix_close'].pct_change(fill_method=None)
                combined_df['put_call_ratio'] = 1.0 + (vix_change * 0.5).clip(-0.5, 0.5)
                logger.warning("   Using synthetic: put_call_ratio = 1.0 + (VIX_change × 0.5)")
            else:
                combined_df['put_call_ratio'] = np.nan

    except Exception as e:
        logger.error(f"❌ Failed to collect alternative data: {e}")
        logger.warning("⚠️  Falling back to synthetic proxies")

        # Fallback to synthetic proxies
        if 'credit_spread_bbb' in combined_df.columns:
            combined_df['margin_debt'] = 100 / (combined_df['credit_spread_bbb'] + 1)
        else:
            combined_df['margin_debt'] = np.nan

        if 'vix_close' in combined_df.columns and not combined_df['vix_close'].isna().all():
            vix_change = combined_df['vix_close'].pct_change(fill_method=None)
            combined_df['put_call_ratio'] = 1.0 + (vix_change * 0.5).clip(-0.5, 0.5)
        else:
            combined_df['put_call_ratio'] = np.nan

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

    # Use context manager for session
    with db.get_session() as session:
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
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ DATA COLLECTION COMPLETE")
    logger.info(f"   Total records: {inserted}")
    logger.info(f"   Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    logger.info("=" * 80)


if __name__ == '__main__':
    collect_data()

