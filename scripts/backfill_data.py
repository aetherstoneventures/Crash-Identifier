"""Script to backfill historical data from 1970-2025."""

import sys
import os
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_collection.fred_collector import FREDCollector
from src.data_collection.yahoo_collector import YahooCollector
from src.data_collection.alternative_collector import AlternativeCollector
from src.utils.database import DatabaseManager, Indicator
from src.utils.validators import DataValidator
from src.utils.config import FRED_API_KEY
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__, log_file='data/logs/backfill.log')


def backfill_historical_data() -> None:
    """Backfill 55 years of historical data (1970-2025)."""
    logger.info("=" * 80)
    logger.info("Starting historical data backfill (1970-2025)")
    logger.info("=" * 80)

    try:
        # Initialize components
        db_manager = DatabaseManager()
        db_manager.create_tables()
        validator = DataValidator()

        # Initialize collectors
        if not FRED_API_KEY:
            logger.error("FRED_API_KEY not set. Please set it in .env file")
            return

        fred_collector = FREDCollector(FRED_API_KEY)
        yahoo_collector = YahooCollector()
        alternative_collector = AlternativeCollector()

        # Fetch data
        logger.info("Fetching FRED data (1970-2025)...")
        fred_data = fred_collector.fetch_all_indicators('1970-01-01', '2025-12-31')
        logger.info(f"FRED data shape: {fred_data.shape}")

        logger.info("Fetching Yahoo Finance data (1970-2025)...")
        sp500_data = yahoo_collector.fetch_price_data('sp500', '1970-01-01', '2025-12-31')
        vix_data = yahoo_collector.fetch_price_data('vix', '1990-01-01', '2025-12-31')
        logger.info(f"S&P 500 data shape: {sp500_data.shape}")
        logger.info(f"VIX data shape: {vix_data.shape}")

        logger.info("Fetching alternative data...")
        alternative_data = alternative_collector.fetch_all_alternative_data('1970-01-01', '2025-12-31')
        logger.info(f"Alternative data shape: {alternative_data.shape}")

        # Merge data
        logger.info("Merging data sources...")
        merged_data = fred_data.copy()

        # Add S&P 500 data
        merged_data['sp500_close'] = sp500_data['Close']
        merged_data['sp500_volume'] = sp500_data['Volume']

        # Add VIX data
        merged_data['vix_close'] = vix_data['Close']

        # Add alternative data
        if not alternative_data.empty:
            merged_data = merged_data.join(alternative_data, how='left')

        logger.info(f"Merged data shape: {merged_data.shape}")

        # Validate
        logger.info("Validating data...")
        validation = validator.validate_dataframe(merged_data)
        logger.info(f"Validation results:")
        logger.info(f"  - Quality score: {validation['quality_score']:.2%}")
        logger.info(f"  - Missing percentage: {validation['missing_values']['missing_percentage']:.2%}")
        logger.info(f"  - Date range: {validation['date_range']}")

        # Fill missing values
        logger.info("Filling missing values...")
        merged_data = validator.fill_missing_values(merged_data)

        # Store in database
        logger.info("Storing data in database...")
        session = db_manager.get_session()

        stored_count = 0
        for date, row in merged_data.iterrows():
            try:
                indicator = Indicator(
                    date=pd.Timestamp(date).date(),
                    yield_10y_3m=row.get('yield_10y_3m'),
                    yield_10y_2y=row.get('yield_10y_2y'),
                    yield_10y=row.get('yield_10y'),
                    credit_spread_bbb=row.get('credit_spread_bbb'),
                    unemployment_rate=row.get('unemployment_rate'),
                    real_gdp=row.get('real_gdp'),
                    cpi=row.get('cpi'),
                    fed_funds_rate=row.get('fed_funds_rate'),
                    industrial_production=row.get('industrial_production'),
                    sp500_close=row.get('sp500_close'),
                    sp500_volume=row.get('sp500_volume'),
                    vix_close=row.get('vix_close'),
                    consumer_sentiment=row.get('consumer_sentiment'),
                    housing_starts=row.get('housing_starts'),
                    m2_money_supply=row.get('m2_money_supply'),
                    debt_to_gdp=row.get('debt_to_gdp'),
                    savings_rate=row.get('savings_rate'),
                    lei=row.get('lei'),
                    shiller_pe=row.get('shiller_pe'),
                    margin_debt=row.get('margin_debt'),
                    put_call_ratio=row.get('put_call_ratio'),
                    data_quality_score=validator.calculate_data_quality_score(
                        pd.DataFrame([row])
                    )
                )
                db_manager.add_indicator(session, indicator)
                stored_count += 1

                if stored_count % 1000 == 0:
                    logger.info(f"Stored {stored_count} records...")

            except Exception as e:
                logger.warning(f"Failed to store record for {date}: {e}")
                continue

        session.close()
        logger.info(f"Backfill complete. Total records stored: {stored_count}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Backfill failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import pandas as pd
    backfill_historical_data()

