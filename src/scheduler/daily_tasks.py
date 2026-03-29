"""Daily pipeline orchestration for data collection and processing."""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.data_collection.fred_collector import FREDCollector
from src.data_collection.yahoo_collector import YahooCollector
from src.data_collection.alternative_collector import AlternativeCollector
from src.utils.database import DatabaseManager, Indicator
from src.utils.validators import DataValidator
from src.utils.config import SCHEDULER_HOUR, SCHEDULER_MINUTE, FRED_API_KEY, LOG_FILE

logger = logging.getLogger(__name__)


class DailyPipeline:
    """Orchestrates daily data collection and processing."""

    def __init__(self):
        """Initialize daily pipeline."""
        self.scheduler = BlockingScheduler()
        self.logger = logging.getLogger(__name__)
        self.db_manager = DatabaseManager()
        self.validator = DataValidator()
        self.fred_data = None
        self.yahoo_data = None
        self.alternative_data = None

    def run_daily_update(self) -> None:
        """Execute full daily update pipeline."""
        try:
            self.logger.info(f"Starting daily update at {datetime.now()}")

            # Step 1: Collect data
            self.collect_fred_data()
            self.collect_yahoo_data()
            self.collect_alternative_data()

            # Step 2: Merge data
            merged_data = self.merge_data()

            # Step 3: Validate
            self.validate_data(merged_data)

            # Step 4: Store
            self.store_data(merged_data)

            self.logger.info(f"Daily update completed successfully at {datetime.now()}")

        except Exception as e:
            self.logger.error(f"Daily update failed: {e}", exc_info=True)

    def collect_fred_data(self) -> None:
        """Collect latest FRED indicators."""
        try:
            if not FRED_API_KEY:
                self.logger.warning("FRED_API_KEY not set, skipping FRED data collection")
                self.fred_data = pd.DataFrame()
                return

            collector = FREDCollector(FRED_API_KEY)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

            self.fred_data = collector.fetch_all_indicators(start_date, end_date)
            self.logger.info(f"Collected FRED data: {self.fred_data.shape}")

        except Exception as e:
            self.logger.error(f"Failed to collect FRED data: {e}")
            self.fred_data = pd.DataFrame()

    def collect_yahoo_data(self) -> None:
        """Collect latest Yahoo Finance data."""
        try:
            collector = YahooCollector()
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

            sp500_data = collector.fetch_price_data('sp500', start_date, end_date)
            vix_data = collector.fetch_price_data('vix', start_date, end_date)

            self.yahoo_data = {
                'sp500': sp500_data,
                'vix': vix_data
            }
            self.logger.info("Collected Yahoo Finance data")

        except Exception as e:
            self.logger.error(f"Failed to collect Yahoo data: {e}")
            self.yahoo_data = {}

    def collect_alternative_data(self) -> None:
        """Collect alternative data sources."""
        try:
            collector = AlternativeCollector()
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

            self.alternative_data = collector.fetch_all_alternative_data(start_date, end_date)
            self.logger.info("Collected alternative data")

        except Exception as e:
            self.logger.error(f"Failed to collect alternative data: {e}")
            self.alternative_data = pd.DataFrame()

    def merge_data(self) -> pd.DataFrame:
        """Merge all data sources."""
        try:
            # Start with FRED data
            merged = self.fred_data.copy() if not self.fred_data.empty else pd.DataFrame()

            # Add Yahoo data
            if self.yahoo_data:
                if 'sp500' in self.yahoo_data and not self.yahoo_data['sp500'].empty:
                    sp500 = self.yahoo_data['sp500']
                    merged['sp500_close'] = sp500['Close']
                    merged['sp500_volume'] = sp500['Volume']

                if 'vix' in self.yahoo_data and not self.yahoo_data['vix'].empty:
                    vix = self.yahoo_data['vix']
                    merged['vix_close'] = vix['Close']

            # Add alternative data
            if not self.alternative_data.empty:
                merged = merged.join(self.alternative_data, how='left')

            self.logger.info(f"Merged data shape: {merged.shape}")
            return merged

        except Exception as e:
            self.logger.error(f"Failed to merge data: {e}")
            return pd.DataFrame()

    def validate_data(self, df: pd.DataFrame) -> None:
        """Validate collected data."""
        try:
            validation = self.validator.validate_dataframe(df)
            self.logger.info(f"Data validation results: {validation}")

            if validation['missing_values']['critical_missing']:
                self.logger.warning(
                    f"Critical missing columns: {validation['missing_values']['critical_missing']}"
                )

        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")

    def store_data(self, df: pd.DataFrame) -> None:
        """Store data in database."""
        try:
            if df.empty:
                self.logger.warning("No data to store")
                return

            session = self.db_manager.get_session()

            for date, row in df.iterrows():
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
                    data_quality_score=self.validator.calculate_data_quality_score(
                        pd.DataFrame([row])
                    )
                )
                self.db_manager.add_indicator(session, indicator)

            session.close()
            self.logger.info(f"Stored {len(df)} records in database")

        except Exception as e:
            self.logger.error(f"Failed to store data: {e}")

    def start_scheduler(self) -> None:
        """Start scheduler (6 AM daily by default)."""
        try:
            self.scheduler.add_job(
                self.run_daily_update,
                CronTrigger(hour=SCHEDULER_HOUR, minute=SCHEDULER_MINUTE),
                id='daily_update',
                name='Daily data collection and processing',
                replace_existing=True
            )
            self.logger.info(
                f"Scheduler started. Updates at {SCHEDULER_HOUR:02d}:{SCHEDULER_MINUTE:02d} daily."
            )
            self.scheduler.start()

        except Exception as e:
            self.logger.error(f"Failed to start scheduler: {e}")
            raise

    def run_once(self) -> None:
        """Run pipeline once (for testing/backfill)."""
        self.run_daily_update()

