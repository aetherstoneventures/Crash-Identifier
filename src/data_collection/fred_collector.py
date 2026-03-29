"""FRED API client for collecting economic indicators."""

import logging
import time
from typing import Dict, Optional
from datetime import datetime, timedelta

import pandas as pd
import fredapi

from src.utils.config import FRED_RATE_LIMIT, FRED_TIMEOUT, MAX_RETRIES, RETRY_BACKOFF_FACTOR

logger = logging.getLogger(__name__)


class FREDCollector:
    """Collects economic indicators from FRED API."""

    INDICATORS = {
        'yield_10y_3m': 'T10Y3M',
        'yield_10y_2y': 'T10Y2Y',
        'yield_10y': 'DGS10',
        'credit_spread_bbb': 'BAMLC0A4CBBB',
        'unemployment_rate': 'UNRATE',
        'real_gdp': 'GDPC1',
        'cpi': 'CPIAUCSL',
        'fed_funds_rate': 'FEDFUNDS',
        'vix': 'VIXCLS',
        'consumer_sentiment': 'UMCSENT',
        'housing_starts': 'HOUST',
        'industrial_production': 'INDPRO',
        'm2_money_supply': 'M2SL',
        'debt_to_gdp': 'GFDEGDQ188S',
        'savings_rate': 'PSAVERT',
        'lei': 'USSLIND'
    }

    def __init__(self, api_key: str):
        """
        Initialize FRED collector.

        Args:
            api_key: FRED API key

        Raises:
            ValueError: If API key is empty
        """
        if not api_key:
            raise ValueError("FRED API key is required")

        self.api_key = api_key
        self.fred = fredapi.Fred(api_key=api_key)
        self.logger = logging.getLogger(__name__)
        self.request_count = 0
        self.last_request_time = 0

    def _rate_limit(self) -> None:
        """Implement rate limiting (120 requests/minute)."""
        self.request_count += 1
        elapsed = time.time() - self.last_request_time

        if self.request_count >= FRED_RATE_LIMIT:
            sleep_time = max(0, 60 - elapsed)
            if sleep_time > 0:
                self.logger.info(f"Rate limit reached. Sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
            self.request_count = 0
            self.last_request_time = time.time()

    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Retry logic with exponential backoff.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If all retries fail
        """
        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit()
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    self.logger.error(f"Failed after {MAX_RETRIES} retries: {e}")
                    raise
                wait_time = RETRY_BACKOFF_FACTOR ** attempt
                self.logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

    def fetch_indicator(
        self,
        indicator_name: str,
        start_date: str = '1970-01-01',
        end_date: Optional[str] = None
    ) -> pd.Series:
        """
        Fetch single indicator from FRED.

        Args:
            indicator_name: Name from INDICATORS dict
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD' (default: today)

        Returns:
            pd.Series with DatetimeIndex

        Raises:
            ValueError: If indicator_name invalid
            Exception: If API call fails
        """
        if indicator_name not in self.INDICATORS:
            raise ValueError(f"Unknown indicator: {indicator_name}")

        series_id = self.INDICATORS[indicator_name]

        try:
            data = self._retry_with_backoff(
                self.fred.get_series,
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )
            self.logger.info(f"Fetched {indicator_name}: {len(data)} observations")
            return data
        except Exception as e:
            self.logger.error(f"Failed to fetch {indicator_name}: {e}")
            raise

    def fetch_all_indicators(
        self,
        start_date: str = '1970-01-01',
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch all 16 indicators.

        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD' (default: today)

        Returns:
            DataFrame with all indicators
        """
        df = pd.DataFrame()
        failed_indicators = []

        for name in self.INDICATORS:
            try:
                series = self.fetch_indicator(name, start_date, end_date)
                df[name] = series
            except Exception as e:
                self.logger.error(f"Skipping {name}: {e}")
                failed_indicators.append(name)
                continue

        if failed_indicators:
            self.logger.warning(f"Failed to fetch: {failed_indicators}")

        self.logger.info(f"Fetched {len(df.columns)} indicators with {len(df)} observations")
        return df

