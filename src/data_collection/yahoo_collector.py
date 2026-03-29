"""Yahoo Finance client for collecting market data."""

import logging
from typing import Dict, Optional

import pandas as pd
import numpy as np
import yfinance as yf

from src.utils.config import YAHOO_TIMEOUT

logger = logging.getLogger(__name__)


class YahooCollector:
    """Collects market data from Yahoo Finance."""

    SYMBOLS = {
        'sp500': '^GSPC'
        # VIX removed - use FRED 'VIXCLS' instead (more reliable)
    }

    def __init__(self):
        """Initialize Yahoo Finance collector."""
        self.logger = logging.getLogger(__name__)

    def fetch_price_data(
        self,
        symbol: str,
        start_date: str = '1970-01-01',
        end_date: Optional[str] = None,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for symbol with retry logic.

        Args:
            symbol: Symbol key from SYMBOLS dict
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD' (default: today)
            max_retries: Maximum number of retry attempts

        Returns:
            DataFrame with Open, High, Low, Close, Volume, Adj Close

        Raises:
            ValueError: If symbol not found
            Exception: If download fails after all retries
        """
        if symbol not in self.SYMBOLS:
            raise ValueError(f"Unknown symbol: {symbol}")

        ticker = self.SYMBOLS[symbol]

        # Try multiple methods to fetch data
        for attempt in range(max_retries):
            try:
                # Method 1: Use yf.download (most reliable)
                if attempt == 0:
                    data = yf.download(
                        ticker,
                        start=start_date,
                        end=end_date,
                        progress=False,
                        timeout=YAHOO_TIMEOUT
                    )
                # Method 2: Use Ticker object
                else:
                    self.logger.info(f"Retry {attempt}/{max_retries} using Ticker API...")
                    ticker_obj = yf.Ticker(ticker)
                    data = ticker_obj.history(
                        start=start_date,
                        end=end_date,
                        timeout=YAHOO_TIMEOUT
                    )

                if data is not None and not data.empty:
                    self.logger.info(f"Fetched {symbol}: {len(data)} days")
                    return data
                else:
                    self.logger.warning(f"Empty data for {symbol}, attempt {attempt + 1}/{max_retries}")

            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {symbol}: {e}")
                if attempt == max_retries - 1:
                    self.logger.error(f"All retries failed for {symbol}")
                    # Return empty DataFrame instead of raising
                    return pd.DataFrame()
                import time
                time.sleep(2)  # Wait before retry

        # If all retries failed, return empty DataFrame
        self.logger.error(f"Failed to fetch {symbol} after {max_retries} attempts")
        return pd.DataFrame()

    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate log returns.

        Args:
            prices: Price series

        Returns:
            Log returns series
        """
        return np.log(prices / prices.shift(1)).dropna()

    def calculate_volatility(
        self,
        returns: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Calculate annualized rolling volatility.

        Args:
            returns: Returns series
            window: Rolling window size

        Returns:
            Annualized volatility series
        """
        return returns.rolling(window).std() * np.sqrt(252) * 100

    def calculate_drawdown(self, prices: pd.Series) -> pd.Series:
        """
        Calculate drawdown from all-time high.

        Args:
            prices: Price series

        Returns:
            Drawdown series (negative values)
        """
        cummax = prices.cummax()
        return (prices - cummax) / cummax * 100

    def fetch_sp500(
        self,
        start_date: str = '1970-01-01',
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch S&P 500 data.

        Note: VIX is now fetched from FRED (VIXCLS) for better reliability.

        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD' (default: today)

        Returns:
            DataFrame with S&P 500 OHLCV data
        """
        try:
            sp500_data = self.fetch_price_data('sp500', start_date, end_date)
            return sp500_data
        except Exception as e:
            self.logger.error(f"Failed to fetch S&P 500 data: {e}")
            raise

    def fetch_sp500_and_vix(
        self,
        start_date: str = '1970-01-01',
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        DEPRECATED: Use fetch_sp500() instead. VIX now comes from FRED.

        Kept for backward compatibility.

        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD' (default: today)

        Returns:
            Dict with 'sp500' DataFrame only
        """
        self.logger.warning("fetch_sp500_and_vix() is deprecated. Use fetch_sp500() instead. VIX now from FRED.")
        try:
            sp500_data = self.fetch_price_data('sp500', start_date, end_date)
            return {
                'sp500': sp500_data,
                'vix': pd.DataFrame()  # Empty - use FRED instead
            }
        except Exception as e:
            self.logger.error(f"Failed to fetch market data: {e}")
            raise

