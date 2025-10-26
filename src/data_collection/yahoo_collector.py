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
        'sp500': '^GSPC',
        'vix': '^VIX'
    }

    def __init__(self):
        """Initialize Yahoo Finance collector."""
        self.logger = logging.getLogger(__name__)

    def fetch_price_data(
        self,
        symbol: str,
        start_date: str = '1970-01-01',
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for symbol.

        Args:
            symbol: Symbol key from SYMBOLS dict
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD' (default: today)

        Returns:
            DataFrame with Open, High, Low, Close, Volume, Adj Close

        Raises:
            ValueError: If symbol not found
            Exception: If download fails
        """
        if symbol not in self.SYMBOLS:
            raise ValueError(f"Unknown symbol: {symbol}")

        ticker = self.SYMBOLS[symbol]

        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                timeout=YAHOO_TIMEOUT
            )
            self.logger.info(f"Fetched {symbol}: {len(data)} days")
            return data
        except Exception as e:
            self.logger.error(f"Failed to fetch {symbol}: {e}")
            raise

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

    def fetch_sp500_and_vix(
        self,
        start_date: str = '1970-01-01',
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch both S&P 500 and VIX data.

        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD' (default: today)

        Returns:
            Dict with 'sp500' and 'vix' DataFrames
        """
        try:
            sp500_data = self.fetch_price_data('sp500', start_date, end_date)
            vix_data = self.fetch_price_data('vix', start_date, end_date)

            return {
                'sp500': sp500_data,
                'vix': vix_data
            }
        except Exception as e:
            self.logger.error(f"Failed to fetch market data: {e}")
            raise

