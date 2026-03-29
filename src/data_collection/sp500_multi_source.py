"""Multi-source S&P 500 data collector with intelligent fallback.

PRIORITY ORDER (updated for maximum history coverage):
1. Yahoo Finance ^GSPC (primary - decades of daily history back to ~1950)
2. FRED SP500 index (secondary - reliable but limited to recent years in practice)
3. Alpha Vantage API (requires API key, 500 calls/day limit)
4. Local CSV cache (last resort)

This ensures we ALWAYS have S&P 500 data for crash detection, while prioritizing
sources that provide the longest continuous history for realistic crash testing.
"""

import os
import logging
from typing import Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from fredapi import Fred
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class SP500MultiSourceCollector:
    """Collects S&P 500 data from multiple sources with intelligent fallback."""
    
    # FRED series for S&P 500
    FRED_SP500_SERIES = 'SP500'  # S&P 500 Index (daily, 1927-present)
    
    # Alpha Vantage settings
    ALPHA_VANTAGE_SYMBOL = 'SPY'  # S&P 500 ETF (more reliable than ^GSPC)
    ALPHA_VANTAGE_URL = 'https://www.alphavantage.co/query'
    
    # Yahoo Finance settings
    YAHOO_SYMBOL = '^GSPC'
    
    # Local cache
    CACHE_DIR = Path('data/cache')
    CACHE_FILE = CACHE_DIR / 'sp500_cache.csv'
    
    def __init__(self):
        """Initialize multi-source collector."""
        self.fred_api_key = os.getenv('FRED_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        
        # Create cache directory
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    def fetch_from_fred(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch S&P 500 from FRED (PRIMARY SOURCE - most reliable).
        
        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
        
        Returns:
            DataFrame with columns ['date', 'close'] or None if failed
        """
        if not self.fred_api_key:
            logger.warning("FRED API key not found")
            return None
        
        try:
            logger.info("Attempting to fetch S&P 500 from FRED (SP500 series)...")
            fred = Fred(api_key=self.fred_api_key)
            
            # Fetch SP500 series (daily S&P 500 index)
            data = fred.get_series(
                self.FRED_SP500_SERIES,
                observation_start=start_date,
                observation_end=end_date
            )
            
            if data is None or data.empty:
                logger.warning("FRED returned empty data")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'date': data.index,
                'close': data.values
            })
            
            # Remove NaN values
            df = df.dropna()
            
            logger.info(f"✅ FRED: Fetched {len(df)} days of S&P 500 data")
            return df
            
        except Exception as e:
            logger.warning(f"FRED fetch failed: {e}")
            return None
    
    def fetch_from_alpha_vantage(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch S&P 500 from Alpha Vantage (SECONDARY SOURCE).
        
        Uses SPY ETF as proxy for S&P 500 (more reliable than ^GSPC).
        
        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
        
        Returns:
            DataFrame with columns ['date', 'close'] or None if failed
        """
        if not self.alpha_vantage_key:
            logger.warning("Alpha Vantage API key not found")
            return None
        
        try:
            logger.info("Attempting to fetch S&P 500 from Alpha Vantage (SPY ETF)...")
            
            # Alpha Vantage TIME_SERIES_DAILY_ADJUSTED endpoint
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': self.ALPHA_VANTAGE_SYMBOL,
                'outputsize': 'full',  # Get all available data
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(self.ALPHA_VANTAGE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Time Series (Daily)' not in data:
                logger.warning(f"Alpha Vantage error: {data.get('Note', data.get('Error Message', 'Unknown error'))}")
                return None
            
            # Parse time series data
            time_series = data['Time Series (Daily)']
            
            df = pd.DataFrame([
                {
                    'date': pd.to_datetime(date),
                    'close': float(values['5. adjusted close'])
                }
                for date, values in time_series.items()
            ])
            
            # Filter by date range
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            df = df.sort_values('date').reset_index(drop=True)
            
            logger.info(f"✅ Alpha Vantage: Fetched {len(df)} days of S&P 500 data (SPY ETF)")
            return df
            
        except Exception as e:
            logger.warning(f"Alpha Vantage fetch failed: {e}")
            return None
    
    def fetch_from_yahoo(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch S&P 500 from Yahoo Finance (^GSPC) using yfinance defaults.

        NOTE: We deliberately **do not** pass a custom requests.Session here,
        because recent versions of yfinance expect to manage their own
        curl_cffi-based session. Passing a plain requests.Session causes
        errors like:
            "Yahoo API requires curl_cffi session not <class 'requests.sessions.Session'>"

        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'

        Returns:
            DataFrame with columns ['date', 'close'] or None if failed
        """
        try:
            logger.info("Attempting to fetch S&P 500 from Yahoo Finance (^GSPC)...")

            # Let yfinance manage its own session for compatibility
            ticker = yf.Ticker(self.YAHOO_SYMBOL)
            data = ticker.history(start=start_date, end=end_date, auto_adjust=True)

            if data is None or data.empty:
                logger.warning("Yahoo Finance returned empty data")
                return None

            # Convert to standard format
            df = pd.DataFrame({
                'date': data.index,
                'close': data['Close'].values,
            })

            df = df.dropna().reset_index(drop=True)

            logger.info(f"✅ Yahoo Finance: Fetched {len(df)} days of S&P 500 data")
            return df

        except Exception as e:
            logger.warning(f"Yahoo Finance fetch failed: {e}")
            return None
    
    def load_from_cache(self) -> Optional[pd.DataFrame]:
        """Load S&P 500 data from local cache (LAST RESORT).
        
        Returns:
            DataFrame with columns ['date', 'close'] or None if cache doesn't exist
        """
        if not self.CACHE_FILE.exists():
            logger.warning("No local cache found")
            return None
        
        try:
            logger.info("Loading S&P 500 from local cache...")
            df = pd.read_csv(self.CACHE_FILE, parse_dates=['date'])
            logger.info(f"✅ Cache: Loaded {len(df)} days of S&P 500 data")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def save_to_cache(self, df: pd.DataFrame):
        """Save S&P 500 data to local cache for future use.
        
        Args:
            df: DataFrame with columns ['date', 'close']
        """
        try:
            df.to_csv(self.CACHE_FILE, index=False)
            logger.info(f"✅ Saved {len(df)} days to cache: {self.CACHE_FILE}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def fetch_sp500(self, start_date: str, end_date: str) -> Tuple[pd.DataFrame, str]:
        """Fetch S&P 500 data with intelligent fallback across multiple sources.
        
        Priority order:
        1. FRED (most reliable, daily data since 1927)
        2. Alpha Vantage (requires API key, 500 calls/day)
        3. Yahoo Finance (often fails due to API changes)
        4. Local cache (last resort, may be outdated)
        
        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
        
        Returns:
            Tuple of (DataFrame with columns ['date', 'close'], source_name)
        
        Raises:
            RuntimeError: If all sources fail
        """
        logger.info("=" * 80)
        logger.info("FETCHING S&P 500 DATA - MULTI-SOURCE FALLBACK")
        logger.info("=" * 80)
        
        # Try each source in priority order
        sources = [
            # Prefer Yahoo for longest continuous history (~1950+) for crash analysis
            ('Yahoo Finance', self.fetch_from_yahoo),
            # FRED remains a strong secondary source but is shorter in practice
            ('FRED', self.fetch_from_fred),
            ('Alpha Vantage', self.fetch_from_alpha_vantage),
            ('Local Cache', lambda s, e: self.load_from_cache()),
        ]
        
        for source_name, fetch_func in sources:
            df = fetch_func(start_date, end_date)
            
            if df is not None and not df.empty:
                logger.info(f"✅ SUCCESS: Using {source_name} as S&P 500 data source")
                
                # Save to cache for future use (except if source was cache itself)
                if source_name != 'Local Cache':
                    self.save_to_cache(df)
                
                return df, source_name
        
        # All sources failed
        logger.error("❌ ALL SOURCES FAILED - Cannot fetch S&P 500 data")
        raise RuntimeError(
            "Failed to fetch S&P 500 data from all sources. "
            "Please check:\n"
            "1. FRED_API_KEY in .env\n"
            "2. ALPHA_VANTAGE_API_KEY in .env\n"
            "3. Internet connection\n"
            "4. Yahoo Finance API status"
        )


if __name__ == '__main__':
    # Test the collector
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    collector = SP500MultiSourceCollector()
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*50)).strftime('%Y-%m-%d')
    
    df, source = collector.fetch_sp500(start_date, end_date)
    
    print(f"\n✅ Fetched {len(df)} days from {source}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nLast 5 rows:")
    print(df.tail())

