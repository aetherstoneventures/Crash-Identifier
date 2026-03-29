"""Alternative data sources for indicators not available in FRED."""

import logging
from typing import Optional
from datetime import datetime

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class AlternativeCollector:
    """Collects data from alternative free sources."""

    def __init__(self):
        """Initialize alternative data collector."""
        self.logger = logging.getLogger(__name__)

    def fetch_shiller_pe(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.Series:
        """
        Fetch Shiller PE (CAPE) ratio from Robert Shiller's data.

        Uses publicly available data from Yale University.

        Args:
            start_date: Start date (not used, returns all available)
            end_date: End date (not used, returns all available)

        Returns:
            Series with Shiller PE values indexed by date
        """
        try:
            # Robert Shiller's CAPE data is available at Yale
            url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"

            # Try to fetch from alternative source (CSV format)
            url_csv = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies-financials/master/data/constituents-financials.csv"

            # For now, return empty series - will be populated from FRED/Yahoo data
            # In production, would parse Shiller's data
            self.logger.info("Shiller PE data not available from free sources")
            return pd.Series(dtype=float)

        except Exception as e:
            self.logger.warning(f"Failed to fetch Shiller PE: {e}")
            return pd.Series(dtype=float)

    def fetch_margin_debt(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.Series:
        """
        Fetch margin debt data from FINRA (FREE Excel download).

        FINRA provides free historical margin debt data via Excel file download.
        Data available from January 1997 to present, updated monthly.

        Source: https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics

        Args:
            start_date: Start date for filtering
            end_date: End date for filtering

        Returns:
            Series with margin debt values (in millions) indexed by date
        """
        try:
            # FINRA provides FREE Excel file with historical margin debt
            # The direct download URL changes monthly, so we try multiple approaches

            self.logger.info("Downloading FREE margin debt data from FINRA...")

            # Try multiple URL patterns (FINRA updates the file monthly)
            urls_to_try = [
                # Pattern 1: Current month format (2025-01)
                "https://www.finra.org/sites/default/files/2025-01/margin-statistics.xlsx",
                # Pattern 2: Previous month format (2024-12)
                "https://www.finra.org/sites/default/files/2024-12/margin-statistics.xlsx",
                # Pattern 3: Generic URL (sometimes works)
                "https://www.finra.org/sites/default/files/margin-statistics.xlsx",
                # Pattern 4: Try scraping the page for the download link
                None  # Placeholder for web scraping approach
            ]

            df = None
            for url in urls_to_try:
                if url is None:
                    # Try scraping the page for the download link
                    try:
                        page_url = "https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics"
                        response = requests.get(page_url, timeout=10)
                        soup = BeautifulSoup(response.content, 'html.parser')

                        # Find download link (look for .xlsx file)
                        for link in soup.find_all('a', href=True):
                            if 'margin-statistics.xlsx' in link['href']:
                                url = link['href']
                                if not url.startswith('http'):
                                    url = 'https://www.finra.org' + url
                                self.logger.info(f"Found download link via scraping: {url}")
                                break

                        if url is None:
                            continue
                    except Exception as e:
                        self.logger.warning(f"Failed to scrape FINRA page: {e}")
                        continue

                try:
                    self.logger.info(f"Trying URL: {url}")
                    df = pd.read_excel(url, sheet_name=0, skiprows=3)
                    self.logger.info(f"Successfully downloaded from: {url}")
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to download from {url}: {e}")
                    continue

            if df is None:
                self.logger.warning("All FINRA download attempts failed")
                return pd.Series(dtype=float)

            # Column names: Date, Debit Balances, Free Credit Cash, Free Credit Securities
            # We want "Debit Balances in Customers' Securities Margin Accounts"
            df.columns = df.columns.str.strip()

            # Find the date column and debit balance column
            date_col = None
            debit_col = None

            for col in df.columns:
                col_lower = str(col).lower()
                if 'date' in col_lower or 'month' in col_lower or 'year' in col_lower:
                    date_col = col
                elif 'debit' in col_lower and ('balance' in col_lower or 'margin' in col_lower):
                    debit_col = col

            if date_col is None or debit_col is None:
                self.logger.warning(f"Could not find required columns. Available: {df.columns.tolist()}")
                return pd.Series(dtype=float)

            # Convert to Series
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            df = df.set_index(date_col)
            margin_debt = pd.to_numeric(df[debit_col], errors='coerce').dropna()

            # Filter by date range if provided
            if start_date:
                margin_debt = margin_debt[margin_debt.index >= pd.to_datetime(start_date)]
            if end_date:
                margin_debt = margin_debt[margin_debt.index <= pd.to_datetime(end_date)]

            self.logger.info(f"Successfully fetched {len(margin_debt)} margin debt records from FINRA (FREE)")
            return margin_debt

        except Exception as e:
            self.logger.warning(f"Failed to fetch FREE FINRA margin debt: {e}")
            self.logger.info("Will use synthetic proxy instead")
            return pd.Series(dtype=float)

    def fetch_put_call_ratio(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.Series:
        """
        Calculate put/call ratio from SPY options data using yfinance (FREE).

        CBOE doesn't offer free API access. Instead, we calculate the put/call ratio
        from SPY (S&P 500 ETF) options volume using yfinance.

        This provides a good proxy for market sentiment:
        - High put/call ratio (>1.0) = bearish sentiment (more puts than calls)
        - Low put/call ratio (<0.7) = bullish sentiment (more calls than puts)

        Note: This is total market put/call ratio across all exchanges, which is
        more comprehensive than CBOE-only data.

        Args:
            start_date: Start date for filtering
            end_date: End date for filtering

        Returns:
            Series with put/call ratio values indexed by date
        """
        try:
            import yfinance as yf
            from datetime import datetime, timedelta

            self.logger.info("Calculating put/call ratio from SPY options (FREE via yfinance)...")

            # Use SPY (S&P 500 ETF) as proxy for market sentiment
            spy = yf.Ticker("SPY")

            # Get options expiration dates with error handling
            try:
                expirations = spy.options
            except AttributeError as e:
                self.logger.warning(f"SPY options attribute error: {e}")
                return pd.Series(dtype=float)
            except Exception as e:
                self.logger.warning(f"Failed to get SPY options expirations: {e}")
                return pd.Series(dtype=float)

            if not expirations or len(expirations) == 0:
                self.logger.warning("No options data available for SPY")
                return pd.Series(dtype=float)

            put_call_ratios = []
            dates = []
            successful_fetches = 0
            max_expirations = min(10, len(expirations))  # Use first 10 or fewer

            # Calculate put/call ratio for recent expirations
            for i, exp_date in enumerate(expirations[:max_expirations]):
                try:
                    self.logger.debug(f"Fetching options chain for {exp_date} ({i+1}/{max_expirations})")
                    opt_chain = spy.option_chain(exp_date)

                    # Validate option chain data
                    if opt_chain is None:
                        self.logger.debug(f"No option chain for {exp_date}")
                        continue

                    if not hasattr(opt_chain, 'puts') or not hasattr(opt_chain, 'calls'):
                        self.logger.debug(f"Invalid option chain structure for {exp_date}")
                        continue

                    # Get put and call volumes with error handling
                    try:
                        put_volume = opt_chain.puts['volume'].fillna(0).sum()
                        call_volume = opt_chain.calls['volume'].fillna(0).sum()
                    except KeyError as e:
                        self.logger.debug(f"Missing volume column for {exp_date}: {e}")
                        continue

                    if call_volume > 0 and put_volume >= 0:
                        pc_ratio = put_volume / call_volume
                        # Sanity check: ratio should be between 0.1 and 10
                        if 0.1 <= pc_ratio <= 10:
                            put_call_ratios.append(pc_ratio)
                            dates.append(pd.to_datetime(exp_date))
                            successful_fetches += 1
                        else:
                            self.logger.debug(f"Put/call ratio {pc_ratio:.2f} out of range for {exp_date}")

                except Exception as e:
                    self.logger.debug(f"Error processing expiration {exp_date}: {e}")
                    continue

            if not put_call_ratios or successful_fetches == 0:
                self.logger.warning(f"Could not calculate put/call ratio from options data (0/{max_expirations} successful)")
                return pd.Series(dtype=float)

            self.logger.info(f"Successfully fetched {successful_fetches}/{max_expirations} option chains")

            # Create Series
            pc_series = pd.Series(put_call_ratios, index=dates)
            pc_series = pc_series.sort_index()

            # Resample to daily and forward-fill
            pc_series = pc_series.resample('D').mean().ffill()

            # Filter by date range if provided
            if start_date:
                pc_series = pc_series[pc_series.index >= pd.to_datetime(start_date)]
            if end_date:
                pc_series = pc_series[pc_series.index <= pd.to_datetime(end_date)]

            self.logger.info(f"Successfully calculated {len(pc_series)} put/call ratio values from SPY options (FREE)")
            return pc_series

        except Exception as e:
            self.logger.warning(f"Failed to calculate put/call ratio from options: {e}")
            self.logger.info("Will use synthetic proxy instead")
            return pd.Series(dtype=float)

    def create_synthetic_indicators(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create synthetic indicators from available data.

        For indicators not available from free sources, create proxies
        from available data.

        Args:
            df: DataFrame with FRED and Yahoo data

        Returns:
            DataFrame with additional synthetic indicators
        """
        df_synthetic = df.copy()

        # Synthetic Shiller PE: Use VIX as proxy for market valuation stress
        if 'vix' in df.columns and 'vix' not in df_synthetic.columns:
            # Inverse relationship: high VIX = low valuation
            df_synthetic['shiller_pe'] = 100 / (df['vix'] + 1)

        # Synthetic margin debt: Use credit spread as proxy
        if 'credit_spread_bbb' in df.columns and 'margin_debt' not in df_synthetic.columns:
            # Margin debt increases when credit spreads are tight
            df_synthetic['margin_debt'] = 100 / (df['credit_spread_bbb'] + 1)

        # Synthetic put/call ratio: Use VIX change as proxy
        if 'vix' in df.columns and 'put_call_ratio' not in df_synthetic.columns:
            vix_change = df['vix'].pct_change()
            # Put/call ratio increases when VIX spikes
            df_synthetic['put_call_ratio'] = 1 + (vix_change * 0.5)

        self.logger.info("Created synthetic indicators from available data")
        return df_synthetic

    def fetch_all_alternative_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch all alternative data sources.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with alternative indicators
        """
        df = pd.DataFrame()

        try:
            shiller_pe = self.fetch_shiller_pe(start_date, end_date)
            if not shiller_pe.empty:
                df['shiller_pe'] = shiller_pe

            margin_debt = self.fetch_margin_debt(start_date, end_date)
            if not margin_debt.empty:
                df['margin_debt'] = margin_debt

            put_call_ratio = self.fetch_put_call_ratio(start_date, end_date)
            if not put_call_ratio.empty:
                df['put_call_ratio'] = put_call_ratio

        except Exception as e:
            self.logger.error(f"Failed to fetch alternative data: {e}")

        return df

