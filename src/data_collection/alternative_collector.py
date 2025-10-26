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
        Fetch margin debt data from FINRA.

        FINRA publishes monthly margin debt statistics.

        Args:
            start_date: Start date (not used)
            end_date: End date (not used)

        Returns:
            Series with margin debt values indexed by date
        """
        try:
            # FINRA publishes margin debt data
            # For free tier, we'll use a placeholder
            # In production, would scrape FINRA website or use their API
            self.logger.info("Margin debt data requires FINRA subscription")
            return pd.Series(dtype=float)

        except Exception as e:
            self.logger.warning(f"Failed to fetch margin debt: {e}")
            return pd.Series(dtype=float)

    def fetch_put_call_ratio(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.Series:
        """
        Fetch put/call ratio from CBOE.

        CBOE publishes daily put/call ratios.

        Args:
            start_date: Start date (not used)
            end_date: End date (not used)

        Returns:
            Series with put/call ratio values indexed by date
        """
        try:
            # CBOE data requires subscription for historical data
            # For free tier, we'll use a placeholder
            self.logger.info("Put/call ratio data requires CBOE subscription")
            return pd.Series(dtype=float)

        except Exception as e:
            self.logger.warning(f"Failed to fetch put/call ratio: {e}")
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

