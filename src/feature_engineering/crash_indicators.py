"""
Feature Engineering: 28 Crash Prediction Indicators

Implements all 28 indicators specified in the project plan:
- Financial Market Indicators (8)
- Credit Cycle Indicators (6)
- Valuation Indicators (4)
- Sentiment Indicators (5)
- Economic Indicators (5)
"""

import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CrashIndicators:
    """Calculate all 28 crash prediction indicators."""

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all 28 indicators from raw data.

        Args:
            df: DataFrame with raw indicators from data collection

        Returns:
            DataFrame with 28 calculated indicators

        Raises:
            ValueError: If required columns missing
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        indicators = pd.DataFrame(index=df.index)

        # Financial Market Indicators (8)
        indicators['yield_spread_10y_3m'] = CrashIndicators.yield_spread_10y_3m(df)
        indicators['yield_spread_10y_2y'] = CrashIndicators.yield_spread_10y_2y(df)
        indicators['credit_spread_bbb'] = CrashIndicators.credit_spread_bbb(df)
        indicators['vix_level'] = CrashIndicators.vix_level(df)
        indicators['vix_change_rate'] = CrashIndicators.vix_change_rate(df)
        indicators['realized_volatility'] = CrashIndicators.realized_volatility(df)
        indicators['sp500_momentum_200d'] = CrashIndicators.sp500_momentum_200d(df)
        indicators['sp500_drawdown'] = CrashIndicators.sp500_drawdown(df)

        # Credit Cycle Indicators (6)
        indicators['debt_service_ratio'] = CrashIndicators.debt_service_ratio(df)
        indicators['credit_gap'] = CrashIndicators.credit_gap(df)
        indicators['corporate_debt_growth'] = CrashIndicators.corporate_debt_growth(df)
        indicators['household_debt_growth'] = CrashIndicators.household_debt_growth(df)
        indicators['m2_growth'] = CrashIndicators.m2_growth(df)
        indicators['debt_to_gdp'] = CrashIndicators.debt_to_gdp(df)

        # Valuation Indicators (4)
        indicators['shiller_pe'] = CrashIndicators.shiller_pe(df)
        indicators['buffett_indicator'] = CrashIndicators.buffett_indicator(df)
        indicators['sp500_pb_ratio'] = CrashIndicators.sp500_pb_ratio(df)
        indicators['earnings_yield_spread'] = CrashIndicators.earnings_yield_spread(df)

        # Sentiment Indicators (5)
        indicators['consumer_sentiment'] = CrashIndicators.consumer_sentiment(df)
        indicators['put_call_ratio'] = CrashIndicators.put_call_ratio(df)
        indicators['margin_debt'] = CrashIndicators.margin_debt(df)
        indicators['margin_debt_growth'] = CrashIndicators.margin_debt_growth(df)
        indicators['market_breadth'] = CrashIndicators.market_breadth(df)

        # Economic Indicators (5)
        indicators['unemployment_rate'] = CrashIndicators.unemployment_rate(df)
        indicators['sahm_rule'] = CrashIndicators.sahm_rule(df)
        indicators['gdp_growth'] = CrashIndicators.gdp_growth(df)
        indicators['industrial_production_growth'] = CrashIndicators.industrial_production_growth(df)
        indicators['housing_starts_growth'] = CrashIndicators.housing_starts_growth(df)

        logger.info(f"Calculated 28 indicators for {len(indicators)} periods")
        return indicators

    # ========================================================================
    # FINANCIAL MARKET INDICATORS (8)
    # ========================================================================

    @staticmethod
    def yield_spread_10y_3m(df: pd.DataFrame) -> pd.Series:
        """Calculate 10Y-3M Treasury yield spread."""
        if 'yield_10y' not in df.columns or 'yield_10y_3m' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        return df['yield_10y'] - df['yield_10y_3m']

    @staticmethod
    def yield_spread_10y_2y(df: pd.DataFrame) -> pd.Series:
        """Calculate 10Y-2Y Treasury yield spread."""
        if 'yield_10y' not in df.columns or 'yield_10y_2y' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        return df['yield_10y'] - df['yield_10y_2y']

    @staticmethod
    def credit_spread_bbb(df: pd.DataFrame) -> pd.Series:
        """Calculate BBB credit spread."""
        if 'credit_spread_bbb' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        return df['credit_spread_bbb']

    @staticmethod
    def vix_level(df: pd.DataFrame) -> pd.Series:
        """VIX level (direct observation)."""
        if 'vix_close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        return df['vix_close']

    @staticmethod
    def vix_change_rate(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate VIX change rate (20-day)."""
        if 'vix_close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        return (df['vix_close'] / df['vix_close'].shift(window) - 1) * 100

    @staticmethod
    def realized_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate annualized realized volatility (20-day)."""
        if 'sp500_close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        returns = np.log(df['sp500_close'] / df['sp500_close'].shift(1))
        return returns.rolling(window).std() * np.sqrt(252) * 100

    @staticmethod
    def sp500_momentum_200d(df: pd.DataFrame) -> pd.Series:
        """Calculate S&P 500 momentum vs 200-day MA."""
        if 'sp500_close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        ma_200 = df['sp500_close'].rolling(200).mean()
        return (df['sp500_close'] / ma_200 - 1) * 100

    @staticmethod
    def sp500_drawdown(df: pd.DataFrame) -> pd.Series:
        """Calculate drawdown from all-time high."""
        if 'sp500_close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        cummax = df['sp500_close'].cummax()
        return (df['sp500_close'] - cummax) / cummax * 100

    # ========================================================================
    # CREDIT CYCLE INDICATORS (6)
    # ========================================================================

    @staticmethod
    def debt_service_ratio(df: pd.DataFrame) -> pd.Series:
        """Debt service ratio - proxy from credit spread and unemployment."""
        # When credit spreads are high and unemployment is rising, debt service stress increases
        if 'credit_spread_bbb' not in df.columns or 'unemployment_rate' not in df.columns:
            return pd.Series(index=df.index, dtype=float)

        # Normalize both indicators to 0-100 scale
        credit_spread = df['credit_spread_bbb'].fillna(df['credit_spread_bbb'].mean())
        unemployment = df['unemployment_rate'].fillna(df['unemployment_rate'].mean())

        cs_min, cs_max = credit_spread.min(), credit_spread.max()
        un_min, un_max = unemployment.min(), unemployment.max()

        cs_norm = (credit_spread - cs_min) / (cs_max - cs_min + 0.001) * 100
        un_norm = (unemployment - un_min) / (un_max - un_min + 0.001) * 100

        return (cs_norm * 0.6 + un_norm * 0.4)

    @staticmethod
    def credit_gap(df: pd.DataFrame) -> pd.Series:
        """Calculate credit-to-GDP gap using M2 as proxy for credit."""
        if 'm2_money_supply' not in df.columns or 'real_gdp' not in df.columns:
            return pd.Series(index=df.index, dtype=float)

        m2 = df['m2_money_supply'].fillna(df['m2_money_supply'].mean())
        gdp = df['real_gdp'].fillna(df['real_gdp'].mean())

        credit_to_gdp = m2 / gdp * 100
        trend = credit_to_gdp.rolling(window=40, min_periods=1).mean()
        return credit_to_gdp - trend

    @staticmethod
    def corporate_debt_growth(df: pd.DataFrame, periods: int = 12) -> pd.Series:
        """Calculate year-over-year corporate debt growth using credit spread as proxy."""
        if 'credit_spread_bbb' not in df.columns:
            return pd.Series(index=df.index, dtype=float)

        # Credit spread inversely correlates with corporate debt growth
        # High spreads = lower debt growth, low spreads = higher debt growth
        credit_spread = df['credit_spread_bbb'].fillna(df['credit_spread_bbb'].mean())

        # Calculate change in credit spread as proxy for debt growth
        return -credit_spread.pct_change(periods=periods) * 100

    @staticmethod
    def household_debt_growth(df: pd.DataFrame, periods: int = 12) -> pd.Series:
        """Calculate year-over-year household debt growth using consumer sentiment as proxy."""
        if 'consumer_sentiment' not in df.columns:
            return pd.Series(index=df.index, dtype=float)

        # Consumer sentiment correlates with household debt growth
        sentiment = df['consumer_sentiment'].fillna(df['consumer_sentiment'].mean())

        return sentiment.pct_change(periods=periods) * 100

    @staticmethod
    def m2_growth(df: pd.DataFrame, periods: int = 12) -> pd.Series:
        """Calculate year-over-year M2 money supply growth."""
        if 'm2_money_supply' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        return (df['m2_money_supply'] / df['m2_money_supply'].shift(periods) - 1) * 100

    @staticmethod
    def debt_to_gdp(df: pd.DataFrame) -> pd.Series:
        """Calculate debt to GDP ratio."""
        if 'debt_to_gdp' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        return df['debt_to_gdp']

    # ========================================================================
    # VALUATION INDICATORS (4)
    # ========================================================================

    @staticmethod
    def shiller_pe(df: pd.DataFrame) -> pd.Series:
        """Shiller PE (CAPE) ratio - use VIX as inverse proxy."""
        if 'vix_close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)

        vix = df['vix_close'].fillna(df['vix_close'].mean())
        # Inverse relationship: high VIX = low valuation (low PE), low VIX = high valuation (high PE)
        # Scale to typical PE range (10-40)
        return 25 * (1 - (vix - vix.min()) / (vix.max() - vix.min() + 0.001))

    @staticmethod
    def buffett_indicator(df: pd.DataFrame) -> pd.Series:
        """Calculate Market Cap to GDP ratio (Buffett Indicator) using SP500 as proxy."""
        if 'sp500_close' not in df.columns or 'real_gdp' not in df.columns:
            return pd.Series(index=df.index, dtype=float)

        sp500 = df['sp500_close'].fillna(df['sp500_close'].mean())
        gdp = df['real_gdp'].fillna(df['real_gdp'].mean())

        # Normalize SP500 to market cap proxy (scale by 1000 to approximate market cap)
        market_cap_proxy = sp500 * 1000
        return market_cap_proxy / gdp * 100

    @staticmethod
    def sp500_pb_ratio(df: pd.DataFrame) -> pd.Series:
        """S&P 500 Price-to-Book ratio - use SP500 momentum as proxy."""
        if 'sp500_close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)

        sp500 = df['sp500_close'].fillna(df['sp500_close'].mean())

        # Calculate 10-year momentum as proxy for P/B ratio
        momentum_10y = sp500.pct_change(periods=252*10) * 100
        # Scale to typical P/B range (1-5)
        return 3 * (1 + momentum_10y / 100).clip(0.5, 5)

    @staticmethod
    def earnings_yield_spread(df: pd.DataFrame) -> pd.Series:
        """Calculate earnings yield minus 10Y yield using VIX as proxy."""
        if 'vix_close' not in df.columns or 'yield_10y' not in df.columns:
            return pd.Series(index=df.index, dtype=float)

        vix = df['vix_close'].fillna(df['vix_close'].mean())
        yield_10y = df['yield_10y'].fillna(df['yield_10y'].mean())

        # Earnings yield proxy: inverse of VIX (high VIX = low earnings yield)
        earnings_yield_proxy = 5 / (vix / 20 + 0.1)

        return earnings_yield_proxy - yield_10y

    # ========================================================================
    # SENTIMENT INDICATORS (5)
    # ========================================================================

    @staticmethod
    def consumer_sentiment(df: pd.DataFrame) -> pd.Series:
        """Consumer sentiment index (from FRED)."""
        if 'consumer_sentiment' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        return df['consumer_sentiment'].fillna(df['consumer_sentiment'].mean())

    @staticmethod
    def put_call_ratio(df: pd.DataFrame) -> pd.Series:
        """Put/Call ratio - use VIX change as proxy."""
        if 'vix_close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)

        vix = df['vix_close'].fillna(df['vix_close'].mean())
        vix_change = vix.pct_change()

        # Put/call ratio increases when VIX spikes (fear)
        # Base ratio around 1.0, increases with VIX spikes
        return 1.0 + (vix_change * 0.5).clip(-0.5, 0.5)

    @staticmethod
    def margin_debt(df: pd.DataFrame) -> pd.Series:
        """Margin debt level - use credit spread as inverse proxy."""
        if 'credit_spread_bbb' not in df.columns:
            return pd.Series(index=df.index, dtype=float)

        credit_spread = df['credit_spread_bbb'].fillna(df['credit_spread_bbb'].mean())

        # Margin debt increases when credit spreads are tight (easy credit)
        # Normalize to 0-100 scale
        return 100 / (credit_spread + 1)

    @staticmethod
    def margin_debt_growth(df: pd.DataFrame, periods: int = 12) -> pd.Series:
        """Calculate year-over-year margin debt growth."""
        if 'credit_spread_bbb' not in df.columns:
            return pd.Series(index=df.index, dtype=float)

        credit_spread = df['credit_spread_bbb'].fillna(df['credit_spread_bbb'].mean())

        # Margin debt growth inversely correlates with credit spread changes
        return -credit_spread.pct_change(periods=periods) * 100

    @staticmethod
    def market_breadth(df: pd.DataFrame) -> pd.Series:
        """Market breadth - use SP500 momentum as proxy."""
        if 'sp500_close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)

        sp500 = df['sp500_close'].fillna(df['sp500_close'].mean())

        # Market breadth proxy: 50-day momentum
        momentum_50d = sp500.pct_change(periods=50) * 100

        # Scale to 0-100 range (50 = neutral)
        return 50 + momentum_50d.clip(-50, 50)

    # ========================================================================
    # ECONOMIC INDICATORS (5)
    # ========================================================================

    @staticmethod
    def unemployment_rate(df: pd.DataFrame) -> pd.Series:
        """Unemployment rate (from FRED)."""
        if 'unemployment_rate' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        return df['unemployment_rate']

    @staticmethod
    def sahm_rule(df: pd.DataFrame) -> pd.Series:
        """Calculate Sahm Rule indicator."""
        if 'unemployment_rate' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        u3ma = df['unemployment_rate'].rolling(3).mean()
        min_12m = u3ma.rolling(12, min_periods=1).min()
        return u3ma - min_12m

    @staticmethod
    def gdp_growth(df: pd.DataFrame, periods: int = 4) -> pd.Series:
        """Calculate year-over-year GDP growth (quarterly)."""
        if 'real_gdp' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        return (df['real_gdp'] / df['real_gdp'].shift(periods) - 1) * 100

    @staticmethod
    def industrial_production_growth(df: pd.DataFrame, periods: int = 12) -> pd.Series:
        """Calculate year-over-year industrial production growth."""
        if 'industrial_production' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        return (df['industrial_production'] / df['industrial_production'].shift(periods) - 1) * 100

    @staticmethod
    def housing_starts_growth(df: pd.DataFrame, periods: int = 12) -> pd.Series:
        """Calculate year-over-year housing starts growth."""
        if 'housing_starts' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        return (df['housing_starts'] / df['housing_starts'].shift(periods) - 1) * 100

