"""Feature Engineering: Market Crash Prediction Indicators.

ALL indicators are derived from REAL economic data sources:
- 16 indicators from FRED (Federal Reserve Economic Data)
- 2 indicators from Yahoo Finance (S&P 500, VIX)
- 2 optional real-data indicators (FINRA margin debt, options put/call ratio)

Design principles:
1. NO synthetic proxies or fabricated indicators.
2. Every calculation uses ONLY past and present data (no look-ahead).
3. All rolling windows use min_periods to avoid NaN-fill bias.
4. Period windows are calibrated for daily-frequency data.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Trading day constants for daily-frequency data
TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_QUARTER = 63
TRADING_DAYS_PER_YEAR = 252


class CrashIndicators:
    """Calculate crash prediction indicators from raw economic data.

    All indicators are either:
    - Direct: Raw data passed through unchanged
    - Derived: Standard financial/economic calculations with NO look-ahead

    The FRED series T10Y3M and T10Y2Y are already yield SPREADS
    (10Y minus 3M, 10Y minus 2Y). They are used directly — never
    subtracted from the 10Y yield.
    """

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators from raw data.

        Args:
            df: DataFrame indexed by date with raw indicator columns.
                Expected columns: sp500_close, vix_close, yield_10y,
                yield_10y_3m (=T10Y3M spread), yield_10y_2y (=T10Y2Y spread),
                credit_spread_bbb, unemployment_rate, real_gdp, cpi,
                fed_funds_rate, industrial_production, consumer_sentiment,
                housing_starts, m2_money_supply, debt_to_gdp, savings_rate,
                lei, margin_debt (optional), put_call_ratio (optional)

        Returns:
            DataFrame with calculated indicators. Same index as input.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        indicators = pd.DataFrame(index=df.index)

        # === YIELD CURVE (3) =============================================
        # T10Y3M IS the 10Y-3M spread. Use directly.
        indicators['yield_spread_10y_3m'] = _safe_col(df, 'yield_10y_3m')
        # T10Y2Y IS the 10Y-2Y spread. Use directly.
        indicators['yield_spread_10y_2y'] = _safe_col(df, 'yield_10y_2y')
        # 10-Year Treasury yield level
        indicators['yield_10y'] = _safe_col(df, 'yield_10y')

        # === CREDIT (1) ==================================================
        indicators['credit_spread_bbb'] = _safe_col(df, 'credit_spread_bbb')

        # === VOLATILITY (4) ==============================================
        indicators['vix_level'] = _safe_col(df, 'vix_close')
        indicators['vix_change_20d'] = _pct_change(df, 'vix_close', 20)
        indicators['realized_volatility_20d'] = _realized_vol(
            df, 'sp500_close', 20
        )
        indicators['sp500_volatility_20d'] = _rolling_return_std(
            df, 'sp500_close', 20
        )

        # === MARKET MOMENTUM (5) =========================================
        indicators['sp500_momentum_200d'] = _momentum(
            df, 'sp500_close', 200
        )
        indicators['sp500_drawdown'] = _drawdown(df, 'sp500_close')
        indicators['sp500_return_5d'] = _pct_change(df, 'sp500_close', 5)
        indicators['sp500_return_20d'] = _pct_change(df, 'sp500_close', 20)
        indicators['sp500_return_60d'] = _pct_change(df, 'sp500_close', 60)

        # === ECONOMIC (7) ================================================
        indicators['unemployment_rate'] = _safe_col(df, 'unemployment_rate')
        indicators['sahm_rule'] = _sahm_rule(df, 'unemployment_rate')
        indicators['gdp_growth_yoy'] = _yoy_growth(df, 'real_gdp')
        indicators['industrial_prod_growth_yoy'] = _yoy_growth(
            df, 'industrial_production'
        )
        indicators['housing_starts_growth_yoy'] = _yoy_growth(
            df, 'housing_starts'
        )
        indicators['m2_growth_yoy'] = _yoy_growth(df, 'm2_money_supply')
        indicators['cpi_yoy'] = _yoy_growth(df, 'cpi')

        # === POLICY & COMPOSITE (3) ======================================
        indicators['fed_funds_rate'] = _safe_col(df, 'fed_funds_rate')
        indicators['debt_to_gdp'] = _safe_col(df, 'debt_to_gdp')
        indicators['lei'] = _safe_col(df, 'lei')

        # === SENTIMENT (2 guaranteed + 2 optional) =======================
        indicators['consumer_sentiment'] = _safe_col(
            df, 'consumer_sentiment'
        )
        indicators['savings_rate'] = _safe_col(df, 'savings_rate')

        # Real margin debt from FINRA (if available in database)
        if 'margin_debt' in df.columns and df['margin_debt'].notna().any():
            indicators['margin_debt'] = df['margin_debt']

        # Real put/call ratio from options data (if available in database)
        if 'put_call_ratio' in df.columns and df['put_call_ratio'].notna().any():
            indicators['put_call_ratio'] = df['put_call_ratio']

        # === RATE-OF-CHANGE SIGNALS (2) ==================================
        indicators['yield_curve_change_20d'] = (
            indicators['yield_spread_10y_3m'].diff(20)
        )
        indicators['credit_spread_change_20d'] = (
            indicators['credit_spread_bbb'].diff(20)
        )

        n_indicators = indicators.shape[1]
        logger.info(
            f"Calculated {n_indicators} real indicators "
            f"for {len(indicators)} periods"
        )
        return indicators


# =========================================================================
# Helper functions — all past-only, no look-ahead
# =========================================================================

def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return column if it exists, else a NaN series."""
    if col in df.columns:
        return df[col].copy()
    logger.warning(f"Column '{col}' not found in data. Returning NaN.")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _pct_change(df: pd.DataFrame, col: str, periods: int) -> pd.Series:
    """Percentage change over N periods. Uses shift() — no look-ahead."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[col].pct_change(periods=periods) * 100


def _realized_vol(
    df: pd.DataFrame, col: str, window: int
) -> pd.Series:
    """Annualized realized volatility from log returns.

    Rolling window, past-only. Result in percentage points.
    """
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    log_returns = np.log(df[col] / df[col].shift(1))
    return (
        log_returns
        .rolling(window=window, min_periods=window)
        .std()
        * np.sqrt(TRADING_DAYS_PER_YEAR)
        * 100
    )


def _rolling_return_std(
    df: pd.DataFrame, col: str, window: int
) -> pd.Series:
    """Rolling standard deviation of simple returns. Past-only."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    returns = df[col].pct_change()
    return returns.rolling(window=window, min_periods=window).std()


def _momentum(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    """Price vs N-day moving average, as percentage deviation."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    ma = df[col].rolling(window=window, min_periods=window).mean()
    return (df[col] / ma - 1) * 100


def _drawdown(df: pd.DataFrame, col: str) -> pd.Series:
    """Drawdown from expanding maximum (running all-time high).

    Uses expanding(min_periods=1).max() — strictly past-only.
    Result is in percentage (always <= 0).
    """
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    cummax = df[col].expanding(min_periods=1).max()
    return (df[col] - cummax) / cummax * 100


def _yoy_growth(df: pd.DataFrame, col: str) -> pd.Series:
    """Year-over-year growth rate for daily-frequency data.

    Uses shift(252) as ~1 year of trading days.
    Handles division by zero gracefully.
    """
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    shifted = df[col].shift(TRADING_DAYS_PER_YEAR)
    result = (df[col] / shifted - 1) * 100
    return result.replace([np.inf, -np.inf], np.nan)


def _sahm_rule(df: pd.DataFrame, col: str) -> pd.Series:
    """Sahm Rule recession indicator for daily-frequency data.

    Calculation: 3-month moving average of unemployment rate minus
    the 12-month minimum of that 3-month moving average.

    Uses 63 trading days ≈ 3 months, 252 trading days ≈ 12 months.
    A value > 0.50 historically signals a recession.
    """
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    u3ma = df[col].rolling(
        window=TRADING_DAYS_PER_QUARTER,
        min_periods=TRADING_DAYS_PER_QUARTER
    ).mean()
    min_12m = u3ma.rolling(
        window=TRADING_DAYS_PER_YEAR,
        min_periods=TRADING_DAYS_PER_QUARTER
    ).min()
    return u3ma - min_12m
