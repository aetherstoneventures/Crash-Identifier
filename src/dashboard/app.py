"""Main Streamlit dashboard application for Market Crash & Bottom Prediction System."""

import sys
import os
from pathlib import Path

# Add the project root to the Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import pickle
import logging

from src.utils.config import DATABASE_URL
from src.utils.database import DatabaseManager, Indicator, Prediction
from src.data_collection.fred_collector import FREDCollector
from src.data_collection.yahoo_collector import YahooCollector
from src.feature_engineering.feature_pipeline import FeaturePipeline
# Note: Models are loaded from pickle files, not imported as classes

# Initialize database manager
db_manager = DatabaseManager()
logger = logging.getLogger(__name__)


@st.cache_data(ttl=300)  # Cache for 5 minutes instead of 1 hour
def load_all_indicators():
    """Load all indicators from database (cached for 5 minutes).

    Uses proper context manager to ensure session is always closed.
    """
    with db_manager.get_session() as session:
        try:
            indicators = session.query(Indicator).order_by(Indicator.date.asc()).all()
            # Detach objects from session before returning (for caching)
            session.expunge_all()
            return indicators
        except Exception as e:
            logger.error(f"Error loading indicators: {e}")
            return []


@st.cache_data(ttl=300)  # Cache for 5 minutes instead of 1 hour
def load_all_predictions():
    """Load all predictions from database (cached for 5 minutes).

    Uses proper context manager to ensure session is always closed.
    """
    with db_manager.get_session() as session:
        try:
            predictions = session.query(Prediction).order_by(Prediction.prediction_date.asc()).all()
            # Detach objects from session before returning (for caching)
            session.expunge_all()
            return predictions
        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
            return []


def indicators_to_dataframe(indicators):
    """Convert indicator objects to DataFrame."""
    if not indicators:
        return pd.DataFrame()

    data = []
    for ind in indicators:
        data.append({
            'date': ind.date,
            # 20 Usable Indicators with 100% Data Coverage
            # Yield curve (3)
            'yield_10y_3m': ind.yield_10y_3m,
            'yield_10y_2y': ind.yield_10y_2y,
            'yield_10y': ind.yield_10y,
            # Credit (1)
            'credit_spread_bbb': ind.credit_spread_bbb,
            # Economic (5)
            'unemployment_rate': ind.unemployment_rate,
            'real_gdp': ind.real_gdp,
            'cpi': ind.cpi,
            'fed_funds_rate': ind.fed_funds_rate,
            'industrial_production': ind.industrial_production,
            # Market (3)
            'sp500_close': ind.sp500_close,
            'sp500_volume': ind.sp500_volume,
            'vix_close': ind.vix_close,
            # Sentiment (1)
            'consumer_sentiment': ind.consumer_sentiment,
            # Housing (1)
            'housing_starts': ind.housing_starts,
            # Monetary (1)
            'm2_money_supply': ind.m2_money_supply,
            # Debt (1)
            'debt_to_gdp': ind.debt_to_gdp,
            # Savings (1)
            'savings_rate': ind.savings_rate,
            # Composite (1)
            'lei': ind.lei,
            # Alternative data sources (2) - SYNTHETIC PROXIES
            'margin_debt': ind.margin_debt,  # ⚠️ SYNTHETIC: 100/(credit_spread+1), NOT real FINRA data
            'put_call_ratio': ind.put_call_ratio,  # ⚠️ SYNTHETIC: 1.0+(VIX_change×0.5), NOT real CBOE data
        })

    return pd.DataFrame(data)


def predictions_to_dataframe(predictions):
    """Convert prediction objects to DataFrame."""
    if not predictions:
        return pd.DataFrame()

    data = []
    for pred in predictions:
        data.append({
            'prediction_date': pred.prediction_date,
            'crash_probability': pred.crash_probability,
            'confidence_lower': pred.confidence_interval_lower,
            'confidence_upper': pred.confidence_interval_upper,
            'bottom_prediction_date': pred.bottom_prediction_date,
            'recovery_prediction_date': pred.recovery_prediction_date,
            'model_version': pred.model_version,
        })

    return pd.DataFrame(data)


@st.cache_data(ttl=300)
def calculate_rate_of_change(pred_df, column_name, periods=[1, 5, 20]):
    """
    Calculate rate of change for predictions.

    Args:
        pred_df: DataFrame with predictions
        column_name: Column to calculate rate of change for
        periods: List of periods to calculate (1-day, 5-day, 20-day)

    Returns:
        DataFrame with rate of change columns
    """
    if pred_df.empty or column_name not in pred_df.columns:
        return pd.DataFrame()

    # Use prediction_date if available, otherwise use date
    date_col = 'prediction_date' if 'prediction_date' in pred_df.columns else 'date'
    result = pred_df[[date_col, column_name]].copy()

    for period in periods:
        # Calculate percentage change over period
        result[f'{column_name}_roc_{period}d'] = pred_df[column_name].pct_change(periods=period) * 100

    return result


@st.cache_data(ttl=300)
def calculate_statistical_predictions(_indicators):
    """Calculate statistical/rule-based crash predictions.

    Note: Statistical predictions are now stored in the database by the training pipeline.
    This function is kept for backward compatibility but returns empty DataFrame.
    Use the predictions from the database instead.
    """
    # Statistical predictions are now generated by train_statistical_model_v2.py
    # and stored in the predictions table with model_version='statistical_v2'
    return pd.DataFrame()


def plot_sp500_price(df):
    """Create S&P 500 price chart."""
    if df.empty or 'sp500_close' not in df.columns:
        return None

    # Get valid data for scaling
    valid_prices = df[df['sp500_close'].notna()]['sp500_close']
    if not valid_prices.empty:
        min_price = valid_prices.min()
        max_price = valid_prices.max()
        range_price = max_price - min_price
        y_min = min_price - (range_price * 0.05)
        y_max = max_price + (range_price * 0.05)
    else:
        y_min = 0
        y_max = 10000

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['sp500_close'],
        mode='lines',
        name='S&P 500',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.1)',
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title='S&P 500 Price History',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        height=500,
        template='plotly_dark',
        yaxis=dict(range=[y_min, y_max], automargin=True),
        xaxis=dict(automargin=True),
        margin=dict(l=70, r=50, t=70, b=50)
    )
    return fig


def plot_vix_indicator(df):
    """Create VIX volatility chart."""
    if df.empty or 'vix_close' not in df.columns:
        return None

    # Get valid data for scaling
    valid_vix = df[df['vix_close'].notna()]['vix_close']
    if not valid_vix.empty:
        min_vix = valid_vix.min()
        max_vix = valid_vix.max()
        range_vix = max_vix - min_vix
        y_min = max(0, min_vix - (range_vix * 0.1))
        y_max = max_vix + (range_vix * 0.1)
    else:
        y_min = 0
        y_max = 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['vix_close'],
        mode='lines',
        name='VIX',
        line=dict(color='#ff7f0e', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 127, 14, 0.2)',
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>VIX:</b> %{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title='VIX Volatility Index',
        xaxis_title='Date',
        yaxis_title='VIX Level',
        hovermode='x unified',
        height=400,
        template='plotly_dark',
        yaxis=dict(range=[y_min, y_max], automargin=True),
        xaxis=dict(automargin=True),
        margin=dict(l=70, r=50, t=70, b=50)
    )
    return fig


def plot_yield_spreads(df):
    """Create yield spread chart using raw yield data."""
    if df.empty:
        return None

    fig = go.Figure()

    if 'yield_10y_2y' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['yield_10y_2y'],
            mode='lines',
            name='10Y-2Y Spread',
            line=dict(color='#2ca02c', width=2),
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Spread:</b> %{y:.3f}<extra></extra>'
        ))

    if 'yield_10y_3m' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['yield_10y_3m'],
            mode='lines',
            name='10Y-3M Spread',
            line=dict(color='#d62728', width=2),
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Spread:</b> %{y:.3f}<extra></extra>'
        ))

    fig.update_layout(
        title='Yield Curve Spreads',
        xaxis_title='Date',
        yaxis_title='Spread (%)',
        hovermode='x unified',
        height=400,
        template='plotly_dark'
    )
    return fig


def plot_crash_probability(df):
    """Create crash probability chart with confidence intervals."""
    if df.empty or 'crash_probability' not in df.columns:
        return None

    fig = go.Figure()

    # Add confidence interval as shaded area
    if 'confidence_upper' in df.columns and 'confidence_lower' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['prediction_date'],
            y=df['confidence_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=df['prediction_date'],
            y=df['confidence_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='95% Confidence',
            fillcolor='rgba(255, 127, 14, 0.2)',
            hoverinfo='skip'
        ))

    # Add main probability line
    fig.add_trace(go.Scatter(
        x=df['prediction_date'],
        y=df['crash_probability'],
        mode='lines+markers',
        name='Crash Probability',
        line=dict(color='#d62728', width=3),
        marker=dict(size=6),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Probability:</b> %{y:.1%}<extra></extra>'
    ))

    # NO STATIC THRESHOLD - Using rate of change indicators instead

    fig.update_layout(
        title='Market Crash Probability Over Time',
        xaxis_title='Date',
        yaxis_title='Probability',
        hovermode='x unified',
        height=500,
        template='plotly_dark',
        yaxis=dict(range=[0, 1])
    )
    return fig


def plot_bottom_predictions(df):
    """Create bottom prediction chart."""
    if df.empty or 'bottom_prediction_date' not in df.columns:
        return None

    fig = go.Figure()

    # Calculate days to bottom
    df_copy = df.copy()
    df_copy['days_to_bottom'] = (pd.to_datetime(df_copy['bottom_prediction_date']) -
                                  pd.to_datetime(df_copy['prediction_date'])).dt.days

    # Get valid data for scaling
    valid_days = df_copy[df_copy['days_to_bottom'].notna()]['days_to_bottom']
    if not valid_days.empty:
        min_days = valid_days.min()
        max_days = valid_days.max()
        range_days = max_days - min_days
        y_min = max(0, min_days - (range_days * 0.1))
        y_max = max_days + (range_days * 0.1)
    else:
        y_min = 0
        y_max = 100

    fig.add_trace(go.Scatter(
        x=df_copy['prediction_date'],
        y=df_copy['days_to_bottom'],
        mode='lines+markers',
        name='Days to Bottom',
        line=dict(color='#9467bd', width=2),
        marker=dict(size=8),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Days to Bottom:</b> %{y:.0f}<extra></extra>'
    ))

    fig.update_layout(
        title='Predicted Days to Market Bottom',
        xaxis_title='Prediction Date',
        yaxis_title='Days',
        hovermode='x unified',
        height=400,
        template='plotly_dark',
        yaxis=dict(range=[y_min, y_max], automargin=True),
        xaxis=dict(automargin=True),
        margin=dict(l=70, r=50, t=70, b=50)
    )
    return fig


def plot_multiple_indicators(df, indicator_names, title):
    """Create multi-indicator chart with normalized scaling."""
    if df.empty:
        return None

    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Normalize each indicator to 0-100 scale for better visualization
    normalized_data = {}
    for indicator in indicator_names:
        if indicator in df.columns:
            valid_data = df[df[indicator].notna()][indicator]
            if not valid_data.empty:
                min_val = valid_data.min()
                max_val = valid_data.max()
                range_val = max_val - min_val

                if range_val > 0:
                    normalized_data[indicator] = (df[indicator] - min_val) / range_val * 100
                else:
                    normalized_data[indicator] = pd.Series(50, index=df.index)

    # Plot normalized indicators
    for idx, indicator in enumerate(indicator_names):
        if indicator in normalized_data:
            data_to_plot = normalized_data[indicator]
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=data_to_plot,
                mode='lines',
                name=indicator.replace('_', ' ').title(),
                line=dict(color=colors[idx % len(colors)], width=2),
                hovertemplate=f'<b>Date:</b> %{{x|%Y-%m-%d}}<br><b>{indicator}:</b> %{{y:.1f}} (normalized)<extra></extra>'
            ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Normalized Value (0-100)',
        hovermode='x unified',
        height=500,
        template='plotly_dark',
        yaxis=dict(range=[0, 100], automargin=True),
        xaxis=dict(automargin=True),
        margin=dict(l=70, r=50, t=70, b=50)
    )
    return fig


def page_bottom_predictions():
    """Bottom Re-entry — predicts trough date / recovery date AFTER v5 fires.

    The bottom predictor is a separate Gradient Boosting Regressor trained on
    11 historical crashes (1980-2022), with 8 macro features sampled at the
    moment a crash starts.  It outputs:
        - days_to_bottom        (regression target #1)
        - days_to_recovery      (regression target #2)

    It is meant to be consulted ONLY when v5 has fired an alarm — never as a
    standalone signal.  When v5 is in ALL CLEAR the values shown here are the
    model's reflexive output for today's macro state, not actionable.
    """
    st.header("📈 Bottom Re-entry — predicted trough & recovery dates")
    st.caption(
        "Downstream of v5: when the v5 alarm fires, this page tells you the "
        "predicted number of days from today to the trough and to the full "
        "recovery, based on a Gradient Boosting model trained on 11 historical "
        "U.S. equity crashes (1980-2022)."
    )

    try:
        predictions = load_all_predictions()
        pred_df = predictions_to_dataframe(predictions)
        if pred_df.empty:
            st.warning("No predictions available yet. Run the pipeline first.")
            return

        bottom_df = pred_df[pred_df["bottom_prediction_date"].notna()].copy()
        if bottom_df.empty:
            st.warning(
                "No bottom predictions stored. Run "
                "`scripts/utils/generate_bottom_predictions.py`."
            )
            return

        bottom_df["days_to_bottom"] = (
            pd.to_datetime(bottom_df["bottom_prediction_date"])
            - pd.to_datetime(bottom_df["prediction_date"])
        ).dt.days
        bottom_df["days_to_recovery"] = (
            pd.to_datetime(bottom_df["recovery_prediction_date"])
            - pd.to_datetime(bottom_df["prediction_date"])
        ).dt.days

        latest = bottom_df.iloc[-1]

        st.subheader("🎯 Latest prediction (always-on output)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            v = latest["crash_probability"]
            st.metric("Reflexive crash prob", f"{v:.1%}" if pd.notna(v) else "N/A")
        with c2:
            v = latest["days_to_bottom"]
            st.metric("Days to bottom", f"{v:.0f}" if pd.notna(v) else "N/A")
        with c3:
            v = latest["bottom_prediction_date"]
            st.metric("Predicted bottom",
                      pd.to_datetime(v).strftime("%Y-%m-%d") if pd.notna(v) else "N/A")
        with c4:
            v = latest["recovery_prediction_date"]
            st.metric("Predicted recovery",
                      pd.to_datetime(v).strftime("%Y-%m-%d") if pd.notna(v) else "N/A")

        st.markdown("---")
        st.subheader("📉 Days-to-bottom over time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bottom_df["prediction_date"],
            y=bottom_df["days_to_bottom"],
            mode="lines",
            name="Days to bottom",
            line=dict(color="#9467bd", width=1.6),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{y:.0f}d to bottom<extra></extra>",
        ))
        fig.update_layout(
            template="plotly_dark", height=380,
            xaxis_title="Prediction date", yaxis_title="Days to bottom",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True, key="bottom_days_chart")

        st.markdown("---")
        with st.expander("ℹ️ Methodology & honest limitations", expanded=False):
            st.markdown("""
**Model.**  Gradient Boosting Regressor — separate models for `days_to_bottom`
and `days_to_recovery`.  Trained on 11 verified U.S. equity crashes
(1980-2022) with these 8 features sampled at crash-start:

1. VIX level
2. 10y-2y yield spread
3. BBB credit spread
4. Unemployment rate
5. Consumer sentiment
6. Leading Economic Index (LEI)
7. S&P 500 30-day momentum
8. VIX 30-day change

**Validation.**  Leave-one-out CV; results in the per-crash table that the
training script (`scripts/training/train_bottom_predictor.py`) prints to
stdout when run.

**Honest limitations.**
- Tiny training set (n=11) → high variance.  Use as one input, not a target.
- Trained independently of v5; consult this page **only** when v5 is in alarm.
- "Reflexive crash prob" shown above is the bottom predictor's own input
  estimate, not v5's blended probability — for the canonical signal use the
  v5 Production tab.
""")

        st.markdown("---")
        st.subheader("History (last 30 days)")
        disp = bottom_df.tail(30)[[
            "prediction_date", "crash_probability",
            "days_to_bottom", "bottom_prediction_date",
            "days_to_recovery", "recovery_prediction_date",
        ]].copy()
        disp["prediction_date"] = pd.to_datetime(disp["prediction_date"]).dt.strftime("%Y-%m-%d")
        disp["bottom_prediction_date"] = pd.to_datetime(
            disp["bottom_prediction_date"]).dt.strftime("%Y-%m-%d")
        disp["recovery_prediction_date"] = pd.to_datetime(
            disp["recovery_prediction_date"]).dt.strftime("%Y-%m-%d")
        disp["crash_probability"] = disp["crash_probability"].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        disp.columns = [
            "Date", "Reflexive prob",
            "Days to bottom", "Predicted bottom",
            "Days to recovery", "Predicted recovery",
        ]
        st.dataframe(disp, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error loading bottom predictions: {e}")
        logger.error("Bottom predictions error", exc_info=True)


def plot_single_indicator(df, indicator_name, label):
    """Plot a single indicator with auto-scaled y-axis. Returns None if column is fully NaN."""
    if indicator_name not in df.columns or df[indicator_name].isna().all():
        return None
    valid_data = df[df[indicator_name].notna()][indicator_name]
    if valid_data.empty:
        return None
    min_val = valid_data.min(); max_val = valid_data.max()
    range_val = max_val - min_val
    y_min = min_val - (range_val * 0.1)
    y_max = max_val + (range_val * 0.1)
    if range_val == 0:
        y_min = min_val * 0.9 if min_val != 0 else -1
        y_max = max_val * 1.1 if max_val != 0 else 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df[indicator_name], mode="lines", name=label,
        line=dict(color="#1f77b4", width=2),
        fill="tozeroy", fillcolor="rgba(31,119,180,0.2)",
        hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{label}: %{{y:.4f}}<extra></extra>",
    ))
    fig.update_layout(
        title=label, xaxis_title="Date", yaxis_title=label,
        hovermode="x unified", height=350, template="plotly_dark",
        margin=dict(l=60, r=50, t=50, b=50),
        yaxis=dict(range=[y_min, y_max], automargin=True),
        xaxis=dict(automargin=True),
    )
    return fig


def page_indicators():
    """Indicators — v5 features first, then 20 base indicators.

    Skips columns that are entirely NULL with an explicit notice (so the
    user sees WHY a slot is empty, instead of silent skip).
    """
    st.header("📋 Indicators")
    st.caption(
        "v5 trains on these features.  The first block lists what the **v5 "
        "model itself** uses (NASDAQ Composite + 4 FRED-only macro stress "
        "features); the second block is the broader 20-indicator panel "
        "consumed by StatV3 and the bottom predictor."
    )

    indicators = load_all_indicators()
    ind_df = indicators_to_dataframe(indicators)
    if ind_df.empty:
        st.warning("No indicators available yet. Run the pipeline.")
        return

    # Pull the v5-only feature columns directly from the DB (they are not
    # exposed on the Indicator ORM yet, so we re-read).
    import sqlite3
    db_path = Path(__file__).resolve().parents[2] / "data" / "market_crash.db"
    try:
        with sqlite3.connect(str(db_path)) as conn:
            v5_df = pd.read_sql(
                "SELECT date, nasdaq_close, baa_10y_spread, oil_wti, "
                "dollar_twi, epu_daily FROM indicators ORDER BY date",
                conn, parse_dates=["date"],
            )
    except Exception:
        v5_df = pd.DataFrame()

    # =================================================================
    # Block 1: v5 features
    # =================================================================
    st.markdown("### 🛡️ v5 model features (5)")
    st.caption(
        "These are the columns that feed the XGBoost predictive model in v5. "
        "Coverage tells you how much usable history the model actually saw."
    )

    v5_features = [
        ("nasdaq_close",     "NASDAQ Composite (FRED: NASDAQCOM)",
         "Primary equity series — drawdown labels are computed from this."),
        ("baa_10y_spread",   "Baa corporate bond − 10y Treasury spread (FRED: BAA10Y)",
         "Credit-stress proxy; widens before/during recessions."),
        ("oil_wti",          "WTI crude oil price (FRED: DCOILWTICO)",
         "Macro-shock proxy; major spikes/crashes precede growth scares."),
        ("dollar_twi",       "Trade-weighted dollar index (FRED: DTWEXBGS)",
         "Risk-off proxy; dollar strength accompanies many crash episodes."),
        ("epu_daily",        "Daily Economic Policy Uncertainty (FRED: USEPUINDXD)",
         "Sentiment/uncertainty proxy; spikes during crisis."),
    ]

    if v5_df.empty:
        st.error("Could not read v5 feature columns from `data/market_crash.db`.")
    else:
        for col, label, why in v5_features:
            if col not in v5_df.columns:
                st.info(f"**{label}** — column not present in DB. "
                        f"Run `scripts/data/fetch_v5_features.py`.")
                continue
            n = v5_df[col].notna().sum()
            total = len(v5_df)
            st.markdown(f"**{label}**  &nbsp; *({n:,}/{total:,} non-null)*")
            st.caption(why)
            if n == 0:
                st.warning("📭 No data populated for this column.")
                continue
            fig = plot_single_indicator(v5_df, col, label)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, key=f"v5_feat_{col}")

    st.markdown("---")

    # =================================================================
    # Block 2: 20 base indicators (StatV3 / bottom predictor inputs)
    # =================================================================
    st.markdown("### 📊 Base indicator panel (20) — used by StatV3 and the bottom predictor")
    st.warning(
        "⚠️ Two of the 20 are **synthetic proxies**: "
        "`margin_debt = 100/(credit_spread + 1)` (NOT real FINRA data); "
        "`put_call_ratio = 1.0 + (VIX_change × 0.5)` (NOT real CBOE data). "
        "All others are real series from FRED or Yahoo Finance."
    )

    all_indicators = [
        ("yield_10y_3m",         "Yield Spread 10Y-3M"),
        ("yield_10y_2y",         "Yield Spread 10Y-2Y"),
        ("yield_10y",            "10-Year Treasury Yield"),
        ("credit_spread_bbb",    "BBB Credit Spread"),
        ("unemployment_rate",    "Unemployment Rate"),
        ("real_gdp",             "Real GDP"),
        ("cpi",                  "CPI Index"),
        ("fed_funds_rate",       "Fed Funds Rate"),
        ("industrial_production","Industrial Production"),
        ("sp500_close",          "S&P 500 Price"),
        ("sp500_volume",         "S&P 500 Volume"),
        ("vix_close",            "VIX Index"),
        ("consumer_sentiment",   "Consumer Sentiment"),
        ("housing_starts",       "Housing Starts"),
        ("m2_money_supply",      "M2 Money Supply"),
        ("debt_to_gdp",          "Debt to GDP"),
        ("savings_rate",         "Savings Rate"),
        ("lei",                  "Leading Economic Index"),
        ("margin_debt",          "Margin Debt (synthetic)"),
        ("put_call_ratio",       "Put/Call Ratio (synthetic)"),
    ]

    # Coverage summary first
    cov_rows = []
    for col, label in all_indicators:
        if col in ind_df.columns:
            n = ind_df[col].notna().sum()
            total = len(ind_df)
            pct = n / total * 100 if total else 0
            cov_rows.append({
                "Indicator": label, "Column": col,
                "Non-null": f"{n:,} / {total:,}",
                "Coverage": f"{pct:.1f}%",
                "Status": "✅ ok" if pct >= 50 else ("⚠️ sparse" if pct > 0 else "❌ empty"),
            })
    cov_df = pd.DataFrame(cov_rows)
    st.dataframe(cov_df, use_container_width=True, hide_index=True)

    # Filter out fully-NULL columns from the chart selector — there is
    # nothing to plot, so silently hide them rather than producing
    # blank checkboxes.
    plottable = [(c, l) for c, l in all_indicators
                 if c in ind_df.columns and ind_df[c].notna().any()]
    hidden = [(c, l) for c, l in all_indicators
              if c in ind_df.columns and not ind_df[c].notna().any()]
    if hidden:
        st.info(
            "Hidden because no data is populated in the DB: "
            + ", ".join(f"`{c}`" for c, _ in hidden)
            + ". Most likely cause: Yahoo Finance is blocked, so series that "
            "depend on it (e.g. `sp500_volume`) cannot be backfilled from FRED."
        )

    if not plottable:
        st.error("No plottable indicators in the DB.")
        return

    st.markdown("#### Charts")
    cols = st.columns(3)
    selected = []
    for i, (col, label) in enumerate(plottable):
        with cols[i % 3]:
            if st.checkbox(label, value=(i < 6), key=f"ind_check_{col}"):
                selected.append((col, label))

    for col, label in selected:
        fig = plot_single_indicator(ind_df, col, label)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, key=f"ind_plot_{col}")


def main():
    """Main dashboard application — 3 v5-aware tabs."""
    st.set_page_config(
        page_title="v5 Crash Detector",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Hide Streamlit chrome
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stSidebar"] {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    tab_v5, tab_fr, tab_ind, tab_bot = st.tabs([
        "🛡️ v5 Production",
        "🔮 Forward Risk (1m)",
        "📋 Indicators",
        "📈 Bottom Re-entry",
    ])

    with tab_v5:
        from src.dashboard.pages import v5_production
        v5_production.render()
    with tab_fr:
        from src.dashboard.pages import forward_risk
        forward_risk.render()
    with tab_ind:
        page_indicators()
    with tab_bot:
        page_bottom_predictions()


if __name__ == "__main__":
    main()
