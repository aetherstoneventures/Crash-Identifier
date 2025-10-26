"""Indicators page for dashboard."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import List

from src.utils.database import DatabaseManager, Indicator


class IndicatorsPage:
    """Indicators page component."""
    
    @staticmethod
    def load_indicators() -> List:
        """Load indicators from database."""
        db = DatabaseManager()
        session = db.get_session()
        try:
            indicators = session.query(Indicator).order_by(
                Indicator.date.desc()
            ).limit(365).all()
        finally:
            session.close()
        return indicators
    
    @staticmethod
    def plot_indicators(indicators_df: pd.DataFrame, columns: List[str]):
        """Plot selected indicators."""
        fig = go.Figure()
        
        for col in columns:
            if col in indicators_df.columns:
                fig.add_trace(go.Scatter(
                    x=indicators_df['date'],
                    y=indicators_df[col],
                    mode='lines',
                    name=col.replace('_', ' ').title()
                ))
        
        fig.update_layout(
            title='Economic Indicators',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render():
        """Render indicators page."""
        st.header("Economic Indicators")
        st.markdown("Real-time economic indicators from FRED and Yahoo Finance")
        
        # Load indicators
        indicators = IndicatorsPage.load_indicators()
        
        if indicators:
            # Convert to DataFrame
            ind_data = []
            for i in indicators:
                ind_data.append({
                    'date': i.date,
                    'yield_spread_10y_2y': i.yield_spread_10y_2y,
                    'credit_spread_baa_aaa': i.credit_spread_baa_aaa,
                    'vix_close': i.vix_close,
                    'unemployment_rate': i.unemployment_rate,
                    'inflation_rate': i.inflation_rate
                })
            
            ind_df = pd.DataFrame(ind_data)
            
            # Display table
            st.subheader("Latest Indicators")
            st.dataframe(ind_df.head(10), use_container_width=True)
            
            # Select indicators to plot
            st.subheader("Indicator Selection")
            available_cols = [
                'yield_spread_10y_2y',
                'credit_spread_baa_aaa',
                'vix_close',
                'unemployment_rate',
                'inflation_rate'
            ]
            
            selected_cols = st.multiselect(
                "Select indicators to plot",
                available_cols,
                default=available_cols[:3]
            )
            
            if selected_cols:
                st.subheader("Indicator Trends")
                IndicatorsPage.plot_indicators(ind_df, selected_cols)
            
            # Statistics
            st.subheader("Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Latest Yield Spread",
                    f"{ind_df['yield_spread_10y_2y'].iloc[0]:.2f}%"
                )
            with col2:
                st.metric(
                    "Latest VIX",
                    f"{ind_df['vix_close'].iloc[0]:.2f}"
                )
            with col3:
                st.metric(
                    "Latest Unemployment",
                    f"{ind_df['unemployment_rate'].iloc[0]:.2f}%"
                )
        else:
            st.info("No indicators available yet")

