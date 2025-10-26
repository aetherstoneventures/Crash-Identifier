"""Chart components for dashboard."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Optional


class PriceChart:
    """Price chart component."""
    
    def __init__(self, prices: pd.Series, title: str = "Price History"):
        """
        Initialize price chart.
        
        Args:
            prices: Price series with datetime index
            title: Chart title
        """
        self.prices = prices
        self.title = title
    
    def render(self):
        """Render price chart."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.prices.index,
            y=self.prices.values,
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.update_layout(
            title=self.title,
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


class IndicatorsChart:
    """Indicators chart component."""
    
    def __init__(self, indicators_df: pd.DataFrame, title: str = "Indicators"):
        """
        Initialize indicators chart.
        
        Args:
            indicators_df: DataFrame with indicators
            title: Chart title
        """
        self.indicators_df = indicators_df
        self.title = title
    
    def render(self):
        """Render indicators chart."""
        fig = go.Figure()
        
        for col in self.indicators_df.columns:
            if col != 'date':
                fig.add_trace(go.Scatter(
                    x=self.indicators_df['date'],
                    y=self.indicators_df[col],
                    mode='lines',
                    name=col
                ))
        
        fig.update_layout(
            title=self.title,
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


class PredictionsChart:
    """Predictions chart component."""
    
    def __init__(self, predictions_df: pd.DataFrame, 
                 title: str = "Predictions", metric: str = "probability"):
        """
        Initialize predictions chart.
        
        Args:
            predictions_df: DataFrame with predictions
            title: Chart title
            metric: Metric to plot (probability, days_to_bottom, etc.)
        """
        self.predictions_df = predictions_df
        self.title = title
        self.metric = metric
    
    def render(self):
        """Render predictions chart."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.predictions_df['date'],
            y=self.predictions_df[self.metric],
            mode='lines+markers',
            name=self.metric.replace('_', ' ').title(),
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=self.title,
            xaxis_title='Date',
            yaxis_title=self.metric.replace('_', ' ').title(),
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

