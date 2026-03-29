"""Bottom predictions page for dashboard."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import List

from src.utils.database import DatabaseManager, Prediction


class BottomPredictionsPage:
    """Bottom predictions page component."""
    
    @staticmethod
    def load_predictions() -> List:
        """Load predictions from database."""
        db = DatabaseManager()
        session = db.get_session()
        try:
            predictions = session.query(Prediction).order_by(
                Prediction.prediction_date.desc()
            ).limit(100).all()
        finally:
            session.close()
        return predictions
    
    @staticmethod
    def plot_days_to_bottom(predictions_df: pd.DataFrame):
        """Plot days to bottom over time."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=predictions_df['prediction_date'],
            y=predictions_df['days_to_bottom'],
            mode='lines+markers',
            name='Days to Bottom',
            line=dict(color='green', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title='Days to Market Bottom Over Time',
            xaxis_title='Date',
            yaxis_title='Days',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render():
        """Render bottom predictions page."""
        st.header("Bottom Predictions")
        st.markdown("Predicted days until market reaches bottom after crash")
        
        # Load predictions
        predictions = BottomPredictionsPage.load_predictions()
        
        if predictions:
            # Convert to DataFrame
            pred_data = []
            for p in predictions:
                pred_data.append({
                    'prediction_date': p.prediction_date,
                    'days_to_bottom': p.bottom_prediction_date,
                    'recovery_date': p.recovery_prediction_date,
                    'model_version': p.model_version
                })
            
            pred_df = pd.DataFrame(pred_data)
            
            # Display table
            st.subheader("Prediction History")
            st.dataframe(pred_df, use_container_width=True)
            
            # Plot
            st.subheader("Days to Bottom Trend")
            BottomPredictionsPage.plot_days_to_bottom(pred_df)
            
            # Statistics
            st.subheader("Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Latest Prediction",
                    f"{pred_df['days_to_bottom'].iloc[0]:.0f} days"
                )
            with col2:
                st.metric(
                    "Average",
                    f"{pred_df['days_to_bottom'].mean():.0f} days"
                )
            with col3:
                st.metric(
                    "Max",
                    f"{pred_df['days_to_bottom'].max():.0f} days"
                )
            with col4:
                st.metric(
                    "Min",
                    f"{pred_df['days_to_bottom'].min():.0f} days"
                )
        else:
            st.info("No predictions available yet")

