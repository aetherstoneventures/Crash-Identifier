"""Crash predictions page for dashboard."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import List

from src.utils.database import DatabaseManager, Prediction


class CrashPredictionsPage:
    """Crash predictions page component."""
    
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
    def plot_crash_probability(predictions_df: pd.DataFrame):
        """Plot crash probability over time."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=predictions_df['prediction_date'],
            y=predictions_df['crash_probability'],
            mode='lines+markers',
            name='Crash Probability',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ))
        
        # Add threshold line
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="orange",
            annotation_text="Alert Threshold"
        )
        
        fig.update_layout(
            title='Crash Probability Over Time',
            xaxis_title='Date',
            yaxis_title='Probability',
            hovermode='x unified',
            height=400,
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render():
        """Render crash predictions page."""
        st.header("Crash Predictions")
        st.markdown("Ensemble model predictions for market crashes")
        
        # Load predictions
        predictions = CrashPredictionsPage.load_predictions()
        
        if predictions:
            # Convert to DataFrame
            pred_data = []
            for p in predictions:
                pred_data.append({
                    'prediction_date': p.prediction_date,
                    'crash_probability': p.crash_probability,
                    'confidence_interval': p.confidence_interval,
                    'model_version': p.model_version
                })
            
            pred_df = pd.DataFrame(pred_data)
            
            # Display table
            st.subheader("Prediction History")
            display_df = pred_df.copy()
            display_df['crash_probability'] = display_df['crash_probability'].apply(
                lambda x: f"{x:.2%}"
            )
            display_df['confidence_interval'] = display_df['confidence_interval'].apply(
                lambda x: f"{x:.2%}"
            )
            st.dataframe(display_df, use_container_width=True)
            
            # Plot
            st.subheader("Probability Trend")
            CrashPredictionsPage.plot_crash_probability(pred_df)
            
            # Statistics
            st.subheader("Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Latest Probability", f"{pred_df['crash_probability'].iloc[0]:.2%}")
            with col2:
                st.metric("Average Probability", f"{pred_df['crash_probability'].mean():.2%}")
            with col3:
                st.metric("Max Probability", f"{pred_df['crash_probability'].max():.2%}")
            with col4:
                st.metric("Min Probability", f"{pred_df['crash_probability'].min():.2%}")
        else:
            st.info("No predictions available yet")

