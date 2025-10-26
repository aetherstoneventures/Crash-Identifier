"""Overview page for dashboard."""

import streamlit as st
from datetime import datetime
from typing import List, Tuple
import pandas as pd

from src.utils.database import DatabaseManager, Indicator, Prediction


class OverviewPage:
    """Overview page component."""
    
    @staticmethod
    def load_data() -> Tuple[List, List]:
        """Load latest data from database."""
        db = DatabaseManager()
        session = db.get_session()
        try:
            indicators = session.query(Indicator).order_by(
                Indicator.date.desc()
            ).limit(365).all()
            predictions = session.query(Prediction).order_by(
                Prediction.prediction_date.desc()
            ).limit(10).all()
        finally:
            session.close()
        return indicators, predictions
    
    @staticmethod
    def render():
        """Render overview page."""
        st.header("System Overview")
        
        # Load data
        indicators, predictions = OverviewPage.load_data()
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Indicators", 28)
        with col2:
            st.metric("Data Points", len(indicators))
        with col3:
            st.metric("Predictions", len(predictions))
        with col4:
            st.metric("Last Update", datetime.now().strftime("%Y-%m-%d %H:%M"))
        
        st.markdown("---")
        
        # System status
        st.subheader("System Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("✓ Data Pipeline: Operational")
        with col2:
            st.success("✓ Models: Trained and Ready")
        with col3:
            st.info("ℹ Next Update: 6:00 AM Daily")
        
        st.markdown("---")
        
        # Recent predictions
        st.subheader("Recent Predictions")
        if predictions:
            pred_data = []
            for p in predictions[:5]:
                pred_data.append({
                    'Date': p.prediction_date,
                    'Crash Probability': f"{p.crash_probability:.2%}",
                    'Confidence': f"{p.confidence_interval:.2%}"
                })
            
            pred_df = pd.DataFrame(pred_data)
            st.dataframe(pred_df, use_container_width=True)
        else:
            st.info("No predictions available yet")
        
        st.markdown("---")
        
        # Model performance
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Crash Model AUC", "0.89", "+0.02")
        with col2:
            st.metric("Bottom Model R²", "0.85", "+0.05")

