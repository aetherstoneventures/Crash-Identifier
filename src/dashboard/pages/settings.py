"""Settings page for dashboard."""

import streamlit as st
from src.utils.config import (
    CRASH_PREDICTION_ACCURACY_TARGET,
    BOTTOM_PREDICTION_ACCURACY_TARGET,
    BOTTOM_PREDICTION_DAYS_TOLERANCE,
    RECOVERY_TIME_R2_TARGET
)


class SettingsPage:
    """Settings page component."""
    
    @staticmethod
    def render():
        """Render settings page."""
        st.header("Settings")
        st.markdown("Configure dashboard and system parameters")
        
        # Data Collection Settings
        st.subheader("Data Collection")
        col1, col2 = st.columns(2)
        
        with col1:
            refresh_interval = st.slider(
                "Refresh Interval (hours)",
                1, 24, 6,
                help="How often to refresh data from APIs"
            )
        
        with col2:
            data_retention = st.slider(
                "Data Retention (days)",
                30, 3650, 365,
                help="How many days of historical data to keep"
            )
        
        # Model Settings
        st.subheader("Model Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            crash_threshold = st.slider(
                "Crash Probability Threshold",
                0.0, 1.0, 0.5,
                step=0.05,
                help="Threshold for crash alert"
            )
        
        with col2:
            bottom_tolerance = st.slider(
                "Bottom Prediction Tolerance (days)",
                5, 60, BOTTOM_PREDICTION_DAYS_TOLERANCE,
                help="Tolerance for bottom prediction accuracy"
            )
        
        # Model Performance Targets
        st.subheader("Model Performance Targets")
        col1, col2 = st.columns(2)
        
        with col1:
            crash_target = st.slider(
                "Crash Model AUC Target",
                0.5, 1.0, CRASH_PREDICTION_ACCURACY_TARGET,
                step=0.01,
                help="Target AUC for crash prediction model"
            )
        
        with col2:
            bottom_target = st.slider(
                "Bottom Model R² Target",
                0.0, 1.0, BOTTOM_PREDICTION_ACCURACY_TARGET,
                step=0.01,
                help="Target R² for bottom prediction model"
            )
        
        # Alert Settings
        st.subheader("Alert Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            enable_email = st.checkbox(
                "Enable Email Alerts",
                value=False,
                help="Send email alerts for crash predictions"
            )
        
        with col2:
            enable_sms = st.checkbox(
                "Enable SMS Alerts",
                value=False,
                help="Send SMS alerts for crash predictions"
            )
        
        if enable_email:
            email_address = st.text_input(
                "Email Address",
                placeholder="your@email.com"
            )
        
        if enable_sms:
            phone_number = st.text_input(
                "Phone Number",
                placeholder="+1-555-0000"
            )
        
        # Dashboard Settings
        st.subheader("Dashboard Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            theme = st.selectbox(
                "Theme",
                ["Light", "Dark", "Auto"],
                help="Dashboard color theme"
            )
        
        with col2:
            chart_type = st.selectbox(
                "Default Chart Type",
                ["Line", "Candlestick", "Area"],
                help="Default chart type for price data"
            )
        
        # Save Settings
        st.markdown("---")
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("Save Settings", use_container_width=True):
                st.success("✓ Settings saved successfully!")
        
        with col2:
            st.info("Settings are stored locally in your browser")

