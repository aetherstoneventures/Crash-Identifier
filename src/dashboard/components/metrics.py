"""Metrics components for dashboard."""

import streamlit as st
from typing import List, Dict, Any


class MetricsCard:
    """Single metrics card component."""
    
    def __init__(self, label: str, value: Any, delta: str = None, color: str = "blue"):
        """
        Initialize metrics card.
        
        Args:
            label: Card label
            value: Metric value
            delta: Change indicator (e.g., "+5%")
            color: Card color (blue, green, red, orange)
        """
        self.label = label
        self.value = value
        self.delta = delta
        self.color = color
    
    def render(self):
        """Render metrics card."""
        st.metric(
            label=self.label,
            value=self.value,
            delta=self.delta
        )


class MetricsRow:
    """Row of metrics cards."""
    
    def __init__(self, metrics: List[Dict[str, Any]], columns: int = 4):
        """
        Initialize metrics row.
        
        Args:
            metrics: List of metric dictionaries with keys: label, value, delta
            columns: Number of columns
        """
        self.metrics = metrics
        self.columns = columns
    
    def render(self):
        """Render metrics row."""
        cols = st.columns(self.columns)
        
        for i, metric in enumerate(self.metrics):
            with cols[i % self.columns]:
                st.metric(
                    label=metric.get('label', ''),
                    value=metric.get('value', 0),
                    delta=metric.get('delta', None)
                )

