"""Rate-of-change based alert system for crash predictions."""

import logging
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class RateOfChangeAlert:
    """
    Alert system based on rate of change of crash probabilities.
    
    Instead of static thresholds, this system monitors how fast probabilities
    are changing to detect emerging risks.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize rate-of-change alert system.
        
        Args:
            window_size: Number of days to use for rate-of-change calculation
        """
        self.window_size = window_size
        self.ml_history = []  # List of (date, probability) tuples
        self.stat_history = []  # List of (date, probability) tuples
        self.logger = logging.getLogger(__name__)
    
    def add_prediction(self, date: datetime, ml_prob: float, stat_prob: float) -> None:
        """
        Add a new prediction to history.
        
        Args:
            date: Prediction date
            ml_prob: ML model crash probability (0-1)
            stat_prob: Statistical model crash probability (0-1)
        """
        self.ml_history.append((date, ml_prob))
        self.stat_history.append((date, stat_prob))
        
        # Keep only recent history
        cutoff_date = datetime.now() - timedelta(days=self.window_size * 2)
        self.ml_history = [(d, p) for d, p in self.ml_history if d >= cutoff_date]
        self.stat_history = [(d, p) for d, p in self.stat_history if d >= cutoff_date]
    
    def calculate_rate_of_change(self, history: list) -> Optional[float]:
        """
        Calculate rate of change of probabilities.
        
        Args:
            history: List of (date, probability) tuples
            
        Returns:
            Rate of change (probability change per day), or None if insufficient data
        """
        if len(history) < 2:
            return None
        
        # Get recent data points
        recent = history[-self.window_size:] if len(history) >= self.window_size else history
        
        if len(recent) < 2:
            return None
        
        dates = [d for d, _ in recent]
        probs = [p for _, p in recent]
        
        # Calculate days elapsed
        days_elapsed = (dates[-1] - dates[0]).days
        if days_elapsed == 0:
            return None
        
        # Calculate rate of change (probability change per day)
        prob_change = probs[-1] - probs[0]
        rate_of_change = prob_change / days_elapsed
        
        return rate_of_change
    
    def get_alert_status(self) -> Dict[str, any]:
        """
        Get current alert status based on rate of change.
        
        Returns:
            Dictionary with alert information
        """
        ml_roc = self.calculate_rate_of_change(self.ml_history)
        stat_roc = self.calculate_rate_of_change(self.stat_history)
        
        # Get current probabilities
        ml_current = self.ml_history[-1][1] if self.ml_history else 0.0
        stat_current = self.stat_history[-1][1] if self.stat_history else 0.0
        
        # Determine alert level based on rate of change
        # Thresholds (probability change per day)
        roc_warning = 0.05  # 5% per day
        roc_critical = 0.10  # 10% per day
        
        ml_alert_level = self._get_alert_level(ml_roc, roc_warning, roc_critical)
        stat_alert_level = self._get_alert_level(stat_roc, roc_warning, roc_critical)
        
        return {
            'timestamp': datetime.now(),
            'ml_probability': ml_current,
            'ml_rate_of_change': ml_roc,
            'ml_alert_level': ml_alert_level,
            'stat_probability': stat_current,
            'stat_rate_of_change': stat_roc,
            'stat_alert_level': stat_alert_level,
            'combined_alert_level': max(ml_alert_level, stat_alert_level),
            'should_alert': max(ml_alert_level, stat_alert_level) >= 2,  # Alert if warning or critical
        }
    
    @staticmethod
    def _get_alert_level(rate_of_change: Optional[float], warning_threshold: float, 
                        critical_threshold: float) -> int:
        """
        Get alert level based on rate of change.
        
        Args:
            rate_of_change: Rate of change (probability change per day)
            warning_threshold: Warning threshold
            critical_threshold: Critical threshold
            
        Returns:
            Alert level: 0=normal, 1=warning, 2=critical
        """
        if rate_of_change is None:
            return 0
        
        # Use absolute value to catch both increases and decreases
        abs_roc = abs(rate_of_change)
        
        if abs_roc >= critical_threshold:
            return 2  # Critical
        elif abs_roc >= warning_threshold:
            return 1  # Warning
        else:
            return 0  # Normal
    
    def get_alert_message(self) -> str:
        """
        Generate human-readable alert message.
        
        Returns:
            Alert message string
        """
        status = self.get_alert_status()
        
        ml_roc = status['ml_rate_of_change']
        stat_roc = status['stat_rate_of_change']
        ml_level = status['ml_alert_level']
        stat_level = status['stat_alert_level']
        
        messages = []
        
        if ml_level == 2:
            messages.append(f"🚨 CRITICAL: ML probability changing at {ml_roc:.2%}/day")
        elif ml_level == 1:
            messages.append(f"⚠️  WARNING: ML probability changing at {ml_roc:.2%}/day")
        
        if stat_level == 2:
            messages.append(f"🚨 CRITICAL: Statistical probability changing at {stat_roc:.2%}/day")
        elif stat_level == 1:
            messages.append(f"⚠️  WARNING: Statistical probability changing at {stat_roc:.2%}/day")
        
        if not messages:
            messages.append("✓ Normal: Probabilities stable")
        
        return "\n".join(messages)
    
    def reset(self) -> None:
        """Reset alert history."""
        self.ml_history = []
        self.stat_history = []

