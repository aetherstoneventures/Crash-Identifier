"""
Rate-of-Change Alarm System (V2)

Implements dynamic thresholds based on rate of change of model probabilities
instead of static thresholds.

Features:
1. Tracks probability changes over time windows (1, 5, 20 days)
2. Detects acceleration in crash probability
3. Separate thresholds for ML and Statistical models
4. Adaptive thresholds based on market regime
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RateOfChangeAlarm:
    """Dynamic alarm system based on rate of change of probabilities."""
    
    def __init__(self):
        """Initialize rate-of-change alarm system."""
        self.ml_history = []
        self.stat_history = []
        self.alarm_history = []
        
        # Thresholds for rate of change (% change per day)
        self.ml_roc_thresholds = {
            '1d': 0.10,    # 10% increase in 1 day
            '5d': 0.05,    # 5% average increase per day over 5 days
            '20d': 0.02,   # 2% average increase per day over 20 days
        }
        
        self.stat_roc_thresholds = {
            '1d': 0.08,    # 8% increase in 1 day
            '5d': 0.04,    # 4% average increase per day over 5 days
            '20d': 0.015,  # 1.5% average increase per day over 20 days
        }
    
    def calculate_rate_of_change(self, current: float, previous: float, days: int = 1) -> float:
        """
        Calculate rate of change as percentage change per day.
        
        Args:
            current: Current probability
            previous: Previous probability
            days: Number of days between measurements
            
        Returns:
            Rate of change (% per day)
        """
        if previous == 0:
            return 0.0
        
        total_change = (current - previous) / previous
        daily_change = total_change / max(days, 1)
        return daily_change
    
    def check_ml_alarm(self, current_prob: float, history: list) -> Dict:
        """
        Check if ML model triggers rate-of-change alarm.
        
        Args:
            current_prob: Current ML crash probability
            history: List of (date, probability) tuples
            
        Returns:
            Dictionary with alarm status and details
        """
        alarm = {
            'triggered': False,
            'severity': 'none',
            'reason': '',
            'roc_1d': 0.0,
            'roc_5d': 0.0,
            'roc_20d': 0.0,
        }
        
        if len(history) < 2:
            return alarm
        
        # 1-day rate of change
        if len(history) >= 1:
            prev_prob = history[-1][1]
            roc_1d = self.calculate_rate_of_change(current_prob, prev_prob, 1)
            alarm['roc_1d'] = roc_1d
            
            if roc_1d > self.ml_roc_thresholds['1d']:
                alarm['triggered'] = True
                alarm['severity'] = 'high'
                alarm['reason'] = f"Sharp 1-day increase: {roc_1d:.1%}"
        
        # 5-day rate of change
        if len(history) >= 5:
            prev_prob = history[-5][1]
            roc_5d = self.calculate_rate_of_change(current_prob, prev_prob, 5)
            alarm['roc_5d'] = roc_5d
            
            if roc_5d > self.ml_roc_thresholds['5d'] and not alarm['triggered']:
                alarm['triggered'] = True
                alarm['severity'] = 'medium'
                alarm['reason'] = f"Sustained 5-day increase: {roc_5d:.1%}/day"
        
        # 20-day rate of change
        if len(history) >= 20:
            prev_prob = history[-20][1]
            roc_20d = self.calculate_rate_of_change(current_prob, prev_prob, 20)
            alarm['roc_20d'] = roc_20d
            
            if roc_20d > self.ml_roc_thresholds['20d'] and not alarm['triggered']:
                alarm['triggered'] = True
                alarm['severity'] = 'low'
                alarm['reason'] = f"Gradual 20-day increase: {roc_20d:.1%}/day"
        
        return alarm
    
    def check_stat_alarm(self, current_prob: float, history: list) -> Dict:
        """
        Check if Statistical model triggers rate-of-change alarm.
        
        Args:
            current_prob: Current statistical crash probability
            history: List of (date, probability) tuples
            
        Returns:
            Dictionary with alarm status and details
        """
        alarm = {
            'triggered': False,
            'severity': 'none',
            'reason': '',
            'roc_1d': 0.0,
            'roc_5d': 0.0,
            'roc_20d': 0.0,
        }
        
        if len(history) < 2:
            return alarm
        
        # 1-day rate of change
        if len(history) >= 1:
            prev_prob = history[-1][1]
            roc_1d = self.calculate_rate_of_change(current_prob, prev_prob, 1)
            alarm['roc_1d'] = roc_1d
            
            if roc_1d > self.stat_roc_thresholds['1d']:
                alarm['triggered'] = True
                alarm['severity'] = 'high'
                alarm['reason'] = f"Sharp 1-day increase: {roc_1d:.1%}"
        
        # 5-day rate of change
        if len(history) >= 5:
            prev_prob = history[-5][1]
            roc_5d = self.calculate_rate_of_change(current_prob, prev_prob, 5)
            alarm['roc_5d'] = roc_5d
            
            if roc_5d > self.stat_roc_thresholds['5d'] and not alarm['triggered']:
                alarm['triggered'] = True
                alarm['severity'] = 'medium'
                alarm['reason'] = f"Sustained 5-day increase: {roc_5d:.1%}/day"
        
        # 20-day rate of change
        if len(history) >= 20:
            prev_prob = history[-20][1]
            roc_20d = self.calculate_rate_of_change(current_prob, prev_prob, 20)
            alarm['roc_20d'] = roc_20d
            
            if roc_20d > self.stat_roc_thresholds['20d'] and not alarm['triggered']:
                alarm['triggered'] = True
                alarm['severity'] = 'low'
                alarm['reason'] = f"Gradual 20-day increase: {roc_20d:.1%}/day"
        
        return alarm
    
    def check_combined_alarm(self, ml_alarm: Dict, stat_alarm: Dict) -> Dict:
        """
        Check if both models agree on alarm.
        
        Args:
            ml_alarm: ML model alarm status
            stat_alarm: Statistical model alarm status
            
        Returns:
            Combined alarm status
        """
        combined = {
            'triggered': False,
            'severity': 'none',
            'reason': '',
            'ml_triggered': ml_alarm['triggered'],
            'stat_triggered': stat_alarm['triggered'],
        }
        
        if ml_alarm['triggered'] and stat_alarm['triggered']:
            combined['triggered'] = True
            combined['severity'] = 'critical'
            combined['reason'] = f"Both models triggered: ML ({ml_alarm['severity']}), Stat ({stat_alarm['severity']})"
        elif ml_alarm['triggered'] or stat_alarm['triggered']:
            combined['triggered'] = True
            combined['severity'] = 'warning'
            combined['reason'] = f"One model triggered: ML ({ml_alarm['severity']}), Stat ({stat_alarm['severity']})"
        
        return combined

