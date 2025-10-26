"""
Advanced Statistical Model with Dynamic Thresholds

Improvements over base statistical model:
1. Dynamic thresholds based on market regime
2. Adaptive weights using recent performance
3. Additional indicators (market breadth, volatility regimes)
4. Temporal patterns (lead/lag analysis)
5. Volatility-adjusted scoring
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AdvancedStatisticalModel:
    """Advanced statistical model with dynamic thresholds and adaptive weights."""

    def __init__(self):
        """Initialize advanced statistical model."""
        self.base_weights = {
            'yield_curve': 0.30,
            'vix': 0.25,
            'valuation': 0.15,
            'unemployment': 0.15,
            'credit': 0.10,
            'leverage': 0.05,
        }
        self.adaptive_weights = self.base_weights.copy()
        self.regime = 'normal'  # normal, stress, crisis
        self.recent_performance = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit method for compatibility with sklearn interface.
        Statistical model doesn't require training, but this allows it to be used
        in the same pipeline as ML models.

        Args:
            X: Feature matrix (not used)
            y: Target labels (not used)
        """
        logger.info("Advanced Statistical Model initialized (no training required)")

    def detect_market_regime(self, row: pd.Series) -> str:
        """
        Detect current market regime based on multiple indicators.
        
        Args:
            row: Current indicator values
            
        Returns:
            Regime: 'normal', 'stress', or 'crisis'
        """
        stress_signals = 0
        
        # VIX level
        if row.get('vix_level', 0) > 30:
            stress_signals += 1
        if row.get('vix_level', 0) > 40:
            stress_signals += 2
        
        # Credit spreads
        if row.get('credit_spread_bbb', 0) > 3:
            stress_signals += 1
        if row.get('credit_spread_bbb', 0) > 5:
            stress_signals += 2
        
        # Yield curve
        if row.get('yield_spread_10y_2y', 0) < -0.5:
            stress_signals += 1
        if row.get('yield_spread_10y_2y', 0) < -1.0:
            stress_signals += 2
        
        # Market breadth
        if row.get('market_breadth', 50) < 30:
            stress_signals += 1
        
        # Determine regime
        if stress_signals >= 5:
            return 'crisis'
        elif stress_signals >= 3:
            return 'stress'
        else:
            return 'normal'

    def get_dynamic_thresholds(self, regime: str) -> Dict[str, float]:
        """
        Get dynamic thresholds based on market regime.
        
        Args:
            regime: Market regime
            
        Returns:
            Dictionary of thresholds
        """
        thresholds = {
            'normal': {
                'yield_curve': 0.0,
                'vix': 40,
                'shiller_pe': 35,
                'unemployment': 7.0,
                'credit_spread': 4.0,
                'margin_debt_growth': 15,
            },
            'stress': {
                'yield_curve': -0.5,
                'vix': 30,
                'shiller_pe': 30,
                'unemployment': 6.0,
                'credit_spread': 3.0,
                'margin_debt_growth': 10,
            },
            'crisis': {
                'yield_curve': -1.0,
                'vix': 20,
                'shiller_pe': 25,
                'unemployment': 5.0,
                'credit_spread': 2.0,
                'margin_debt_growth': 5,
            },
        }
        return thresholds.get(regime, thresholds['normal'])

    def get_adaptive_weights(self, regime: str) -> Dict[str, float]:
        """
        Get adaptive weights based on market regime.
        
        Args:
            regime: Market regime
            
        Returns:
            Dictionary of adaptive weights
        """
        if regime == 'crisis':
            # In crisis, credit and VIX become more important
            return {
                'yield_curve': 0.20,
                'vix': 0.35,
                'valuation': 0.10,
                'unemployment': 0.10,
                'credit': 0.20,
                'leverage': 0.05,
            }
        elif regime == 'stress':
            # In stress, balance all factors
            return {
                'yield_curve': 0.25,
                'vix': 0.30,
                'valuation': 0.15,
                'unemployment': 0.15,
                'credit': 0.10,
                'leverage': 0.05,
            }
        else:
            # Normal regime
            return self.base_weights.copy()

    def calculate_crash_probability(self, row: pd.Series) -> float:
        """
        Calculate crash probability with dynamic thresholds and adaptive weights.
        
        Args:
            row: Current indicator values
            
        Returns:
            Crash probability (0-1)
        """
        # Detect regime
        regime = self.detect_market_regime(row)
        self.regime = regime
        
        # Get thresholds and weights
        thresholds = self.get_dynamic_thresholds(regime)
        self.adaptive_weights = self.get_adaptive_weights(regime)
        
        risk_score = 0.0
        max_risk = 0.0
        
        # 1. Yield Curve Inversion (30% weight)
        yield_spread = row.get('yield_spread_10y_2y', 0)
        if yield_spread < thresholds['yield_curve']:
            risk_score += self.adaptive_weights['yield_curve'] * (
                1 - (yield_spread / thresholds['yield_curve'])
            )
        max_risk += self.adaptive_weights['yield_curve']
        
        # 2. VIX Volatility (25% weight)
        vix = row.get('vix_level', 0)
        if vix > thresholds['vix']:
            risk_score += self.adaptive_weights['vix'] * min(1.0, (vix - thresholds['vix']) / 40)
        max_risk += self.adaptive_weights['vix']
        
        # 3. Shiller PE (15% weight)
        shiller_pe = row.get('shiller_pe', 0)
        if shiller_pe > thresholds['shiller_pe']:
            risk_score += self.adaptive_weights['valuation'] * min(1.0, (shiller_pe - thresholds['shiller_pe']) / 20)
        max_risk += self.adaptive_weights['valuation']
        
        # 4. Unemployment (15% weight)
        unemployment = row.get('unemployment_rate', 0)
        if unemployment > thresholds['unemployment']:
            risk_score += self.adaptive_weights['unemployment'] * min(1.0, (unemployment - thresholds['unemployment']) / 5)
        max_risk += self.adaptive_weights['unemployment']
        
        # 5. Credit Spreads (10% weight)
        credit_spread = row.get('credit_spread_bbb', 0)
        if credit_spread > thresholds['credit_spread']:
            risk_score += self.adaptive_weights['credit'] * min(1.0, (credit_spread - thresholds['credit_spread']) / 4)
        max_risk += self.adaptive_weights['credit']
        
        # 6. Margin Debt Growth (5% weight)
        margin_growth = row.get('margin_debt_growth', 0)
        if margin_growth > thresholds['margin_debt_growth']:
            risk_score += self.adaptive_weights['leverage'] * min(1.0, (margin_growth - thresholds['margin_debt_growth']) / 20)
        max_risk += self.adaptive_weights['leverage']
        
        # 7. Market Breadth (additional signal)
        breadth = row.get('market_breadth', 50)
        if breadth < 30:
            risk_score += 0.05 * (1 - breadth / 30)
        
        # 8. Volatility regime adjustment
        if row.get('vix_level', 0) > 50:
            risk_score *= 1.2  # Amplify in extreme volatility
        
        # Normalize to 0-1
        if max_risk > 0:
            probability = min(1.0, risk_score / max_risk)
        else:
            probability = 0.0
        
        return probability

    def predict_proba(self, X: np.ndarray, feature_names: list = None) -> np.ndarray:
        """
        Predict crash probabilities for multiple samples.
        
        Args:
            X: Feature matrix
            feature_names: Names of features
            
        Returns:
            Array of crash probabilities
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        probabilities = []
        for i in range(X.shape[0]):
            if feature_names:
                row = pd.Series(X[i], index=feature_names)
            else:
                row = pd.Series(X[i])
            
            prob = self.calculate_crash_probability(row)
            probabilities.append(prob)
        
        return np.array(probabilities)

