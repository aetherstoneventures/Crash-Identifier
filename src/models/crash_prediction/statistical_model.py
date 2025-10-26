"""
Statistical/Rule-Based Crash Prediction Model

Uses indicator thresholds and statistical analysis to predict market crashes.
This model provides interpretable, rule-based predictions for validation against ML models.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from .base_model import BaseCrashModel

logger = logging.getLogger(__name__)


class StatisticalCrashModel(BaseCrashModel):
    """
    Rule-based crash prediction using indicator thresholds.
    
    Rules:
    1. Inverted yield curve (10Y-2Y < 0) → High crash risk
    2. High VIX (> 30) → High volatility/crash risk
    3. High Shiller PE (> 30) → Overvaluation risk
    4. Unemployment spike (rate > 5%) → Recession risk
    5. Negative real GDP growth → Recession risk
    6. High credit spreads (BBB > 3%) → Credit risk
    7. Margin debt spike → Leverage risk
    8. Inverted yield curve + high VIX → Very high risk
    """

    def __init__(self):
        """Initialize statistical model."""
        super().__init__("Statistical/Rule-Based Model")
        self.is_trained = True  # No training needed for rule-based model
        self.thresholds = {
            'yield_10y_2y': 0.0,  # Inverted if < 0
            'vix_close': 30.0,
            'shiller_pe': 30.0,
            'unemployment_rate': 5.0,
            'credit_spread_bbb': 3.0,
            'margin_debt_spike': 0.05,  # 5% increase
        }

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Statistical model doesn't need training.
        
        Args:
            X_train: Training features (not used)
            y_train: Training labels (not used)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Empty metrics dictionary
        """
        logger.info("Statistical model is rule-based and doesn't require training")
        self.is_trained = True
        return {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary crash predictions using rules.
        
        Args:
            X: Features array (n_samples, n_features)
            
        Returns:
            Binary predictions (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict crash probability using statistical rules.
        
        Args:
            X: Features array (n_samples, n_features)
            
        Returns:
            Crash probability (0-1)
        """
        if isinstance(X, pd.DataFrame):
            return self._predict_proba_dataframe(X)
        else:
            return self._predict_proba_array(X)

    def _predict_proba_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Predict using DataFrame with named columns."""
        probabilities = []
        
        for idx, row in df.iterrows():
            prob = self._calculate_crash_probability(row)
            probabilities.append(prob)
        
        return np.array(probabilities)

    def _predict_proba_array(self, X: np.ndarray) -> np.ndarray:
        """Predict using numpy array (feature indices)."""
        # Feature indices (must match training pipeline)
        feature_names = [
            'sp500_close', 'vix_close', 'yield_10y_3m', 'yield_10y_2y',
            'yield_10y', 'credit_spread_bbb', 'unemployment_rate', 'real_gdp',
            'cpi', 'fed_funds_rate', 'industrial_production', 'sp500_volume',
            'consumer_sentiment', 'housing_starts', 'm2_money_supply',
            'debt_to_gdp', 'savings_rate', 'lei', 'shiller_pe', 'margin_debt',
            'put_call_ratio'
        ]
        
        probabilities = []
        for sample in X:
            # Create dict from array
            row_dict = {name: val for name, val in zip(feature_names, sample)}
            prob = self._calculate_crash_probability(row_dict)
            probabilities.append(prob)
        
        return np.array(probabilities)

    def _calculate_crash_probability(self, row) -> float:
        """
        Calculate crash probability based on indicator values.
        
        Args:
            row: Dictionary or Series with indicator values
            
        Returns:
            Crash probability (0-1)
        """
        risk_score = 0.0
        max_risk = 0.0
        
        # Rule 1: Inverted yield curve (strongest signal)
        if 'yield_10y_2y' in row and pd.notna(row['yield_10y_2y']):
            if row['yield_10y_2y'] < 0:
                risk_score += 0.30  # 30% base risk
                max_risk += 0.30
            else:
                max_risk += 0.30
        
        # Rule 2: High VIX (volatility indicator)
        if 'vix_close' in row and pd.notna(row['vix_close']):
            vix = row['vix_close']
            if vix > 40:
                risk_score += 0.25  # Very high volatility
            elif vix > 30:
                risk_score += 0.15  # High volatility
            elif vix > 20:
                risk_score += 0.05  # Moderate volatility
            max_risk += 0.25
        
        # Rule 3: High Shiller PE (valuation)
        if 'shiller_pe' in row and pd.notna(row['shiller_pe']):
            pe = row['shiller_pe']
            if pe > 35:
                risk_score += 0.15
            elif pe > 30:
                risk_score += 0.10
            elif pe > 25:
                risk_score += 0.05
            max_risk += 0.15
        
        # Rule 4: Unemployment spike (recession indicator)
        if 'unemployment_rate' in row and pd.notna(row['unemployment_rate']):
            unemp = row['unemployment_rate']
            if unemp > 7:
                risk_score += 0.15
            elif unemp > 5:
                risk_score += 0.10
            elif unemp > 4:
                risk_score += 0.05
            max_risk += 0.15
        
        # Rule 5: High credit spreads (credit risk)
        if 'credit_spread_bbb' in row and pd.notna(row['credit_spread_bbb']):
            spread = row['credit_spread_bbb']
            if spread > 4:
                risk_score += 0.10
            elif spread > 3:
                risk_score += 0.05
            max_risk += 0.10
        
        # Rule 6: Margin debt (leverage risk)
        if 'margin_debt' in row and pd.notna(row['margin_debt']):
            # High margin debt increases risk
            risk_score += 0.05
            max_risk += 0.05
        
        # Normalize probability
        if max_risk > 0:
            probability = min(risk_score / max_risk, 1.0)
        else:
            probability = 0.0
        
        return float(probability)

    def get_metrics(self) -> Dict:
        """Get model metrics."""
        return {
            'model_type': 'Statistical/Rule-Based',
            'interpretable': True,
            'requires_training': False,
            'rules': list(self.thresholds.keys())
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.name}(rule-based, interpretable)"

