"""Enhanced Statistical Crash Model V3 - Hybrid Approach Foundation.

This version includes:
1. Multi-threshold logic based on crash severity (5%, 10%, 15%, 20%, 30%)
2. Weighted indicator scoring with dynamic weights
3. Volatility regime detection (low/normal/high/extreme)
4. Crash severity classification (Minor/Moderate/Major/Severe/Extreme)
5. Interpretable factor breakdown for each prediction
6. Designed to work with ML refinement layer (hybrid approach)
"""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import os

from src.models.crash_prediction.base_model import BaseCrashModel

logger = logging.getLogger(__name__)


class StatisticalModelV3(BaseCrashModel):
    """Enhanced statistical model with multi-threshold and regime awareness."""

    def __init__(self):
        super().__init__(name="Statistical Model V3")

        # Get crash threshold from environment (default 5%)
        self.crash_threshold = float(os.getenv('CRASH_THRESHOLD', '5.0'))

        # Multi-level thresholds for different crash severities
        self.severity_thresholds = {
            'minor': 5.0,      # 5-10% drawdown
            'moderate': 10.0,  # 10-15% drawdown
            'major': 15.0,     # 15-20% drawdown
            'severe': 20.0,    # 20-30% drawdown
            'extreme': 30.0    # >30% drawdown
        }

        # Volatility regime thresholds (VIX-based)
        self.volatility_regimes = {
            'low': (0, 15),      # VIX < 15
            'normal': (15, 25),  # VIX 15-25
            'high': (25, 35),    # VIX 25-35
            'extreme': (35, 100) # VIX > 35
        }

        # Dynamic risk weights (adjusted by regime)
        self.base_weights = {
            'yield_curve': 0.14,
            'volatility': 0.12,
            'credit_stress': 0.12,
            'hy_credit': 0.10,
            'economic': 0.09,
            'labor_market': 0.07,
            'market_momentum': 0.05,
            'sentiment': 0.03,
            'financial_conditions': 0.14,  # NFCI — comprehensive stress
            'momentum_shock': 0.14          # Sudden moves: Black Monday/COVID
        }

        # Regime-specific weight adjustments
        self.regime_weight_multipliers = {
            'low': {
                'yield_curve': 1.2,  # More weight on fundamentals in calm markets
                'volatility': 0.8,
                'credit_stress': 1.1,
                'hy_credit': 1.0,
                'economic': 1.1,
                'labor_market': 1.0,
                'market_momentum': 0.9,
                'sentiment': 0.9,
                'financial_conditions': 1.0,
                'momentum_shock': 1.2  # Shock score matters even in calm markets
            },
            'normal': {
                'yield_curve': 1.0,
                'volatility': 1.0,
                'credit_stress': 1.0,
                'hy_credit': 1.0,
                'economic': 1.0,
                'labor_market': 1.0,
                'market_momentum': 1.0,
                'sentiment': 1.0,
                'financial_conditions': 1.0,
                'momentum_shock': 1.0
            },
            'high': {
                'yield_curve': 0.9,
                'volatility': 1.3,  # More weight on volatility
                'credit_stress': 1.2,
                'hy_credit': 1.2,
                'economic': 0.9,
                'labor_market': 1.0,
                'market_momentum': 1.1,
                'sentiment': 1.0,
                'financial_conditions': 1.3,
                'momentum_shock': 1.2
            },
            'extreme': {
                'yield_curve': 0.7,
                'volatility': 1.4,
                'credit_stress': 1.2,
                'hy_credit': 1.5,
                'economic': 0.8,
                'labor_market': 1.1,
                'market_momentum': 1.2,
                'sentiment': 1.0,
                'financial_conditions': 1.4,
                'momentum_shock': 1.5  # Shock is critical in extreme regimes
            }
        }

        # Indicator thresholds (calibrated from historical data)
        self.thresholds = {
            # Yield curve
            'yield_10y_2y_inversion': 0.0,
            'yield_10y_3m_inversion': 0.0,
            'yield_10y_2y_deep_inversion': -0.5,

            # Volatility
            'vix_low': 15.0,
            'vix_normal': 25.0,
            'vix_high': 35.0,
            'vix_extreme': 45.0,
            'vix_spike_1d': 0.20,  # 20% daily increase
            'vix_spike_5d': 0.50,  # 50% 5-day increase

            # Credit stress
            'credit_spread_normal': 2.0,
            'credit_spread_elevated': 3.0,
            'credit_spread_high': 4.0,
            'credit_spread_extreme': 5.0,
            'credit_spread_spike': 0.5,

            # Economic
            'unemployment_rising_threshold': 0.3,  # 0.3% increase
            'unemployment_high': 5.0,
            'unemployment_very_high': 7.0,
            'industrial_prod_decline': -0.02,

            # Market momentum
            'drawdown_minor': -0.05,
            'drawdown_moderate': -0.10,
            'drawdown_major': -0.15,
            'drawdown_severe': -0.20,
            'sp500_decline_5d': -0.05,
            'sp500_decline_20d': -0.10,

            # Sentiment
            'sentiment_normal': 80.0,
            'sentiment_low': 70.0,
            'sentiment_very_low': 60.0,
            'sentiment_extreme_low': 50.0
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Statistical model doesn't need training, but calibrates thresholds."""
        logger.info("Statistical Model V3 - Calibrating thresholds from historical data")

        if X_train is not None and len(X_train) > 0:
            self._calibrate_thresholds(X_train, y_train)

        self.is_trained = True
        return {
            'model_type': 'Statistical V3',
            'calibrated': True,
            'crash_threshold': self.crash_threshold
        }

    def _calibrate_thresholds(self, X: pd.DataFrame, y: pd.Series):
        """Calibrate thresholds based on historical crash periods."""
        logger.info("Calibrating thresholds from historical data...")

        # Identify crash periods
        crash_data = X[y == 1]
        normal_data = X[y == 0]

        if len(crash_data) == 0:
            logger.warning("No crash data available for calibration")
            return

        # Calibrate VIX thresholds
        if 'vix_level' in crash_data.columns:
            crash_vix_median = crash_data['vix_level'].median()
            crash_vix_75th = crash_data['vix_level'].quantile(0.75)
            normal_vix_median = normal_data['vix_level'].median()

            self.thresholds['vix_normal'] = max(20.0, normal_vix_median * 1.2)
            self.thresholds['vix_high'] = max(30.0, crash_vix_median * 0.9)
            self.thresholds['vix_extreme'] = max(40.0, crash_vix_75th)

            logger.info(f"  VIX thresholds: normal={self.thresholds['vix_normal']:.1f}, "
                       f"high={self.thresholds['vix_high']:.1f}, "
                       f"extreme={self.thresholds['vix_extreme']:.1f}")

        # Calibrate credit spread thresholds
        if 'credit_spread_bbb' in crash_data.columns:
            crash_spread_median = crash_data['credit_spread_bbb'].median()
            crash_spread_75th = crash_data['credit_spread_bbb'].quantile(0.75)

            self.thresholds['credit_spread_elevated'] = max(2.5, crash_spread_median * 0.8)
            self.thresholds['credit_spread_high'] = max(3.5, crash_spread_median)
            self.thresholds['credit_spread_extreme'] = max(4.5, crash_spread_75th)

            logger.info(f"  Credit spread thresholds: elevated={self.thresholds['credit_spread_elevated']:.2f}, "
                       f"high={self.thresholds['credit_spread_high']:.2f}, "
                       f"extreme={self.thresholds['credit_spread_extreme']:.2f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary crash predictions."""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict crash probability using enhanced statistical rules."""
        probabilities = []

        for idx, row in X.iterrows():
            prob, _ = self._calculate_crash_probability_with_factors(row)
            probabilities.append(prob)

        return np.array(probabilities)

    def predict_with_explanation(self, X: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Predict with detailed factor breakdown for interpretability."""
        probabilities = []
        explanations = []

        for idx, row in X.iterrows():
            prob, factors = self._calculate_crash_probability_with_factors(row)
            probabilities.append(prob)
            explanations.append(factors)

        return np.array(probabilities), pd.DataFrame(explanations)

    def _detect_volatility_regime(self, row: pd.Series) -> str:
        """Detect current volatility regime based on VIX."""
        vix = row.get('vix_level', 20.0)

        if pd.isna(vix):
            return 'normal'

        for regime, (low, high) in self.volatility_regimes.items():
            if low <= vix < high:
                return regime

        return 'extreme'

    def _get_regime_adjusted_weights(self, regime: str) -> Dict[str, float]:
        """Get risk weights adjusted for current volatility regime."""
        multipliers = self.regime_weight_multipliers.get(regime, self.regime_weight_multipliers['normal'])

        adjusted_weights = {}
        total_weight = 0.0

        for factor, base_weight in self.base_weights.items():
            adjusted_weights[factor] = base_weight * multipliers.get(factor, 1.0)
            total_weight += adjusted_weights[factor]

        # Normalize to sum to 1.0
        for factor in adjusted_weights:
            adjusted_weights[factor] /= total_weight

        return adjusted_weights

    def _calculate_crash_probability_with_factors(self, row: pd.Series) -> Tuple[float, Dict[str, Any]]:
        """Calculate crash probability with detailed factor breakdown."""

        # Detect volatility regime
        regime = self._detect_volatility_regime(row)
        weights = self._get_regime_adjusted_weights(regime)

        factor_scores = {}

        # 1. YIELD CURVE ANALYSIS
        yield_score = self._calculate_yield_curve_score(row)
        factor_scores['yield_curve'] = yield_score

        # 2. VOLATILITY ANALYSIS
        volatility_score = self._calculate_volatility_score(row)
        factor_scores['volatility'] = volatility_score

        # 3. CREDIT STRESS ANALYSIS
        credit_score = self._calculate_credit_stress_score(row)
        factor_scores['credit_stress'] = credit_score

        # 4. ECONOMIC INDICATORS
        economic_score = self._calculate_economic_score(row)
        factor_scores['economic'] = economic_score

        # 5. MARKET MOMENTUM
        momentum_score = self._calculate_momentum_score(row)
        factor_scores['market_momentum'] = momentum_score

        # 6. SENTIMENT
        sentiment_score = self._calculate_sentiment_score(row)
        factor_scores['sentiment'] = sentiment_score

        # 7. HIGH YIELD CREDIT STRESS (new — more sensitive than IG)
        hy_score = self._calculate_hy_credit_score(row)
        factor_scores['hy_credit'] = hy_score

        # 8. LABOR MARKET (weekly initial claims + recession prob)
        labor_score = self._calculate_labor_market_score(row)
        factor_scores['labor_market'] = labor_score

        # 9. FINANCIAL CONDITIONS (NFCI — comprehensive stress index from Fed)
        fin_cond_score = self._calculate_financial_conditions_score(row)
        factor_scores['financial_conditions'] = fin_cond_score

        # 10. MOMENTUM SHOCK (sudden large moves — catches Black Monday / COVID style)
        shock_score = self._calculate_momentum_shock_score(row)
        factor_scores['momentum_shock'] = shock_score

        # Calculate weighted risk score
        total_risk = sum(
            factor_scores[factor] * weights[factor]
            for factor in factor_scores
        )

        # Apply non-linear boost for multiple high-risk factors
        high_risk_count = sum(1 for score in factor_scores.values() if score > 0.7)
        if high_risk_count >= 3:
            total_risk = min(total_risk * 1.4, 1.0)  # 40% boost
        elif high_risk_count >= 2:
            total_risk = min(total_risk * 1.2, 1.0)  # 20% boost

        # Prepare explanation
        explanation = {
            'total_risk': total_risk,
            'regime': regime,
            **{f'{factor}_score': score for factor, score in factor_scores.items()},
            **{f'{factor}_weight': weights[factor] for factor in factor_scores},
            'high_risk_factors': high_risk_count
        }

        return float(total_risk), explanation

    def _calculate_yield_curve_score(self, row: pd.Series) -> float:
        """Calculate yield curve risk score including velocity (flattening speed)."""
        score = 0.0

        # 10Y-2Y spread level
        if 'yield_spread_10y_2y' in row and pd.notna(row['yield_spread_10y_2y']):
            spread_2y = row['yield_spread_10y_2y']
            if spread_2y < self.thresholds['yield_10y_2y_deep_inversion']:
                score += 0.7  # Deep inversion — very strong signal
            elif spread_2y < self.thresholds['yield_10y_2y_inversion']:
                score += 0.5  # Inversion — strong signal
            elif spread_2y < 0.25:
                score += 0.15  # Near-flat (dangerous zone — as in 2018)

        # 10Y-3M spread level
        if 'yield_spread_10y_3m' in row and pd.notna(row['yield_spread_10y_3m']):
            spread_3m = row['yield_spread_10y_3m']
            if spread_3m < self.thresholds['yield_10y_3m_inversion']:
                score += 0.5  # Additional confirmation
            elif spread_3m < 0.30:
                score += 0.15  # Near-flat

        # YIELD CURVE VELOCITY: how fast is it flattening? (catches 2018 style)
        # 63-day velocity (3-month flattening rate)
        if 'yield_curve_velocity_63d' in row and pd.notna(row.get('yield_curve_velocity_63d')):
            vel_63d = row['yield_curve_velocity_63d']
            if vel_63d < -0.40:
                score += 0.5  # Very rapid flattening (>40bp in 3 months)
            elif vel_63d < -0.25:
                score += 0.3  # Rapid flattening (25-40bp in 3 months)
            elif vel_63d < -0.15:
                score += 0.15  # Moderate flattening

        # 120-day velocity (6-month — slower trend but more persistent)
        if 'yield_curve_velocity_120d' in row and pd.notna(row.get('yield_curve_velocity_120d')):
            vel_120d = row['yield_curve_velocity_120d']
            if vel_120d < -0.60:
                score += 0.3  # Persistent flattening trend
            elif vel_120d < -0.35:
                score += 0.15

        return min(score, 1.0)

    def _calculate_volatility_score(self, row: pd.Series) -> float:
        """Calculate volatility risk score."""
        score = 0.0

        # VIX level
        if 'vix_level' in row and pd.notna(row['vix_level']):
            vix = row['vix_level']
            if vix > self.thresholds['vix_extreme']:
                score += 0.8
            elif vix > self.thresholds['vix_high']:
                score += 0.6
            elif vix > self.thresholds['vix_normal']:
                score += 0.3

        # VIX change (20-day)
        if 'vix_change_20d' in row and pd.notna(row['vix_change_20d']):
            if row['vix_change_20d'] > self.thresholds['vix_spike_5d']:
                score += 0.5  # Large VIX increase over 20 days

        return min(score, 1.0)

    def _calculate_credit_stress_score(self, row: pd.Series) -> float:
        """Calculate credit stress risk score."""
        score = 0.0

        # Credit spread level
        if 'credit_spread_bbb' in row and pd.notna(row['credit_spread_bbb']):
            spread = row['credit_spread_bbb']
            if spread > self.thresholds['credit_spread_extreme']:
                score += 0.8
            elif spread > self.thresholds['credit_spread_high']:
                score += 0.6
            elif spread > self.thresholds['credit_spread_elevated']:
                score += 0.3

        # Credit spread widening (20-day change)
        if 'credit_spread_change_20d' in row and pd.notna(row['credit_spread_change_20d']):
            if row['credit_spread_change_20d'] > self.thresholds['credit_spread_spike']:
                score += 0.4

        return min(score, 1.0)

    def _calculate_economic_score(self, row: pd.Series) -> float:
        """Calculate economic deterioration risk score."""
        score = 0.0

        # Unemployment level
        if 'unemployment_rate' in row and pd.notna(row['unemployment_rate']):
            unemp = row['unemployment_rate']
            if unemp > self.thresholds['unemployment_very_high']:
                score += 0.5
            elif unemp > self.thresholds['unemployment_high']:
                score += 0.3

        # Unemployment rising (Sahm Rule: 3-mo avg minus 12-mo low)
        if 'sahm_rule' in row and pd.notna(row['sahm_rule']):
            if row['sahm_rule'] > self.thresholds['unemployment_rising_threshold']:
                score += 0.3

        # Industrial production declining (YoY growth)
        if 'industrial_prod_growth_yoy' in row and pd.notna(row['industrial_prod_growth_yoy']):
            if row['industrial_prod_growth_yoy'] < self.thresholds['industrial_prod_decline']:
                score += 0.3

        return min(score, 1.0)

    def _calculate_momentum_score(self, row: pd.Series) -> float:
        """Calculate market momentum risk score."""
        score = 0.0

        # Drawdown level
        if 'sp500_drawdown' in row and pd.notna(row['sp500_drawdown']):
            dd = row['sp500_drawdown']
            if dd < self.thresholds['drawdown_severe']:
                score += 0.8
            elif dd < self.thresholds['drawdown_major']:
                score += 0.6
            elif dd < self.thresholds['drawdown_moderate']:
                score += 0.4
            elif dd < self.thresholds['drawdown_minor']:
                score += 0.2

        # Short-term decline (5-day)
        if 'sp500_return_5d' in row and pd.notna(row['sp500_return_5d']):
            if row['sp500_return_5d'] < self.thresholds['sp500_decline_5d']:
                score += 0.2

        # Medium-term decline (20-day)
        if 'sp500_return_20d' in row and pd.notna(row['sp500_return_20d']):
            if row['sp500_return_20d'] < self.thresholds['sp500_decline_20d']:
                score += 0.3

        return min(score, 1.0)


    def _calculate_hy_credit_score(self, row: pd.Series) -> float:
        """Calculate High Yield credit stress score (more sensitive than IG spreads)."""
        score = 0.0
        # HY spread level (US HY typically 300-400bp normal, 600+ elevated, 900+ crisis)
        if 'hy_spread' in row and pd.notna(row.get('hy_spread')):
            hy = row['hy_spread']
            if hy > 9.0:
                score += 0.9   # Crisis level (2008: 20%, COVID peak: 11%)
            elif hy > 6.0:
                score += 0.7   # High stress
            elif hy > 4.5:
                score += 0.4   # Elevated
            elif hy > 3.5:
                score += 0.2   # Slightly elevated

        # HY spread widening (20-day change) — fast-moving signal
        if 'hy_spread_change_20d' in row and pd.notna(row.get('hy_spread_change_20d')):
            chg = row['hy_spread_change_20d']
            if chg > 2.0:
                score += 0.6   # Rapid widening (crisis onset)
            elif chg > 1.0:
                score += 0.4
            elif chg > 0.5:
                score += 0.2

        # HY-IG divergence: when HY widens much faster than IG, liquidity risk rises
        if ('hy_spread' in row and 'credit_spread_bbb' in row and
                pd.notna(row.get('hy_spread')) and pd.notna(row.get('credit_spread_bbb'))):
            divergence = row['hy_spread'] / max(row['credit_spread_bbb'], 0.1)
            if divergence > 5.5:
                score += 0.3   # Unusual bifurcation: junk >> investment grade

        # NY Fed recession probability (12-month ahead)
        if 'recession_prob' in row and pd.notna(row.get('recession_prob')):
            rp = row['recession_prob']
            if rp > 50:
                score += 0.5
            elif rp > 30:
                score += 0.3
            elif rp > 15:
                score += 0.1

        # Economic Policy Uncertainty
        if 'epu_index' in row and pd.notna(row.get('epu_index')):
            epu = row['epu_index']
            if epu > 300:
                score += 0.4   # Extreme uncertainty (2020 COVID, 2008 peak ~500)
            elif epu > 200:
                score += 0.2
            elif epu > 150:
                score += 0.1

        # EPU acceleration (rapid increase in uncertainty — catches trade war onset)
        if 'epu_acceleration' in row and pd.notna(row.get('epu_acceleration')):
            acc = row['epu_acceleration']
            if acc > 100:
                score += 0.3  # EPU jumped 100+ points in 30 days
            elif acc > 50:
                score += 0.15

        # EPU sustained elevated (90-day MA)
        if 'epu_ma_90d' in row and pd.notna(row.get('epu_ma_90d')):
            epu_90 = row['epu_ma_90d']
            if epu_90 > 200:
                score += 0.2  # Sustained high uncertainty

        return min(score, 1.0)

    def _calculate_labor_market_score(self, row: pd.Series) -> float:
        """Calculate labor market stress score using initial claims + Sahm rule."""
        score = 0.0

        # Weekly initial claims level
        if 'initial_claims' in row and pd.notna(row.get('initial_claims')):
            claims = row['initial_claims']
            # Normal ~200-250K; Concern >310K; Crisis >400K (2020 peak: 6.9M)
            if claims > 400000:
                score += 0.8
            elif claims > 310000:
                score += 0.5
            elif claims > 270000:
                score += 0.2

        # 13-week change in initial claims (acceleration signal)
        if 'initial_claims_change_13w' in row and pd.notna(row.get('initial_claims_change_13w')):
            chg = row['initial_claims_change_13w']
            if chg > 0.50:
                score += 0.5   # Claims up 50%+ in 13 weeks — strongly recessionary
            elif chg > 0.25:
                score += 0.3
            elif chg > 0.10:
                score += 0.1

        # Sahm rule (if available)
        if 'sahm_rule' in row and pd.notna(row.get('sahm_rule')):
            if row['sahm_rule'] >= 0.5:
                score += 0.4   # Official Sahm trigger
            elif row['sahm_rule'] >= 0.3:
                score += 0.2

        return min(score, 1.0)


    def _calculate_sentiment_score(self, row: pd.Series) -> float:
        """Calculate sentiment risk score."""
        score = 0.0

        if 'consumer_sentiment' in row and pd.notna(row['consumer_sentiment']):
            sentiment = row['consumer_sentiment']
            if sentiment < self.thresholds['sentiment_extreme_low']:
                score = 0.8
            elif sentiment < self.thresholds['sentiment_very_low']:
                score = 0.6
            elif sentiment < self.thresholds['sentiment_low']:
                score = 0.3

        return score

    def _calculate_financial_conditions_score(self, row: pd.Series) -> float:
        """Score based on NFCI (National Financial Conditions Index).

        NFCI thresholds:
        - < -0.5 : Very loose (low risk)
        - -0.5 to 0: Normal
        - 0 to 0.5: Mildly tight
        - 0.5 to 1.0: Elevated stress
        - > 1.0: Severe (2008 GFC peak ~4, COVID peak ~3)
        """
        score = 0.0

        if 'nfci' in row and pd.notna(row.get('nfci')):
            nfci = row['nfci']
            if nfci > 1.5:
                score += 0.8
            elif nfci > 0.8:
                score += 0.6
            elif nfci > 0.3:
                score += 0.3
            elif nfci > 0:
                score += 0.1

        if 'anfci' in row and pd.notna(row.get('anfci')):
            anfci = row['anfci']  # Adjusted — removes economic cycle influence
            if anfci > 1.0:
                score += 0.3
            elif anfci > 0.5:
                score += 0.15

        if 'stress_composite' in row and pd.notna(row.get('stress_composite')):
            sc = row['stress_composite']
            if sc > 2.0:
                score += 0.5
            elif sc > 1.0:
                score += 0.3
            elif sc > 0.5:
                score += 0.1

        return min(score, 1.0)

    def _calculate_momentum_shock_score(self, row: pd.Series) -> float:
        """Score based on sudden large price moves — catches Black Monday / COVID.

        This is a reactive signal, not leading. But it prevents missing crashes
        that have no macro precursor (exogenous shocks). The score fires hard
        when multiple momentum signals trigger simultaneously.
        """
        score = 0.0

        # Large absolute 5-day SP500 move (magnitude — rises on both crashes and spikes)
        if 'sp500_shock_5d' in row and pd.notna(row.get('sp500_shock_5d')):
            shock = row['sp500_shock_5d']
            if shock > 0.08:
                score += 0.8   # >8% in 5 days: crisis level
            elif shock > 0.05:
                score += 0.5   # >5% in 5 days: major shock
            elif shock > 0.03:
                score += 0.2   # >3% in 5 days: notable

        # VIX acceleration (rapid increase = panic onset)
        if 'vix_momentum_5d' in row and pd.notna(row.get('vix_momentum_5d')):
            vix_acc = row['vix_momentum_5d']
            if vix_acc > 0.5:
                score += 0.4   # VIX up 50%+ in 5 days
            elif vix_acc > 0.25:
                score += 0.2

        # Credit spread rapid widening
        if 'credit_momentum_5d' in row and pd.notna(row.get('credit_momentum_5d')):
            cs_acc = row['credit_momentum_5d']
            if cs_acc > 0.5:
                score += 0.3
            elif cs_acc > 0.2:
                score += 0.15

        return min(score, 1.0)

    def classify_crash_severity(self, probability: float) -> str:
        """Classify crash severity based on probability."""
        if probability >= 0.8:
            return 'extreme'
        elif probability >= 0.6:
            return 'severe'
        elif probability >= 0.4:
            return 'major'
        elif probability >= 0.25:
            return 'moderate'
        elif probability >= 0.15:
            return 'minor'
        else:
            return 'none'

