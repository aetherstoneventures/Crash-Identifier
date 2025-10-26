"""
Comprehensive Accuracy Metrics for Crash Prediction Models

Calculates:
1. Precision, Recall, F1-Score
2. ROC-AUC and PR-AUC
3. Confusion matrix metrics
4. Calibration metrics
5. Backtesting results
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_curve, auc, brier_score_loss,
    calibration_curve
)
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AccuracyMetrics:
    """Calculate comprehensive accuracy metrics for crash prediction models."""

    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                        y_pred_proba: np.ndarray = None) -> Dict:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (binary)
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = np.mean(y_true == y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Derived metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Probability metrics
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            
            # PR-AUC
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['pr_auc'] = auc(recall_vals, precision_vals)
            
            # Brier score (calibration)
            metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
            
            # Calibration error
            prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
            metrics['calibration_error'] = np.mean(np.abs(prob_true - prob_pred))
        
        return metrics

    @staticmethod
    def calculate_backtesting_metrics(predictions_df: pd.DataFrame,
                                     market_data_df: pd.DataFrame) -> Dict:
        """
        Calculate backtesting metrics against historical crashes.
        
        Args:
            predictions_df: DataFrame with predictions and dates
            market_data_df: DataFrame with market data and crash labels
            
        Returns:
            Dictionary of backtesting metrics
        """
        metrics = {}
        
        # Merge predictions with market data
        merged = predictions_df.merge(market_data_df, on='date', how='inner')
        
        if len(merged) == 0:
            logger.warning("No overlapping dates for backtesting")
            return metrics
        
        # Calculate lead time (days before crash predicted)
        if 'crash_label' in merged.columns:
            crashes = merged[merged['crash_label'] == 1]
            
            if len(crashes) > 0:
                # Predictions before crashes
                lead_times = []
                for idx, crash in crashes.iterrows():
                    crash_date = crash['date']
                    pred_before = merged[
                        (merged['date'] < crash_date) &
                        (merged['crash_probability'] > 0.5) &
                        ((crash_date - merged['date']).dt.days <= 60)
                    ]
                    if len(pred_before) > 0:
                        lead_time = (crash_date - pred_before['date'].max()).days
                        lead_times.append(lead_time)
                
                if lead_times:
                    metrics['avg_lead_time_days'] = np.mean(lead_times)
                    metrics['min_lead_time_days'] = np.min(lead_times)
                    metrics['max_lead_time_days'] = np.max(lead_times)
                    metrics['crashes_predicted'] = len(lead_times)
                    metrics['total_crashes'] = len(crashes)
                    metrics['crash_detection_rate'] = len(lead_times) / len(crashes)
        
        # False alarm rate
        high_prob_periods = merged[merged['crash_probability'] > 0.5]
        if len(high_prob_periods) > 0:
            if 'crash_label' in merged.columns:
                false_alarms = high_prob_periods[high_prob_periods['crash_label'] == 0]
                metrics['false_alarm_rate'] = len(false_alarms) / len(high_prob_periods)
        
        return metrics

    @staticmethod
    def calculate_model_comparison(models_metrics: Dict[str, Dict]) -> pd.DataFrame:
        """
        Create comparison DataFrame for multiple models.
        
        Args:
            models_metrics: Dictionary of model names to metrics
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for model_name, metrics in models_metrics.items():
            row = {
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1-Score': metrics.get('f1', 0),
                'ROC-AUC': metrics.get('roc_auc', 0),
                'PR-AUC': metrics.get('pr_auc', 0),
                'Specificity': metrics.get('specificity', 0),
                'Sensitivity': metrics.get('sensitivity', 0),
            }
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)

    @staticmethod
    def generate_report(models_metrics: Dict[str, Dict],
                       backtesting_metrics: Dict[str, Dict] = None) -> str:
        """
        Generate comprehensive accuracy report.
        
        Args:
            models_metrics: Dictionary of model metrics
            backtesting_metrics: Dictionary of backtesting metrics
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE MODEL ACCURACY REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Model comparison
        report.append("MODEL PERFORMANCE COMPARISON")
        report.append("-" * 80)
        comparison_df = AccuracyMetrics.calculate_model_comparison(models_metrics)
        report.append(comparison_df.to_string(index=False))
        report.append("")
        
        # Detailed metrics for each model
        report.append("DETAILED METRICS BY MODEL")
        report.append("-" * 80)
        for model_name, metrics in models_metrics.items():
            report.append(f"\n{model_name}:")
            for key, value in sorted(metrics.items()):
                if isinstance(value, float):
                    report.append(f"  {key}: {value:.4f}")
                else:
                    report.append(f"  {key}: {value}")
        
        # Backtesting results
        if backtesting_metrics:
            report.append("\n" + "=" * 80)
            report.append("BACKTESTING RESULTS")
            report.append("-" * 80)
            for model_name, metrics in backtesting_metrics.items():
                report.append(f"\n{model_name}:")
                for key, value in sorted(metrics.items()):
                    if isinstance(value, float):
                        report.append(f"  {key}: {value:.4f}")
                    else:
                        report.append(f"  {key}: {value}")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)

