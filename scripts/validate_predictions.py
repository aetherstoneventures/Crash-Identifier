#!/usr/bin/env python3
"""
Prediction Validation Framework

Validates crash predictions against historical market data and reliable sources.
This script can be run anytime to verify prediction accuracy.

Usage:
    python3 scripts/validate_predictions.py
    python3 scripts/validate_predictions.py --start-date 2020-01-01 --end-date 2020-12-31
    python3 scripts/validate_predictions.py --generate-report
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import argparse
import json

from src.utils.database import DatabaseManager, Indicator, Prediction
from src.data_collection.yahoo_collector import YahooCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictionValidator:
    """Validates crash predictions against historical market data."""
    
    def __init__(self):
        """Initialize validator."""
        self.db_manager = DatabaseManager()
        self.yahoo_collector = YahooCollector()
        self.validation_results = {}
    
    def get_historical_crashes(self, start_date=None, end_date=None):
        """
        Get historical market crashes from database.
        
        A crash is defined as a 20% drawdown from peak.
        """
        session = self.db_manager.get_session()
        try:
            indicators = session.query(Indicator).order_by(Indicator.date.asc()).all()
            session.close()
            
            if not indicators:
                logger.warning("No indicators found in database")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for ind in indicators:
                data.append({
                    'date': ind.date,
                    'sp500_close': ind.sp500_close
                })
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            
            # Calculate rolling maximum (peak)
            df['peak'] = df['sp500_close'].rolling(window=252, min_periods=1).max()
            
            # Calculate drawdown
            df['drawdown'] = (df['sp500_close'] - df['peak']) / df['peak']
            
            # Identify crashes (20% drawdown)
            df['is_crash'] = df['drawdown'] <= -0.20
            
            # Filter by date range if provided
            if start_date:
                df = df[df['date'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['date'] <= pd.to_datetime(end_date)]
            
            return df
        
        except Exception as e:
            logger.error(f"Error getting historical crashes: {e}")
            session.close()
            return pd.DataFrame()
    
    def get_predictions_for_period(self, start_date=None, end_date=None):
        """Get predictions for a specific period."""
        session = self.db_manager.get_session()
        try:
            predictions = session.query(Prediction).order_by(Prediction.prediction_date.asc()).all()
            session.close()
            
            if not predictions:
                logger.warning("No predictions found in database")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for pred in predictions:
                data.append({
                    'prediction_date': pred.prediction_date,
                    'crash_probability': pred.crash_probability,
                    'confidence_lower': pred.confidence_interval_lower,
                    'confidence_upper': pred.confidence_interval_upper
                })
            
            df = pd.DataFrame(data)
            df['prediction_date'] = pd.to_datetime(df['prediction_date'])
            
            # Filter by date range if provided
            if start_date:
                df = df[df['prediction_date'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['prediction_date'] <= pd.to_datetime(end_date)]
            
            return df
        
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            session.close()
            return pd.DataFrame()
    
    def validate_crash_predictions(self, start_date=None, end_date=None):
        """
        Validate crash predictions against actual market crashes.
        
        Metrics:
        - True Positives: Predicted crash, actual crash occurred
        - True Negatives: Predicted no crash, no crash occurred
        - False Positives: Predicted crash, no crash occurred
        - False Negatives: Predicted no crash, crash occurred
        """
        logger.info("Validating crash predictions...")
        
        # Get historical crashes
        crashes_df = self.get_historical_crashes(start_date, end_date)
        if crashes_df.empty:
            logger.warning("No historical data available")
            return {}
        
        # Get predictions
        pred_df = self.get_predictions_for_period(start_date, end_date)
        if pred_df.empty:
            logger.warning("No predictions available")
            return {}
        
        # Merge on date
        merged = pd.merge_asof(
            pred_df.sort_values('prediction_date'),
            crashes_df.sort_values('date'),
            left_on='prediction_date',
            right_on='date',
            direction='backward'
        )
        
        # Calculate metrics
        threshold = 0.5  # 50% probability threshold
        
        # Predictions
        pred_crash = merged['crash_probability'] >= threshold
        
        # Actual crashes (look ahead 60 days)
        actual_crash = []
        for idx, row in merged.iterrows():
            future_data = crashes_df[
                (crashes_df['date'] > row['prediction_date']) &
                (crashes_df['date'] <= row['prediction_date'] + timedelta(days=60))
            ]
            actual_crash.append(future_data['is_crash'].any())
        
        merged['actual_crash'] = actual_crash
        
        # Calculate confusion matrix
        tp = ((pred_crash) & (merged['actual_crash'])).sum()
        tn = ((~pred_crash) & (~merged['actual_crash'])).sum()
        fp = ((pred_crash) & (~merged['actual_crash'])).sum()
        fn = ((~pred_crash) & (merged['actual_crash'])).sum()
        
        # Calculate metrics
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            'total_predictions': len(merged),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'threshold': threshold
        }
        
        self.validation_results['crash_predictions'] = results
        return results
    
    def print_validation_report(self):
        """Print validation report."""
        print("\n" + "="*80)
        print("PREDICTION VALIDATION REPORT")
        print("="*80 + "\n")
        
        if 'crash_predictions' in self.validation_results:
            results = self.validation_results['crash_predictions']
            print("CRASH PREDICTION VALIDATION")
            print("-" * 80)
            print(f"Total Predictions Analyzed: {results['total_predictions']}")
            print(f"True Positives (Correct Crash Predictions): {results['true_positives']}")
            print(f"True Negatives (Correct No-Crash Predictions): {results['true_negatives']}")
            print(f"False Positives (Incorrect Crash Predictions): {results['false_positives']}")
            print(f"False Negatives (Missed Crashes): {results['false_negatives']}")
            print()
            print(f"Accuracy:  {results['accuracy']:.1%}")
            print(f"Precision: {results['precision']:.1%}")
            print(f"Recall:    {results['recall']:.1%}")
            print(f"F1 Score:  {results['f1_score']:.3f}")
            print()
            print(f"Threshold: {results['threshold']:.0%}")
            print()
        
        print("="*80 + "\n")
    
    def save_validation_report(self, filename='validation_report.json'):
        """Save validation report to JSON file."""
        filepath = Path('data') / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {filepath}")


def main():
    """Main validation script."""
    parser = argparse.ArgumentParser(
        description='Validate crash predictions against historical market data'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate and save validation report'
    )
    
    args = parser.parse_args()
    
    # Create validator
    validator = PredictionValidator()
    
    # Run validation
    logger.info("Starting prediction validation...")
    results = validator.validate_crash_predictions(args.start_date, args.end_date)
    
    # Print report
    validator.print_validation_report()
    
    # Save report if requested
    if args.generate_report:
        validator.save_validation_report()
    
    logger.info("Validation complete!")


if __name__ == '__main__':
    main()

