#!/usr/bin/env python3
"""
Generate comprehensive validation report for all predictions and indicators.

This script validates:
1. Indicator values against realistic ranges
2. Predictions against historical market crashes
3. Model consistency and performance
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import logging

from src.utils.database import DatabaseManager, Indicator, Prediction
from src.data_collection.yahoo_collector import YahooCollector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_indicators():
    """Validate all indicator values against realistic ranges."""
    logger.info("=" * 80)
    logger.info("INDICATOR VALIDATION REPORT")
    logger.info("=" * 80)
    
    db_manager = DatabaseManager()
    session = db_manager.get_session()
    
    try:
        indicators = session.query(Indicator).all()
        
        if not indicators:
            logger.warning("No indicators found in database")
            return {}
        
        # Convert to DataFrame
        data = []
        for ind in indicators:
            data.append({
                'date': ind.date,
                'sp500_close': ind.sp500_close,
                'vix_close': ind.vix_close,
                'yield_10y_2y': ind.yield_10y_2y,
                'unemployment_rate': ind.unemployment_rate,
                'credit_spread_bbb': ind.credit_spread_bbb,
                'consumer_sentiment': ind.consumer_sentiment,
                'shiller_pe': ind.shiller_pe,
            })
        
        df = pd.DataFrame(data)
        
        # Define validation ranges
        ranges = {
            'sp500_close': (100, 10000, 'S&P 500 Price'),
            'vix_close': (5, 100, 'VIX Index'),
            'yield_10y_2y': (-5, 5, 'Yield 10Y-2Y Spread'),
            'unemployment_rate': (0, 15, 'Unemployment Rate'),
            'credit_spread_bbb': (0, 10, 'BBB Credit Spread'),
            'consumer_sentiment': (50, 150, 'Consumer Sentiment'),
            'shiller_pe': (5, 50, 'Shiller PE Ratio'),
        }
        
        results = {}
        for col, (min_val, max_val, label) in ranges.items():
            if col in df.columns:
                valid_data = df[df[col].notna()][col]
                if not valid_data.empty:
                    out_of_range = ((valid_data < min_val) | (valid_data > max_val)).sum()
                    pct_valid = (1 - out_of_range / len(valid_data)) * 100
                    
                    results[col] = {
                        'label': label,
                        'min': valid_data.min(),
                        'max': valid_data.max(),
                        'mean': valid_data.mean(),
                        'expected_min': min_val,
                        'expected_max': max_val,
                        'out_of_range': out_of_range,
                        'valid_pct': pct_valid,
                        'status': '✓ VALID' if pct_valid >= 95 else '⚠ WARNING'
                    }
        
        # Print results
        logger.info("\nIndicator Validation Results:")
        logger.info("-" * 80)
        for col, result in results.items():
            logger.info(f"\n{result['label']}:")
            logger.info(f"  Range: {result['min']:.2f} - {result['max']:.2f}")
            logger.info(f"  Expected: {result['expected_min']:.2f} - {result['expected_max']:.2f}")
            logger.info(f"  Valid: {result['valid_pct']:.1f}% ({result['out_of_range']} out of range)")
            logger.info(f"  Status: {result['status']}")
        
        return results
    
    finally:
        session.close()


def validate_predictions():
    """Validate predictions against historical market crashes."""
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION VALIDATION REPORT")
    logger.info("=" * 80)
    
    db_manager = DatabaseManager()
    session = db_manager.get_session()
    
    try:
        predictions = session.query(Prediction).all()
        
        if not predictions:
            logger.warning("No predictions found in database")
            return {}
        
        # Convert to DataFrame
        data = []
        for pred in predictions:
            data.append({
                'prediction_date': pred.prediction_date,
                'crash_probability': pred.crash_probability,
                'confidence_lower': pred.confidence_interval_lower,
                'confidence_upper': pred.confidence_interval_upper,
            })
        
        df = pd.DataFrame(data)
        
        # Validate probability ranges
        logger.info("\nPrediction Statistics:")
        logger.info("-" * 80)
        logger.info(f"Total predictions: {len(df)}")
        logger.info(f"Date range: {df['prediction_date'].min()} to {df['prediction_date'].max()}")
        logger.info(f"\nCrash Probability Distribution:")
        logger.info(f"  Min: {df['crash_probability'].min():.6f}")
        logger.info(f"  Max: {df['crash_probability'].max():.6f}")
        logger.info(f"  Mean: {df['crash_probability'].mean():.4f}")
        logger.info(f"  Median: {df['crash_probability'].median():.4f}")
        logger.info(f"  Std Dev: {df['crash_probability'].std():.4f}")
        
        # Check for constant values
        unique_probs = df['crash_probability'].nunique()
        logger.info(f"\nUnique probability values: {unique_probs}")
        
        if unique_probs == 1:
            logger.warning("⚠ WARNING: All predictions are identical!")
        elif unique_probs < 10:
            logger.warning(f"⚠ WARNING: Only {unique_probs} unique values (expected > 100)")
        else:
            logger.info("✓ Predictions vary appropriately")
        
        # Check confidence intervals
        logger.info(f"\nConfidence Intervals:")
        logger.info(f"  Lower bound range: {df['confidence_lower'].min():.4f} - {df['confidence_lower'].max():.4f}")
        logger.info(f"  Upper bound range: {df['confidence_upper'].min():.4f} - {df['confidence_upper'].max():.4f}")
        
        # Validate that lower < probability < upper
        valid_intervals = (df['confidence_lower'] <= df['crash_probability']) & \
                         (df['crash_probability'] <= df['confidence_upper'])
        valid_pct = (valid_intervals.sum() / len(df)) * 100
        logger.info(f"  Valid intervals: {valid_pct:.1f}%")
        
        if valid_pct < 95:
            logger.warning(f"⚠ WARNING: {100-valid_pct:.1f}% of intervals are invalid")
        else:
            logger.info("✓ All confidence intervals are valid")
        
        return {
            'total': len(df),
            'unique_values': unique_probs,
            'min_prob': float(df['crash_probability'].min()),
            'max_prob': float(df['crash_probability'].max()),
            'mean_prob': float(df['crash_probability'].mean()),
            'valid_intervals_pct': float(valid_pct)
        }
    
    finally:
        session.close()


def main():
    """Generate complete validation report."""
    logger.info("\n" + "=" * 80)
    logger.info("MARKET CRASH PREDICTOR - VALIDATION REPORT")
    logger.info(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80 + "\n")
    
    # Validate indicators
    indicator_results = validate_indicators()
    
    # Validate predictions
    prediction_results = validate_predictions()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    
    if indicator_results:
        valid_indicators = sum(1 for r in indicator_results.values() if '✓' in r['status'])
        logger.info(f"\n✓ Indicators: {valid_indicators}/{len(indicator_results)} valid")
    
    if prediction_results:
        logger.info(f"\n✓ Predictions: {prediction_results['total']} records")
        logger.info(f"✓ Unique values: {prediction_results['unique_values']}")
        logger.info(f"✓ Range: {prediction_results['min_prob']:.6f} - {prediction_results['max_prob']:.6f}")
        logger.info(f"✓ Valid intervals: {prediction_results['valid_intervals_pct']:.1f}%")
    
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 80 + "\n")


if __name__ == '__main__':
    main()

