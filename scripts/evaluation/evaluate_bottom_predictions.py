"""
Evaluate Bottom Prediction Models on Historical Crashes

Tests how well the bottom prediction models identify market recovery points
for all 11 historical crashes.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from src.utils.database import DatabaseManager, CrashEvent, Prediction, Indicator

def evaluate_bottom_predictions():
    """Evaluate bottom prediction models on historical crashes."""

    logger.info("=" * 80)
    logger.info("EVALUATING BOTTOM PREDICTION MODELS")
    logger.info("=" * 80)

    db = DatabaseManager()
    db.create_tables()  # Ensure tables exist
    session = db.get_session()
    
    # Get crashes
    crashes = session.query(CrashEvent).all()
    logger.info(f"\nAnalyzing {len(crashes)} historical crashes:")
    logger.info("=" * 80)
    
    total_accuracy = 0
    detected_bottoms = 0
    
    for crash in crashes:
        # Get actual trough date
        trough_date = crash.trough_date
        recovery_date = crash.recovery_date
        
        # Get predictions during crash period
        crash_start = crash.start_date
        crash_end = crash.end_date
        
        # Get S&P 500 data during crash
        indicators = session.query(Indicator).filter(
            Indicator.date >= crash_start,
            Indicator.date <= crash_end
        ).order_by(Indicator.date).all()
        
        if not indicators:
            continue
        
        # Find actual bottom (minimum close price)
        prices = [ind.sp500_close for ind in indicators if ind.sp500_close]
        if not prices:
            continue
        
        min_price = min(prices)
        actual_bottom_idx = next(i for i, ind in enumerate(indicators) if ind.sp500_close == min_price)
        actual_bottom_date = indicators[actual_bottom_idx].date
        
        # Calculate days from crash start to bottom
        days_to_bottom = (actual_bottom_date - crash_start).days
        
        # Get predictions for bottom
        preds = session.query(Prediction).filter(
            Prediction.prediction_date >= crash_start,
            Prediction.prediction_date <= crash_end,
            Prediction.bottom_prediction_date.isnot(None)
        ).all()
        
        if preds:
            # Get most common predicted bottom
            predicted_bottoms = [p.bottom_prediction_date for p in preds if p.bottom_prediction_date]
            if predicted_bottoms:
                predicted_bottom = max(set(predicted_bottoms), key=predicted_bottoms.count)
                days_error = abs((predicted_bottom - actual_bottom_date).days)
                accuracy = max(0, 100 - days_error)
                detected_bottoms += 1
                total_accuracy += accuracy
                
                logger.info(f"\n{crash.start_date} ({crash.crash_type})")
                logger.info(f"  Actual bottom: {actual_bottom_date} ({days_to_bottom} days from crash)")
                logger.info(f"  Predicted bottom: {predicted_bottom}")
                logger.info(f"  Error: {days_error} days")
                logger.info(f"  Accuracy: {accuracy:.0f}%")
            else:
                logger.info(f"\n{crash.start_date} - No bottom predictions")
        else:
            logger.info(f"\n{crash.start_date} - No predictions during crash")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Crashes analyzed: {len(crashes)}")
    logger.info(f"Bottoms detected: {detected_bottoms}/{len(crashes)}")
    if detected_bottoms > 0:
        logger.info(f"Average accuracy: {total_accuracy/detected_bottoms:.0f}%")
    
    session.close()

if __name__ == '__main__':
    evaluate_bottom_predictions()

