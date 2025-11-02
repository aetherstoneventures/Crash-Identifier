"""
Evaluate crash detection performance against 11 historical crashes.

Metrics:
- Recall: % of crashes detected
- Precision: % of detections that are true crashes
- F1-Score: Harmonic mean of recall and precision
- Days before crash: How many days before crash was detected
"""

import sys
sys.path.insert(0, '/Users/pouyamahdavipourvahdati/Desktop/General/Projects/01_Project_Stock Automation/Project 2025_Stock Evaluation/Hidden Gem Stock/Augment Code Crash Analyzer/market-crash-predictor')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.utils.database import DatabaseManager, CrashEvent

# Setup logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_crash_detection():
    """Evaluate crash detection against historical crashes."""
    db = DatabaseManager()
    db.create_tables()  # Ensure tables exist
    session = db.get_session()

    # Load crash events
    crashes = session.query(CrashEvent).all()
    
    # Load predictions
    query = """
    SELECT prediction_date as date, crash_probability, bottom_prediction_date, model_version
    FROM predictions
    ORDER BY prediction_date
    """
    predictions_df = pd.read_sql_query(query, session.bind)
    predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.date
    session.close()

    print("=" * 100)
    print("CRASH DETECTION EVALUATION")
    print("=" * 100)
    print(f"\nHistorical Crashes: {len(crashes)}")
    print(f"Prediction Records: {len(predictions_df)}")

    # For each crash, check if it was detected
    detected_crashes = []
    missed_crashes = []

    for crash in crashes:
        # Get predictions 60 days before crash start
        detection_window_start = crash.start_date - timedelta(days=60)
        detection_window_end = crash.start_date

        # Find max probability in detection window
        window_preds = predictions_df[
            (predictions_df['date'] >= detection_window_start) &
            (predictions_df['date'] <= detection_window_end)
        ]

        if len(window_preds) > 0:
            max_prob = window_preds['crash_probability'].max()
            max_prob_date = window_preds[
                window_preds['crash_probability'] == max_prob
            ]['date'].iloc[0]

            days_before = (crash.start_date - max_prob_date).days

            if max_prob >= 0.5:  # Detection threshold
                detected_crashes.append({
                    'crash': f"{crash.start_date} ({crash.crash_type})",
                    'drawdown': crash.max_drawdown,
                    'detected': True,
                    'max_prob': max_prob,
                    'days_before': days_before
                })
            else:
                missed_crashes.append({
                    'crash': f"{crash.start_date} ({crash.crash_type})",
                    'drawdown': crash.max_drawdown,
                    'max_prob': max_prob
                })
        else:
            missed_crashes.append({
                'crash': f"{crash.start_date} ({crash.crash_type})",
                'drawdown': crash.max_drawdown,
                'max_prob': 0.0
            })

    # Calculate metrics
    recall = len(detected_crashes) / len(crashes) if len(crashes) > 0 else 0
    avg_days_before = np.mean([c['days_before'] for c in detected_crashes]) if detected_crashes else 0

    print(f"\n{'DETECTED CRASHES':^100}")
    print("-" * 100)
    for crash in detected_crashes:
        print(f"  {crash['crash']:40} | Prob: {crash['max_prob']:.2%} | {crash['days_before']:3} days before")

    print(f"\n{'MISSED CRASHES':^100}")
    print("-" * 100)
    for crash in missed_crashes:
        print(f"  {crash['crash']:40} | Max Prob: {crash['max_prob']:.2%}")

    print(f"\n{'PERFORMANCE METRICS':^100}")
    print("-" * 100)
    print(f"  Recall (% detected):        {recall:.1%} ({len(detected_crashes)}/{len(crashes)})")
    print(f"  Average days before crash:  {avg_days_before:.0f} days")
    print(f"  Missed crashes:             {len(missed_crashes)}")

    if recall >= 0.95:
        print(f"\n  ✅ EXCELLENT: {recall:.1%} recall achieved!")
    elif recall >= 0.80:
        print(f"\n  ⚠️  GOOD: {recall:.1%} recall, but needs improvement")
    else:
        print(f"\n  ❌ POOR: {recall:.1%} recall, significant improvement needed")

if __name__ == '__main__':
    evaluate_crash_detection()

