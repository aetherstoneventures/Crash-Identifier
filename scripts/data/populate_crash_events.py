"""Populate crash_events table with DYNAMICALLY DETECTED crashes/corrections.

PARADIGM SHIFT: Dynamic Crash Detection
========================================
Instead of hardcoding 11 historical crashes, this script AUTOMATICALLY DETECTS
all market corrections/crashes from S&P 500 data based on user-defined threshold.

ALGORITHM: Peak-to-Trough Drawdown Detection
---------------------------------------------
1. Load the resolved equity-index price series (indicators table)
2. Calculate rolling maximum (peak) for each day
3. Calculate drawdown from peak: (current_price - peak) / peak * 100
4. Detect when drawdown exceeds CRASH_THRESHOLD (from .env)
5. Find trough (lowest point) during drawdown period
6. Find recovery (when price returns to pre-crash peak)
7. Classify crash severity based on max drawdown

CRASH CLASSIFICATION:
---------------------
- Minor Correction: 5-10% drawdown
- Moderate Correction: 10-15% drawdown
- Major Correction: 15-20% drawdown
- Severe Crash: 20-30% drawdown
- Extreme Crash: >30% drawdown

BENEFITS:
---------
- Larger training dataset (50-200+ events vs 11 hardcoded)
- User-configurable threshold (5% default for early warnings)
- Automatic updates as new data arrives
- Mathematically rigorous peak-to-trough detection
- No manual date entry or human bias

CONFIGURATION:
--------------
Set CRASH_THRESHOLD in .env file (default: 5.0%)
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.database import DatabaseManager, CrashEvent, Indicator

# Load environment variables
load_dotenv()

# Get crash threshold from environment (default 5%)
CRASH_THRESHOLD = float(os.getenv('CRASH_THRESHOLD', 5.0))

# Minimum duration for a crash (days) - prevents noise
MIN_CRASH_DURATION = 5

# Minimum recovery duration (days) - ensures crash is over
MIN_RECOVERY_DURATION = 10


def classify_crash_severity(max_drawdown_pct):
    """Classify crash based on maximum drawdown percentage.
    
    Args:
        max_drawdown_pct: Maximum drawdown as positive percentage (e.g., 15.5 for -15.5%)
    
    Returns:
        str: Crash classification
    """
    abs_dd = abs(max_drawdown_pct)
    
    if abs_dd >= 30:
        return 'Extreme Crash'
    elif abs_dd >= 20:
        return 'Severe Crash'
    elif abs_dd >= 15:
        return 'Major Correction'
    elif abs_dd >= 10:
        return 'Moderate Correction'
    else:
        return 'Minor Correction'


def detect_crashes_from_data(df, threshold_pct=5.0):
    """Detect all crashes/corrections from index price data using peak-to-trough algorithm.
    
    Algorithm:
    1. Calculate cumulative maximum (running peak)
    2. Calculate drawdown from peak
    3. Identify periods where drawdown exceeds threshold
    4. Find start (peak), trough (bottom), and recovery dates
    5. Calculate crash statistics
    
    Args:
        df: DataFrame with columns ['date', 'price']
        threshold_pct: Minimum drawdown percentage to classify as crash (default 5%)
    
    Returns:
        List of crash dictionaries with keys:
            - start_date: Peak before crash
            - trough_date: Lowest point
            - recovery_date: Return to pre-crash level
            - max_drawdown: Maximum drawdown percentage
            - duration_days: Days from start to trough
            - recovery_days: Days from trough to recovery
            - crash_type: Classification based on severity
    """
    # Sort by date and remove NaN prices
    df = df.sort_values('date').copy()
    df = df[df['price'].notna()].reset_index(drop=True)
    
    if len(df) < 100:
        print(f"⚠️  Insufficient data: {len(df)} days (need at least 100)")
        return []
    
    # Calculate rolling maximum (peak) - use expanding window
    df['peak'] = df['price'].expanding().max()
    
    # Calculate drawdown from peak (as percentage)
    df['drawdown_pct'] = ((df['price'] - df['peak']) / df['peak']) * 100
    
    # Identify crash periods (drawdown exceeds threshold)
    df['in_crash'] = df['drawdown_pct'] <= -threshold_pct
    
    # Find crash start/end points using state changes
    df['crash_start'] = (~df['in_crash'].shift(1, fill_value=False)) & df['in_crash']
    df['crash_end'] = df['in_crash'].shift(1, fill_value=False) & (~df['in_crash'])
    
    crashes = []
    
    # Iterate through crash periods
    crash_starts = df[df['crash_start']].index.tolist()
    
    for start_idx in crash_starts:
        # Find the peak before crash started
        # Look back to find the actual peak (where drawdown was 0)
        peak_idx = start_idx
        while peak_idx > 0 and df.loc[peak_idx, 'drawdown_pct'] < 0:
            peak_idx -= 1
        
        # Find crash end (when drawdown returns above threshold)
        end_candidates = df[(df.index > start_idx) & (~df['in_crash'])].index
        if len(end_candidates) == 0:
            # Crash hasn't recovered yet (ongoing)
            end_idx = len(df) - 1
            recovery_idx = None
        else:
            end_idx = end_candidates[0]
            
            # Find recovery date (when price returns to pre-crash peak)
            peak_price = df.loc[peak_idx, 'price']
            recovery_candidates = df[(df.index >= end_idx) & (df['price'] >= peak_price)].index
            recovery_idx = recovery_candidates[0] if len(recovery_candidates) > 0 else None
        
        # Find trough (lowest point during crash)
        crash_period = df.loc[peak_idx:end_idx]
        trough_idx = crash_period['price'].idxmin()
        
        # Calculate crash statistics
        start_date = df.loc[peak_idx, 'date']
        trough_date = df.loc[trough_idx, 'date']
        recovery_date = df.loc[recovery_idx, 'date'] if recovery_idx is not None else None
        
        peak_price = df.loc[peak_idx, 'price']
        trough_price = df.loc[trough_idx, 'price']
        max_drawdown = ((trough_price - peak_price) / peak_price) * 100
        
        duration_days = (trough_date - start_date).days
        recovery_days = (recovery_date - trough_date).days if recovery_date else None
        
        # Filter out very short crashes (likely noise)
        if duration_days < MIN_CRASH_DURATION:
            continue
        
        # Classify crash severity
        crash_type = classify_crash_severity(max_drawdown)
        
        crashes.append({
            'start_date': start_date,
            'trough_date': trough_date,
            'recovery_date': recovery_date,
            'max_drawdown': max_drawdown,
            'duration_days': duration_days,
            'recovery_days': recovery_days,
            'crash_type': crash_type,
            'peak_price': peak_price,
            'trough_price': trough_price
        })
    
    return crashes


def populate_crash_events():
    """Populate crash_events table with dynamically detected crashes."""
    print("=" * 80)
    print("DYNAMIC CRASH DETECTION - PARADIGM SHIFT")
    print("=" * 80)
    print(f"Crash Threshold: {CRASH_THRESHOLD}%")
    print(f"Minimum Crash Duration: {MIN_CRASH_DURATION} days")
    print("=" * 80)
    
    db = DatabaseManager()
    db.create_tables()
    
    # Load the equity index from the database.
    #
    # This deliberately bypasses the ORM and uses the same price-column
    # resolver as the v6 feature builder. The ORM's `Indicator.sp500_close`
    # is fed by FRED's SP500 series, which is licensed as a rolling 10-year
    # window — labelling crashes from it produced an episode table that
    # began in 2016 and silently omitted every crash before then
    # (1987, 2000, 2008). See docs/V6_POSTMORTEM.md.
    import sqlite3
    from src.v6.config import DB_PATH
    from src.v6.features.builder import resolve_price_column

    print("\nLoading index price data from database...")
    with sqlite3.connect(str(DB_PATH)) as con:
        raw = pd.read_sql_query("SELECT * FROM indicators ORDER BY date ASC", con)
    if raw.empty:
        print("❌ No indicator data found. Run collect_data.py first!")
        return
    raw["date"] = pd.to_datetime(raw["date"])
    price_col = resolve_price_column(raw.set_index("date"))

    df = pd.DataFrame({
        "date": raw["date"],
        "price": pd.to_numeric(raw[price_col], errors="coerce"),
    })

    print(f"✅ Loaded {len(df)} days from column '{price_col}'")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Detect crashes
    print(f"\nDetecting crashes with {CRASH_THRESHOLD}% threshold...")
    crashes = detect_crashes_from_data(df, threshold_pct=CRASH_THRESHOLD)
    
    if not crashes:
        print("❌ No crashes detected. Try lowering CRASH_THRESHOLD in .env")
        return
    
    print(f"✅ Detected {len(crashes)} crashes/corrections")

    # The detector emits one record per threshold crossing, so a single
    # episode can appear several times with the same peak. Keep the deepest
    # record per (start, trough) pair so the table holds one row per episode.
    deduped = {}
    for crash in crashes:
        key = (crash['start_date'], crash['trough_date'])
        best = deduped.get(key)
        if best is None or crash['max_drawdown'] < best['max_drawdown']:
            deduped[key] = crash
    if len(deduped) < len(crashes):
        print(f"   Deduplicated to {len(deduped)} distinct episodes")
    crashes = sorted(deduped.values(), key=lambda c: c['start_date'])
    
    # Populate database
    print("\nPopulating database...")
    with db.get_session() as session:
        # Clear existing data
        session.query(CrashEvent).delete()
        session.commit()
        
        # Add detected crashes
        for crash in crashes:
            recovery_months = crash['recovery_days'] // 30 if crash['recovery_days'] else None
            
            event = CrashEvent(
                start_date=crash['start_date'],
                end_date=crash['recovery_date'] if crash['recovery_date'] else crash['trough_date'],
                trough_date=crash['trough_date'],
                recovery_date=crash['recovery_date'],
                max_drawdown=crash['max_drawdown'],
                recovery_months=recovery_months,
                crash_type=crash['crash_type'],
                notes=f"Auto-detected: {crash['duration_days']}d to bottom, Peak ${crash['peak_price']:.0f} → Trough ${crash['trough_price']:.0f}"
            )
            session.add(event)
        
        session.commit()
    
    print(f"✅ Successfully populated {len(crashes)} crashes")
    
    # Display summary by severity
    print("\n" + "=" * 80)
    print("CRASH SUMMARY BY SEVERITY")
    print("=" * 80)
    
    severity_counts = {}
    for crash in crashes:
        severity = crash['crash_type']
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    for severity in ['Minor Correction', 'Moderate Correction', 'Major Correction', 'Severe Crash', 'Extreme Crash']:
        count = severity_counts.get(severity, 0)
        if count > 0:
            print(f"  {severity}: {count} events")
    
    # Display recent crashes
    print("\n" + "=" * 80)
    print("RECENT CRASHES (Last 10)")
    print("=" * 80)
    
    for crash in crashes[-10:]:
        recovery_str = crash['recovery_date'].strftime('%Y-%m-%d') if crash['recovery_date'] else 'Ongoing'
        print(f"  {crash['start_date'].strftime('%Y-%m-%d')} → {crash['trough_date'].strftime('%Y-%m-%d')} → {recovery_str}")
        print(f"    {crash['crash_type']}: {crash['max_drawdown']:.1f}% drawdown, {crash['duration_days']} days to bottom")
    
    print("\n" + "=" * 80)
    print(f"✅ DYNAMIC CRASH DETECTION COMPLETE - {len(crashes)} EVENTS READY FOR TRAINING")
    print("=" * 80)


if __name__ == '__main__':
    populate_crash_events()

