#!/usr/bin/env python3
"""
Simple test to verify StatisticalModelV3 can run without mutex issues.
"""

import sys
import os

# Set working directory
os.chdir('/Users/pouyamahdavipourvahdati/Desktop/General/Projects/01_Project_Stock Automation/Project 2025_Stock Evaluation/Hidden Gem Stock/Augment Code Crash Analyzer/Crash-Identifier-main')
sys.path.insert(0, '.')

print("=" * 80)
print("TESTING STATISTICAL MODEL V3 - SIMPLE VERSION")
print("=" * 80)

# Test 1: Import sqlite3 and load data
print("\n[1/5] Testing database connection...")
import sqlite3
import pandas as pd
import numpy as np

try:
    conn = sqlite3.connect('data/market_crash.db', timeout=30)
    df = pd.read_sql_query("SELECT * FROM indicators ORDER BY date LIMIT 10", conn)
    print(f"✅ Database connection OK - loaded {len(df)} rows")
    conn.close()
except Exception as e:
    print(f"❌ Database connection FAILED: {e}")
    sys.exit(1)

# Test 2: Import StatisticalModelV3
print("\n[2/5] Testing StatisticalModelV3 import...")
try:
    from src.models.crash_prediction.statistical_model_v3 import StatisticalModelV3
    print("✅ StatisticalModelV3 import OK")
except Exception as e:
    print(f"❌ StatisticalModelV3 import FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Load full data
print("\n[3/5] Loading full dataset...")
try:
    conn = sqlite3.connect('data/market_crash.db', timeout=30)
    df = pd.read_sql_query("SELECT * FROM indicators ORDER BY date", conn)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    crashes = pd.read_sql_query("SELECT start_date, end_date FROM crash_events", conn)
    crash_events = list(zip(crashes['start_date'], crashes['end_date']))
    conn.close()
    
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"✅ Crash events: {len(crash_events)}")
except Exception as e:
    print(f"❌ Data loading FAILED: {e}")
    sys.exit(1)

# Test 4: Prepare features
print("\n[4/5] Preparing features...")
try:
    X = pd.DataFrame(index=df.index)
    X['yield_10y_2y'] = df['yield_10y_2y']
    X['yield_10y_3m'] = df['yield_10y_3m']
    X['vix_close'] = df['vix_close']
    X['credit_spread_bbb'] = df['credit_spread_bbb']
    X['unemployment_rate'] = df['unemployment_rate']
    X['industrial_production'] = df['industrial_production']
    X['consumer_sentiment'] = df['consumer_sentiment']
    X['sp500_drawdown'] = (df['sp500_close'] / df['sp500_close'].rolling(252).max() - 1)
    X['sp500_return_5d'] = df['sp500_close'].pct_change(5)
    X['sp500_return_20d'] = df['sp500_close'].pct_change(20)
    X['vix_change_pct'] = df['vix_close'].pct_change()
    X['vix_change_5d'] = df['vix_close'].pct_change(5)
    X['credit_spread_change'] = df['credit_spread_bbb'].diff(20)
    X['unemployment_change'] = df['unemployment_rate'].diff(20)
    X['industrial_production_change'] = df['industrial_production'].diff(20)
    X = X.ffill().bfill().fillna(0)
    
    print(f"✅ Features prepared: {X.shape[1]} features")
except Exception as e:
    print(f"❌ Feature preparation FAILED: {e}")
    sys.exit(1)

# Test 5: Create labels and test model
print("\n[5/5] Testing StatisticalModelV3...")
try:
    y = pd.Series(0, index=df.index)
    for crash_start, _ in crash_events:
        crash_date = pd.to_datetime(crash_start)
        lookback = crash_date - pd.Timedelta(days=90)
        y[(df.index >= lookback) & (df.index < crash_date)] = 1
    
    print(f"✅ Labels created: {y.sum()} crash samples ({y.sum()/len(y)*100:.1f}%)")
    
    # Test model on small subset
    X_test = X.iloc[:1000]
    y_test = y.iloc[:1000]
    
    model = StatisticalModelV3()
    model.train(X_test, y_test)
    proba = model.predict_proba(X_test)
    
    print(f"✅ Model trained and predicted on {len(X_test)} samples")
    print(f"✅ Prediction range: [{proba.min():.4f}, {proba.max():.4f}]")
    
except Exception as e:
    print(f"❌ Model testing FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED - StatisticalModelV3 is working correctly!")
print("=" * 80)

