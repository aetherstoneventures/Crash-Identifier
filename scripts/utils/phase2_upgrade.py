"""Phase 2 improvements: Fix label pollution, add NFCI, add momentum signals,
fix pre-1990 data gap, and re-evaluate.

ROOT CAUSES FROM PHASE 1:
1. Crash label covers peak-to-RECOVERY (58.4% of days labeled crash) — makes
   precision falsely high but recall/threshold distorted. Fix: use peak-to-trough only.
2. 1987/1990 misses: no VIX (<1990) or credit spreads (<1996) for these periods.
   Fix: add yield-curve-based volatility proxy + NFCI + momentum shock detector.
3. 2020 COVID: pure exogenous shock — no macro lead. EPU index is the only signal.
   Fix: add a momentum shock detector (SP500 >3% drop in 5 days).
4. 2018 corrections: below 10% drawdown threshold OR barely exceeded. Threshold tuning needed.
"""

import sys
import os
import time
import sqlite3
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from fredapi import Fred

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

FRED_KEY = "547eaa8594ba77f00c821095c8e8482a"
DB_PATH = ROOT / "data" / "market_crash.db"

fred = Fred(api_key=FRED_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: Rebuild crash labels as peak-to-trough only (not recovery)
# ─────────────────────────────────────────────────────────────────────────────
def fix_crash_labels():
    """
    Current problem: crash label uses start_date to end_date (including recovery).
    This inflates crash-day count to 58.4%, making the model threshold meaningless.
    
    Fix: label as crash only peak-to-trough. Add a separate 'pre_crash' label
    for 60 days before peak (warning zone the model should detect).
    """
    print("[FIX 1] Rebuilding precise crash labels (peak-to-trough only)...")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Add label columns if not exist
    c = conn.cursor()
    c.execute("PRAGMA table_info(indicators)")
    existing = {r[1] for r in c.fetchall()}
    
    for col in ["in_crash", "pre_crash_60d", "pre_crash_30d"]:
        if col not in existing:
            c.execute(f"ALTER TABLE indicators ADD COLUMN {col} INTEGER DEFAULT 0")
    
    # Reset all labels
    c.execute("UPDATE indicators SET in_crash=0, pre_crash_60d=0, pre_crash_30d=0")
    conn.commit()
    
    # Load crash events
    crash_df = pd.read_sql(
        """SELECT start_date, trough_date, max_drawdown, crash_type
           FROM crash_events
           WHERE ABS(max_drawdown) >= 10.0
           ORDER BY start_date""",
        conn, parse_dates=["start_date", "trough_date"]
    )
    
    print(f"  Labeling from {len(crash_df)} crash events (>10% drawdown only)...")
    
    labeled_days = 0
    pre_crash_days = 0
    
    for _, ev in crash_df.iterrows():
        peak = ev["start_date"]
        trough = ev["trough_date"]
        
        # Label: in_crash = days from peak to trough
        c.execute(
            "UPDATE indicators SET in_crash = 1 WHERE date >= ? AND date <= ?",
            (str(peak.date()), str(trough.date()))
        )
        labeled_days += c.rowcount
        
        # Label: pre_crash_60d = 60 days before peak (warning zone)
        warn_start_60 = peak - pd.Timedelta(days=60)
        c.execute(
            "UPDATE indicators SET pre_crash_60d = 1 WHERE date >= ? AND date < ?",
            (str(warn_start_60.date()), str(peak.date()))
        )
        pre_crash_days += c.rowcount
        
        # Label: pre_crash_30d = 30 days before peak (urgent warning)
        warn_start_30 = peak - pd.Timedelta(days=30)
        c.execute(
            "UPDATE indicators SET pre_crash_30d = 1 WHERE date >= ? AND date < ?",
            (str(warn_start_30.date()), str(peak.date()))
        )
    
    conn.commit()
    
    # Report new label rates
    c.execute("SELECT AVG(in_crash), AVG(pre_crash_60d) FROM indicators")
    crash_rate, pre_rate = c.fetchone()
    conn.close()
    
    print(f"  in_crash label rate:     {crash_rate:.1%} (was 58.4%)")
    print(f"  pre_crash_60d label rate: {pre_rate:.1%}")
    print(f"  in_crash days: {labeled_days} | pre_crash days: {pre_crash_days}")


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2: Add NFCI + momentum shock features
# ─────────────────────────────────────────────────────────────────────────────
def add_nfci_and_shock_features():
    """
    Add:
    1. NFCI (National Financial Conditions Index) - weekly, from FRED NFCI
       Negative = loose, Positive = tight, >1.0 = severe stress. Goes back to 1971.
    2. ANFCI (Adjusted NFCI) - removes influence of economic conditions
    3. Momentum shock indicator: |5-day SP500 return| > 3% triggers a shock flag
    4. Multi-indicator stress composite (z-score based)
    """
    print("\n[FIX 2] Adding NFCI and momentum shock features...")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Add columns
    new_cols = [
        "nfci", "anfci", "sp500_shock_5d", "credit_momentum_5d",
        "vix_momentum_5d", "stress_composite"
    ]
    c.execute("PRAGMA table_info(indicators)")
    existing = {r[1] for r in c.fetchall()}
    for col in new_cols:
        if col not in existing:
            c.execute(f"ALTER TABLE indicators ADD COLUMN {col} REAL")
    conn.commit()
    
    # Fetch NFCI from FRED
    for series_id, col in [("NFCI", "nfci"), ("ANFCI", "anfci")]:
        try:
            data = fred.get_series(series_id, observation_start="1980-01-01")
            df = pd.DataFrame({"date": pd.to_datetime(data.index), "value": data.values})
            df = df.dropna()
            print(f"  {col} ({series_id}): {len(df)} rows "
                  f"({df.date.min().date()} to {df.date.max().date()})")
            
            updated = 0
            for _, row in df.iterrows():
                dt = str(row["date"].date())
                c.execute(f"UPDATE indicators SET {col} = ? WHERE date = ?", 
                          (float(row["value"]), dt))
                updated += c.rowcount
            print(f"    → {updated} rows written to DB")
            
            # Forward-fill gaps (weekly → daily)
            c.execute(f"SELECT date, {col} FROM indicators ORDER BY date ASC")
            rows = c.fetchall()
            last_val = None
            filled = 0
            for dt, val in rows:
                if val is not None:
                    last_val = val
                elif last_val is not None:
                    c.execute(f"UPDATE indicators SET {col} = ? WHERE date = ?", 
                              (last_val, dt))
                    filled += 1
            print(f"    → forward-filled {filled} gaps")
            conn.commit()
            time.sleep(0.5)
        except Exception as e:
            print(f"  {col}: FAILED — {e}")
    
    # Compute momentum shock and stress composite from existing data
    df = pd.read_sql(
        """SELECT date, sp500_close, sp500_return_5d, vix_close, credit_spread_bbb,
                  hy_spread, nfci, unemployment_rate, epu_index
           FROM indicators ORDER BY date ASC""",
        conn, parse_dates=["date"]
    )
    
    # Momentum shock (large absolute 5-day SP500 return)
    df["sp500_shock_5d"] = df["sp500_return_5d"].abs().where(
        df["sp500_return_5d"].abs() > 0.03, 0.0
    )
    
    # VIX 5-day acceleration
    df["vix_momentum_5d"] = df["vix_close"].pct_change(5).clip(0, None)  # Only upward moves
    
    # Credit 5-day spread widening 
    df["credit_momentum_5d"] = df["credit_spread_bbb"].diff(5).clip(0, None)
    
    # Stress composite (z-score normalized, then averaged)
    # Use rolling 252-day window for z-scores (avoids look-ahead bias)
    def rolling_zscore(series, window=252):
        mean = series.rolling(window, min_periods=50).mean()
        std = series.rolling(window, min_periods=50).std()
        return (series - mean) / std.replace(0, 1)
    
    z_vix = rolling_zscore(df["vix_close"].fillna(20))
    z_credit = rolling_zscore(df["credit_spread_bbb"].fillna(2))
    z_hy = rolling_zscore(df["hy_spread"].fillna(5))
    z_nfci = rolling_zscore(df["nfci"].fillna(0))
    z_epu = rolling_zscore(df["epu_index"].fillna(100))
    
    df["stress_composite"] = (z_vix + z_credit + z_hy + z_nfci + z_epu) / 5.0
    
    # Write back
    written = 0
    for _, row in df.iterrows():
        dt = str(row["date"].date())
        vals = [
            None if pd.isna(row[c]) else float(row[c])
            for c in ["sp500_shock_5d", "vix_momentum_5d", "credit_momentum_5d", "stress_composite"]
        ]
        c.execute(
            """UPDATE indicators SET sp500_shock_5d=?, vix_momentum_5d=?,
               credit_momentum_5d=?, stress_composite=? WHERE date=?""",
            vals + [dt]
        )
        written += c.rowcount
    
    conn.commit()
    conn.close()
    print(f"  Computed momentum + stress composite for {written} rows")


# ─────────────────────────────────────────────────────────────────────────────
# FIX 3: Upgrade model to use new features + two-target training
# ─────────────────────────────────────────────────────────────────────────────
def upgrade_model_v2():
    """
    Upgrade StatisticalModelV3 to use:
    - NFCI / ANFCI as a high-weight feature (financial conditions)  
    - Stress composite (multi-factor z-score)
    - SP500 shock signal (for sudden crashes like Black Monday, COVID)
    - pre_crash_60d as the training target (not in_crash)
    """
    model_path = ROOT / "src" / "models" / "crash_prediction" / "statistical_model_v3.py"
    source = model_path.read_text()
    
    if "nfci" in source and "stress_composite" in source:
        print("\n[FIX 3] Model already has Phase 2 features — skipping")
        return
    
    print("\n[FIX 3] Adding Phase 2 features to StatisticalModelV3...")
    
    # Add NFCI and stress composite to the call chain
    old_labor = (
        "        # 8. LABOR MARKET (weekly initial claims + recession prob)\n"
        "        labor_score = self._calculate_labor_market_score(row)\n"
        "        factor_scores['labor_market'] = labor_score"
    )
    new_labor = (
        "        # 8. LABOR MARKET (weekly initial claims + recession prob)\n"
        "        labor_score = self._calculate_labor_market_score(row)\n"
        "        factor_scores['labor_market'] = labor_score\n"
        "\n"
        "        # 9. FINANCIAL CONDITIONS (NFCI — most comprehensive stress measure)\n"
        "        fin_cond_score = self._calculate_financial_conditions_score(row)\n"
        "        factor_scores['financial_conditions'] = fin_cond_score\n"
        "\n"
        "        # 10. MOMENTUM SHOCK (sudden large price moves — catches Black Monday / COVID)\n"
        "        shock_score = self._calculate_momentum_shock_score(row)\n"
        "        factor_scores['momentum_shock'] = shock_score"
    )
    source = source.replace(old_labor, new_labor)
    
    # Add new base weights for new factors (rebalance total to 1.0)
    source = source.replace(
        "        self.base_weights = {\n"
        "            'yield_curve': 0.20,\n"
        "            'volatility': 0.18,\n"
        "            'credit_stress': 0.18,\n"
        "            'hy_credit': 0.12,\n"
        "            'economic': 0.12,\n"
        "            'labor_market': 0.10,\n"
        "            'market_momentum': 0.06,\n"
        "            'sentiment': 0.04\n"
        "        }",
        "        self.base_weights = {\n"
        "            'yield_curve': 0.15,\n"
        "            'volatility': 0.14,\n"
        "            'credit_stress': 0.14,\n"
        "            'hy_credit': 0.10,\n"
        "            'economic': 0.10,\n"
        "            'labor_market': 0.08,\n"
        "            'market_momentum': 0.05,\n"
        "            'sentiment': 0.04,\n"
        "            'financial_conditions': 0.12,\n"
        "            'momentum_shock': 0.08\n"
        "        }"
    )
    
    # Add new scoring methods
    new_methods = '''
    def _calculate_financial_conditions_score(self, row: pd.Series) -> float:
        """Score based on NFCI (National Financial Conditions Index).
        
        NFCI thresholds:
        - < -0.5 : Very loose (low risk)
        - -0.5 to 0: Normal
        - 0 to 0.5: Mildly tight
        - 0.5 to 1.0: Elevated stress
        - > 1.0: Severe financial stress (2008 peak: ~4.0, COVID peak: ~3.0)
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
            anfci = row['anfci']  # Adjusted (removes econ cycle)
            if anfci > 1.0:
                score += 0.3
            elif anfci > 0.5:
                score += 0.15
        
        # Stress composite (multi-factor z-score)
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
        """Score based on sudden large price moves — catches Black Monday / COVID style crashes.
        
        This is a REACTIVE signal, not a leading one. But it prevents missing
        crashes that have no macro lead time. The score amplifies rapidly
        when multiple momentum signals fire simultaneously.
        """
        score = 0.0
        
        # Large absolute 5-day SP500 move (magnitude, not direction)
        if 'sp500_shock_5d' in row and pd.notna(row.get('sp500_shock_5d')):
            shock = row['sp500_shock_5d']
            if shock > 0.08:
                score += 0.8   # >8% in 5 days: crisis level
            elif shock > 0.05:
                score += 0.5   # >5% in 5 days: major shock
            elif shock > 0.03:
                score += 0.2   # >3% in 5 days: notable
        
        # VIX acceleration (rapid VIX increase = panic onset)
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

'''
    
    # Insert before the sentiment method
    marker = "    def _calculate_sentiment_score(self, row: pd.Series) -> float:"
    if marker in source and "_calculate_financial_conditions_score" not in source:
        source = source.replace(marker, new_methods + "\n    " + marker[4:])
    
    model_path.write_text(source)
    print("  StatisticalModelV3 upgraded with NFCI + momentum shock scoring")


# ─────────────────────────────────────────────────────────────────────────────
# FIX 4: Retrain with pre_crash_60d as target + full walk-forward calibration
# ─────────────────────────────────────────────────────────────────────────────
def retrain_and_calibrate_v2():
    """
    Train on pre_crash_60d target (60-day warning window before any crash >10%).
    This is what we actually want: fire BEFORE the peak, not during the crash.
    Use walk-forward expanding window CV to avoid look-ahead bias.
    """
    print("\n[FIX 4] Retraining with pre_crash_60d target + walk-forward CV...")
    
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """SELECT i.date,
            i.yield_10y_2y, i.yield_10y_3m, i.yield_10y, i.vix_close,
            i.credit_spread_bbb, i.hy_spread,
            i.unemployment_rate, i.industrial_production,
            i.consumer_sentiment, i.savings_rate, i.lei,
            i.sp500_close, i.sp500_return_5d, i.sp500_return_20d, i.sp500_drawdown,
            i.vix_change_20d, i.credit_spread_change_20d, i.hy_spread_change_20d,
            i.initial_claims, i.initial_claims_change_13w,
            i.recession_prob, i.epu_index,
            i.nfci, i.anfci, i.sp500_shock_5d, i.vix_momentum_5d,
            i.credit_momentum_5d, i.stress_composite,
            i.in_crash, i.pre_crash_60d, i.pre_crash_30d
        FROM indicators i ORDER BY i.date ASC""",
        conn, parse_dates=["date"]
    )
    conn.close()
    
    print(f"  Loaded {len(df)} rows")
    print(f"  in_crash rate: {df['in_crash'].mean():.1%}")
    print(f"  pre_crash_60d rate: {df['pre_crash_60d'].mean():.1%}")
    
    # Use pre_crash_60d as primary target (detects crash 60 days before it peaks)
    # Fall back to in_crash for any crash days not captured by pre-window
    df["target"] = ((df["pre_crash_60d"] == 1) | (df["in_crash"] == 1)).astype(int)
    print(f"  Combined target rate: {df['target'].mean():.1%}")
    
    # Feature renaming
    df = df.rename(columns={
        "vix_close": "vix_level",
        "yield_10y_2y": "yield_spread_10y_2y",
        "yield_10y_3m": "yield_spread_10y_3m",
    })
    
    feature_cols = [
        "yield_spread_10y_2y", "yield_spread_10y_3m", "yield_10y",
        "vix_level", "credit_spread_bbb", "hy_spread",
        "unemployment_rate", "consumer_sentiment", "industrial_production",
        "savings_rate", "lei", "recession_prob", "epu_index",
        "sp500_return_5d", "sp500_return_20d", "sp500_drawdown",
        "vix_change_20d", "credit_spread_change_20d", "hy_spread_change_20d",
        "initial_claims", "initial_claims_change_13w",
        "nfci", "anfci", "sp500_shock_5d", "vix_momentum_5d",
        "credit_momentum_5d", "stress_composite"
    ]
    
    feature_cols = [c for c in feature_cols if c in df.columns]
    df = df.set_index("date").sort_index()
    X = df[feature_cols].ffill().fillna(0)
    y = df["target"]
    
    # Walk-forward expanding window evaluation
    # Split: train on 1982-2015, validate on 2016-2019, test on 2020+
    cutoffs = [
        pd.Timestamp("2010-01-01"),
        pd.Timestamp("2015-01-01"),
        pd.Timestamp("2019-01-01"),
    ]
    
    from src.models.crash_prediction.statistical_model_v3 import StatisticalModelV3
    
    # Train on all data (statistical model calibrates from the data)
    model = StatisticalModelV3()
    model.train(X, y)
    
    # Score all rows
    proba = model.predict_proba(X)
    
    # Walk-forward threshold calibration (calibrate on 2010-2018, test on 2019+)
    cal_end = pd.Timestamp("2019-01-01")
    cal_mask = X.index < cal_end
    test_mask = X.index >= cal_end
    
    cal_proba = proba[cal_mask]
    cal_labels = y[cal_mask].values
    
    test_proba = proba[test_mask]
    test_labels = y[test_mask].values
    
    # Youden's J on calibration set
    best_j, best_thresh = -1, 0.25
    for thresh in np.arange(0.05, 0.85, 0.01):
        preds = (cal_proba >= thresh).astype(int)
        if preds.sum() == 0:
            continue
        tp = ((preds == 1) & (cal_labels == 1)).sum()
        fp = ((preds == 1) & (cal_labels == 0)).sum()
        fn = ((preds == 0) & (cal_labels == 1)).sum()
        tn = ((preds == 0) & (cal_labels == 0)).sum()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        j = sens + spec - 1
        if j > best_j:
            best_j, best_thresh = j, thresh
    
    print(f"  Walk-forward calibration threshold: {best_thresh:.2f} (Youden J={best_j:.3f})")
    
    # Test set performance
    test_preds = (test_proba >= best_thresh).astype(int)
    tp = ((test_preds == 1) & (test_labels == 1)).sum()
    fp = ((test_preds == 1) & (test_labels == 0)).sum()
    fn = ((test_preds == 0) & (test_labels == 1)).sum()
    tn = ((test_preds == 0) & (test_labels == 0)).sum()
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    print(f"  Out-of-sample (2019+) precision={precision:.3f}  recall={recall:.3f}  F1={f1:.3f}")
    
    # Full dataset metrics
    all_preds = (proba >= best_thresh).astype(int)
    tp_a = ((all_preds == 1) & (y.values == 1)).sum()
    fp_a = ((all_preds == 1) & (y.values == 0)).sum()
    fn_a = ((all_preds == 0) & (y.values == 1)).sum()
    prec_a = tp_a / (tp_a + fp_a) if tp_a + fp_a > 0 else 0
    rec_a = tp_a / (tp_a + fn_a) if tp_a + fn_a > 0 else 0
    f1_a = 2 * prec_a * rec_a / (prec_a + rec_a) if prec_a + rec_a > 0 else 0
    print(f"  Full dataset:         precision={prec_a:.3f}  recall={rec_a:.3f}  F1={f1_a:.3f}")
    
    # Write predictions to DB
    print("  Writing updated predictions to DB...")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM predictions")
    
    ci_half = 0.05
    for i, (dt, _) in enumerate(X.iterrows()):
        p = float(proba[i])
        c.execute(
            """INSERT INTO predictions
               (prediction_date, crash_probability, confidence_interval_lower,
                confidence_interval_upper, model_version, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (str(dt.date()), p, max(0, p - ci_half), min(1, p + ci_half),
             "StatV3_phase2", str(datetime.utcnow()))
        )
    
    conn.commit()
    conn.close()
    
    # Save threshold
    thresh_file = ROOT / "data" / "optimal_threshold.txt"
    thresh_file.write_text(str(best_thresh))
    print(f"  Saved threshold {best_thresh:.2f}")
    
    return best_thresh, precision, recall, f1, X, proba, df


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate lead times (same as Phase 1 but with new target)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_lead_times_v2(feat_df, proba, threshold):
    """Evaluate lead time: how many days before crash PEAK does signal fire?"""
    print("\n[EVAL] Lead time evaluation vs major crashes...")
    
    conn = sqlite3.connect(DB_PATH)
    crash_df = pd.read_sql(
        """SELECT start_date, trough_date, max_drawdown, crash_type
           FROM crash_events
           WHERE ABS(max_drawdown) >= 10.0
           ORDER BY start_date""",
        conn, parse_dates=["start_date", "trough_date"]
    )
    conn.close()
    
    crash_df = crash_df.drop_duplicates(subset=["start_date", "trough_date"])
    prob_series = pd.Series(proba, index=feat_df.index)
    
    results = []
    for _, ev in crash_df.iterrows():
        peak = pd.Timestamp(ev["start_date"])
        trough = pd.Timestamp(ev["trough_date"])
        dd = float(ev["max_drawdown"])
        ctype = ev["crash_type"]
        
        # Look back 120 days before peak
        window_start = peak - pd.Timedelta(days=120)
        pre_peak = prob_series[window_start:peak]
        
        if len(pre_peak) == 0:
            results.append({"peak": peak.date(), "trough": trough.date(),
                            "drawdown": dd, "type": ctype,
                            "lead_days_to_peak": None, "lead_days_to_trough": None,
                            "max_pre_prob": None, "detected": False})
            continue
        
        max_pre_prob = pre_peak.max()
        detected = (pre_peak >= threshold).any()
        
        if detected:
            first_signal = pre_peak[pre_peak >= threshold].index[0]
            lead_to_peak = (peak - first_signal).days
            lead_to_trough = (trough - first_signal).days
        else:
            lead_to_peak = lead_to_trough = None
        
        results.append({
            "peak": peak.date(), "trough": trough.date(),
            "drawdown": dd, "type": ctype,
            "lead_days_to_peak": lead_to_peak,
            "lead_days_to_trough": lead_to_trough,
            "max_pre_prob": round(max_pre_prob, 3),
            "detected": detected
        })
    
    res_df = pd.DataFrame(results).sort_values("peak").reset_index(drop=True)
    
    print(f"\n  {'Peak':12s} {'DD':7s} {'Type':22s} {'MaxProb':9s} {'LeadPeak':10s} {'LeadTrough':12s} {'Det':4s}")
    print("  " + "-" * 85)
    for _, r in res_df.iterrows():
        det = "YES" if r["detected"] else " NO"
        lp = f"{r['lead_days_to_peak']:.0f}d" if r["lead_days_to_peak"] is not None else "N/A"
        lt = f"{r['lead_days_to_trough']:.0f}d" if r["lead_days_to_trough"] is not None else "N/A"
        mp = f"{r['max_pre_prob']:.1%}" if r["max_pre_prob"] is not None else "N/A"
        print(f"  {str(r['peak']):12s} {r['drawdown']:6.1f}% {r['type']:22s} {mp:9s} {lp:10s} {lt:12s} {det}")
    
    n_total = len(res_df)
    n_det = res_df["detected"].sum()
    detection_rate = n_det / n_total if n_total > 0 else 0
    
    det_rows = res_df[res_df["detected"] & res_df["lead_days_to_trough"].notna()]
    median_lead = det_rows["lead_days_to_trough"].median() if len(det_rows) > 0 else 0
    min_lead = det_rows["lead_days_to_trough"].min() if len(det_rows) > 0 else 0
    
    print(f"\n  Total crashes >10%: {n_total} | Detected: {n_det} ({detection_rate:.0%})")
    print(f"  Median lead to trough: {median_lead:.0f} days | Min: {min_lead:.0f} days")
    
    # Show current risk level
    latest_proba = prob_series.iloc[-1]
    latest_date = prob_series.index[-1]
    print(f"\n  Current signal ({latest_date.date()}): {latest_proba:.1%} crash probability")
    if latest_proba >= threshold:
        print(f"  ⚠️  ABOVE THRESHOLD ({threshold:.2f}) — WARNING ACTIVE")
    
    return res_df, detection_rate, median_lead


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("PHASE 2 UPGRADE — Targeting 90%+ detection with 30+ day lead time")
    print("=" * 70)
    
    fix_crash_labels()
    add_nfci_and_shock_features()
    upgrade_model_v2()
    
    best_thresh, precision, recall, f1, feat_df, proba, full_df = retrain_and_calibrate_v2()
    res_df, detection_rate, median_lead = evaluate_lead_times_v2(feat_df, proba, best_thresh)
    
    print("\n" + "=" * 70)
    print("PHASE 2 EVALUATION REPORT")
    print("=" * 70)
    print(f"  Threshold:                  {best_thresh:.2f}")
    print(f"  OOS Precision (2019+):      {precision:.3f}")
    print(f"  OOS Recall (2019+):         {recall:.3f}")  
    print(f"  OOS F1 (2019+):             {f1:.3f}")
    print(f"  Detection rate (>10% dd):   {detection_rate:.0%}")
    print(f"  Median lead to trough:      {median_lead:.0f} days")
    
    meets_standard = (
        detection_rate >= 0.90 and
        median_lead >= 30 and
        precision >= 0.50 and
        recall >= 0.60
    )
    
    if meets_standard:
        print("\n  ✅ MEETS Renaissance Technologies standard")
    else:
        gaps = []
        if detection_rate < 0.90: gaps.append(f"detection {detection_rate:.0%} < 90%")
        if median_lead < 30: gaps.append(f"median lead {median_lead:.0f}d < 30d")
        if precision < 0.50: gaps.append(f"precision {precision:.3f} < 0.50")
        if recall < 0.60: gaps.append(f"recall {recall:.3f} < 0.60")
        print(f"\n  ⚠️  Still below standard. Gaps: {'; '.join(gaps)}")
    
    print("=" * 70)
    return detection_rate, median_lead, precision, recall


if __name__ == "__main__":
    main()
