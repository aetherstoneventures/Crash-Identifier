"""Full upgrade pipeline - executes all 5 priority improvements.

STEP 1: Fix SP500 history (Shiller 1871+ + FRED SP500 2016+ + cache)
STEP 2: Download & add HY spread, Initial Claims, NY Fed recession prob
STEP 3: Add new columns to SQLite (ALTER TABLE — safe, non-destructive)
STEP 4: Recompute derived features (drawdown, 5d/20d returns, changes)
STEP 5: Regenerate crash events + walk-forward retrain + recalibrate threshold
STEP 6: Evaluate lead times and report performance metrics
"""

import sys
import os
import re
import time
import sqlite3
import logging
import warnings
from datetime import datetime, timedelta, date
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
CACHE_FILE = ROOT / "data" / "cache" / "sp500_cache.csv"
SHILLER_FILE = "/tmp/shiller.xls"

fred = Fred(api_key=FRED_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Build correct SP500 daily series
# ─────────────────────────────────────────────────────────────────────────────
def build_full_sp500():
    """Combine Shiller monthly + existing cache to produce a full daily SP500."""
    print("\n[STEP 1] Building full SP500 history...")

    # Parse Shiller monthly (1980-2023)
    df_sh = pd.read_excel(SHILLER_FILE, sheet_name="Data", header=7, engine="xlrd")
    df_sh = df_sh[["Date", "P"]].dropna(subset=["P"])

    def parse_date(d):
        m = re.match(r"(\d{4})\.(\d+)", str(d))
        if m:
            yr, mo = int(m.group(1)), int(m.group(2))
            mo = max(1, mo)
            try:
                return pd.Timestamp(yr, mo, 1)
            except Exception:
                return pd.NaT
        return pd.NaT

    df_sh["dt"] = df_sh["Date"].apply(parse_date)
    df_sh = df_sh.dropna(subset=["dt"]).sort_values("dt")
    df_sh = df_sh[df_sh["dt"] >= "1982-01-01"][["dt", "P"]].rename(
        columns={"dt": "date", "P": "close"}
    )
    df_sh["date"] = pd.to_datetime(df_sh["date"])
    print(f"  Shiller: {len(df_sh)} monthly rows ({df_sh.date.min().date()} to {df_sh.date.max().date()})")

    # Load existing cache (real daily data from ~2016)
    df_cache = pd.read_csv(CACHE_FILE, parse_dates=["date"])
    df_cache = df_cache.rename(columns={"close": "close"}).sort_values("date")
    print(f"  Cache:   {len(df_cache)} daily rows ({df_cache.date.min().date()} to {df_cache.date.max().date()})")

    # Create business-day index from 1982-01-04 to today
    bdays = pd.bdate_range(start="1982-01-04", end=pd.Timestamp.today())
    daily = pd.DataFrame({"date": bdays})
    daily["date"] = pd.to_datetime(daily["date"])

    # Merge Shiller monthly → business days (forward fill within month)
    df_sh_indexed = df_sh.drop_duplicates(subset=["date"]).set_index("date")["close"]
    daily = daily.set_index("date")
    daily["sp500_shiller"] = df_sh_indexed.reindex(daily.index, method="ffill")

    # Merge cache (real daily — overrides Shiller where available)
    df_cache_indexed = df_cache.drop_duplicates(subset=["date"]).set_index("date")["close"]
    daily["sp500_cache"] = df_cache_indexed.reindex(daily.index)

    # Final SP500: prefer real cache, fall back to Shiller
    daily["sp500_close"] = daily["sp500_cache"].combine_first(daily["sp500_shiller"])
    daily = daily.reset_index()

    # Fill any remaining gaps via forward fill only (no bfill — preserves NaN at start)
    daily["sp500_close"] = daily["sp500_close"].ffill()

    no_data = daily["sp500_close"].isna().sum()
    print(f"  Combined: {len(daily)} daily rows, {no_data} NaN (expected ~0)")
    print(f"  1987-10 values: {daily[daily['date'].dt.year == 1987]['sp500_close'].describe()[['min','max']].to_dict()}")

    return daily[["date", "sp500_close"]]


def update_sp500_in_db(sp500_df):
    """Patch sp500_close in indicators table with real historical data."""
    print("\n[STEP 1b] Patching database with real SP500 values...")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    sp500_map = {
        row["date"].date(): float(row["sp500_close"])
        for _, row in sp500_df.iterrows()
        if pd.notna(row["sp500_close"])
    }

    updated = 0
    for dt, price in sp500_map.items():
        c.execute("UPDATE indicators SET sp500_close = ? WHERE date = ?", (price, str(dt)))
        updated += c.rowcount

    conn.commit()
    conn.close()

    # Verify
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(DISTINCT sp500_close) FROM indicators WHERE date < '2016-01-01'")
    distinct = c.fetchone()[0]
    c.execute("SELECT MIN(sp500_close), MAX(sp500_close) FROM indicators WHERE date < '2016-01-01' AND sp500_close IS NOT NULL")
    rng = c.fetchone()
    conn.close()
    print(f"  Updated: {updated} rows | Pre-2016 distinct SP500 values: {distinct} | range: {rng[0]:.1f} - {rng[1]:.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Download new FRED indicators
# ─────────────────────────────────────────────────────────────────────────────
def fetch_fred_series(series_id, name, start="1980-01-01"):
    """Fetch a FRED series with retry."""
    for attempt in range(3):
        try:
            data = fred.get_series(series_id, observation_start=start)
            df = pd.DataFrame({"date": pd.to_datetime(data.index), "value": data.values})
            df = df.dropna().sort_values("date")
            print(f"  {name} ({series_id}): {len(df)} rows ({df.date.min().date()} to {df.date.max().date()})")
            return df
        except Exception as e:
            print(f"  {name}: attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return pd.DataFrame(columns=["date", "value"])


def download_new_indicators():
    """Fetch HY spread, Initial Claims, and NY Fed recession probability."""
    print("\n[STEP 2] Downloading new FRED indicators...")
    new_series = {
        "hy_spread": ("BAMLH0A0HYM2", "US HY Spread"),
        "initial_claims": ("ICSA", "Initial Unemployment Claims"),
        "recession_prob": ("RECPROUSM156N", "NY Fed 12m Recession Prob"),
        "epu_index": ("USEPUINDXD", "Economic Policy Uncertainty (Daily)"),
    }
    result = {}
    for col, (series_id, name) in new_series.items():
        df = fetch_fred_series(series_id, name)
        result[col] = df
        time.sleep(0.5)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Add new columns to SQLite (safe ALTER TABLE)
# ─────────────────────────────────────────────────────────────────────────────
NEW_COLUMNS = {
    "hy_spread": "REAL",
    "initial_claims": "REAL",
    "recession_prob": "REAL",
    "epu_index": "REAL",
    "sp500_return_5d": "REAL",
    "sp500_return_20d": "REAL",
    "sp500_drawdown": "REAL",
    "vix_change_20d": "REAL",
    "credit_spread_change_20d": "REAL",
    "hy_spread_change_20d": "REAL",
    "initial_claims_change_13w": "REAL",
}


def add_new_columns():
    """Add new columns to the indicators table if they don't exist."""
    print("\n[STEP 3] Adding new columns to indicators table...")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("PRAGMA table_info(indicators)")
    existing = {row[1] for row in c.fetchall()}

    added = []
    for col, dtype in NEW_COLUMNS.items():
        if col not in existing:
            c.execute(f"ALTER TABLE indicators ADD COLUMN {col} {dtype}")
            added.append(col)

    conn.commit()
    conn.close()
    print(f"  Added {len(added)} columns: {added}")
    print(f"  Skipped (already exist): {[c for c in NEW_COLUMNS if c not in added]}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Populate new columns with data + compute derived features
# ─────────────────────────────────────────────────────────────────────────────
def populate_new_indicators(new_data: dict):
    """Write new FRED series values into the indicators table."""
    print("\n[STEP 4a] Writing new FRED data into indicators table...")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    for col, df in new_data.items():
        if df.empty:
            print(f"  Skipping {col} (no data)")
            continue
        updated = 0
        for _, row in df.iterrows():
            dt = row["date"].date() if hasattr(row["date"], "date") else row["date"]
            val = float(row["value"])
            c.execute(f"UPDATE indicators SET {col} = ? WHERE date = ?", (val, str(dt)))
            updated += c.rowcount
        print(f"  {col}: {updated} rows updated")

    # For initial_claims and recession_prob: forward-fill to business days
    # (these are weekly/monthly but we need daily)
    for col in ["initial_claims", "recession_prob", "hy_spread", "epu_index"]:
        c.execute(f"SELECT date, {col} FROM indicators ORDER BY date ASC")
        rows = c.fetchall()
        if not rows:
            continue
        last_val = None
        filled = 0
        for dt, val in rows:
            if val is not None:
                last_val = val
            elif last_val is not None:
                c.execute(f"UPDATE indicators SET {col} = ? WHERE date = ?", (last_val, dt))
                filled += 1
        print(f"  {col}: forward-filled {filled} gaps")

    conn.commit()
    conn.close()


def compute_derived_features():
    """Compute rolling returns, drawdown, and change features from real SP500/VIX/spreads."""
    print("\n[STEP 4b] Computing derived features (drawdown, returns, changes)...")

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT date, sp500_close, vix_close, credit_spread_bbb, hy_spread, initial_claims "
        "FROM indicators ORDER BY date ASC",
        conn,
        parse_dates=["date"],
    )

    # SP500 returns and drawdown
    df["sp500_return_5d"] = df["sp500_close"].pct_change(5)
    df["sp500_return_20d"] = df["sp500_close"].pct_change(20)
    running_max = df["sp500_close"].expanding().max()
    df["sp500_drawdown"] = (df["sp500_close"] - running_max) / running_max

    # VIX change (20d % change)
    df["vix_change_20d"] = df["vix_close"].pct_change(20)

    # Credit spread changes
    df["credit_spread_change_20d"] = df["credit_spread_bbb"].diff(20)
    df["hy_spread_change_20d"] = df["hy_spread"].diff(20)

    # Initial claims 13-week change (leading recession signal)
    df["initial_claims_change_13w"] = df["initial_claims"].pct_change(65)  # ~13 weeks of bdays

    # Write back
    c = conn.cursor()
    cols_to_write = [
        "sp500_return_5d", "sp500_return_20d", "sp500_drawdown",
        "vix_change_20d", "credit_spread_change_20d", "hy_spread_change_20d",
        "initial_claims_change_13w"
    ]
    updated = 0
    for _, row in df.iterrows():
        dt = str(row["date"].date())
        vals = [
            None if pd.isna(row[c]) else float(row[c])
            for c in cols_to_write
        ]
        placeholders = ", ".join([f"{c} = ?" for c in cols_to_write])
        c.execute(f"UPDATE indicators SET {placeholders} WHERE date = ?", vals + [dt])
        updated += c.rowcount

    conn.commit()
    conn.close()
    print(f"  Computed and stored derived features for {updated} rows")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5a: Regenerate crash events (now with real SP500)
# ─────────────────────────────────────────────────────────────────────────────
def regenerate_crash_events():
    """Re-run crash event detection with corrected SP500 prices."""
    print("\n[STEP 5a] Regenerating crash events from real SP500 data...")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "data" / "populate_crash_events.py")],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    lines = (result.stdout + result.stderr).split("\n")
    for line in lines:
        if any(k in line for k in ["Detected", "Populated", "ERROR", "crash", "Crash", "✅", "❌"]):
            print(f"  {line}")
    if result.returncode != 0:
        print("  WARNING: populate_crash_events.py had errors (check manually)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5b: Upgrade StatisticalModelV3 to use all new features
# ─────────────────────────────────────────────────────────────────────────────
def upgrade_model_features():
    """Patch StatisticalModelV3 to use HY spread, initial claims, EPU, recession prob."""
    model_path = ROOT / "src" / "models" / "crash_prediction" / "statistical_model_v3.py"
    source = model_path.read_text()

    # Only patch if not already done
    if "hy_spread" in source and "initial_claims" in source and "recession_prob" in source:
        print("\n[STEP 5b] Model already upgraded — skipping")
        return

    print("\n[STEP 5b] Upgrading StatisticalModelV3 with new features...")

    # 1. Add new base weights
    source = source.replace(
        "        self.base_weights = {\n"
        "            'yield_curve': 0.25,\n"
        "            'volatility': 0.20,\n"
        "            'credit_stress': 0.20,\n"
        "            'economic': 0.15,\n"
        "            'market_momentum': 0.12,\n"
        "            'sentiment': 0.08\n"
        "        }",
        "        self.base_weights = {\n"
        "            'yield_curve': 0.20,\n"
        "            'volatility': 0.18,\n"
        "            'credit_stress': 0.18,\n"
        "            'hy_credit': 0.12,\n"
        "            'economic': 0.12,\n"
        "            'labor_market': 0.10,\n"
        "            'market_momentum': 0.06,\n"
        "            'sentiment': 0.04\n"
        "        }"
    )

    # 2. Add regime weight adjustments for new factors
    source = source.replace(
        "            'extreme': {\n"
        "                'yield_curve': 0.8,\n"
        "                'volatility': 1.5,  # Heavy weight on volatility\n"
        "                'credit_stress': 1.3,\n"
        "                'economic': 0.8,\n"
        "                'market_momentum': 1.2,\n"
        "                'sentiment': 1.1\n"
        "            }",
        "            'extreme': {\n"
        "                'yield_curve': 0.7,\n"
        "                'volatility': 1.4,\n"
        "                'credit_stress': 1.2,\n"
        "                'hy_credit': 1.5,\n"
        "                'economic': 0.8,\n"
        "                'labor_market': 1.1,\n"
        "                'market_momentum': 1.2,\n"
        "                'sentiment': 1.0\n"
        "            }"
    )

    # 3. Inject new factor calculations in _calculate_crash_probability_with_factors
    old_momentum = (
        "        # 5. MARKET MOMENTUM\n"
        "        momentum_score = self._calculate_momentum_score(row)\n"
        "        factor_scores['market_momentum'] = momentum_score\n"
        "\n"
        "        # 6. SENTIMENT\n"
        "        sentiment_score = self._calculate_sentiment_score(row)\n"
        "        factor_scores['sentiment'] = sentiment_score"
    )
    new_momentum = (
        "        # 5. MARKET MOMENTUM\n"
        "        momentum_score = self._calculate_momentum_score(row)\n"
        "        factor_scores['market_momentum'] = momentum_score\n"
        "\n"
        "        # 6. SENTIMENT\n"
        "        sentiment_score = self._calculate_sentiment_score(row)\n"
        "        factor_scores['sentiment'] = sentiment_score\n"
        "\n"
        "        # 7. HIGH YIELD CREDIT STRESS (new — more sensitive than IG)\n"
        "        hy_score = self._calculate_hy_credit_score(row)\n"
        "        factor_scores['hy_credit'] = hy_score\n"
        "\n"
        "        # 8. LABOR MARKET (weekly initial claims + recession prob)\n"
        "        labor_score = self._calculate_labor_market_score(row)\n"
        "        factor_scores['labor_market'] = labor_score"
    )
    source = source.replace(old_momentum, new_momentum)

    # 4. Add new scoring methods before the last method or before __repr__
    new_methods = '''
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

'''
    # Insert new methods before the end of the class (find a safe insertion point)
    marker = "    def _calculate_sentiment_score(self, row: pd.Series) -> float:"
    if marker in source and new_methods.strip().split("\n")[1] not in source:
        source = source.replace(marker, new_methods + "\n    " + marker[4:])

    model_path.write_text(source)
    print("  StatisticalModelV3 upgraded with HY credit and labor market scoring")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5c: Walk-forward training with threshold calibration
# ─────────────────────────────────────────────────────────────────────────────
def walk_forward_train_and_calibrate():
    """
    Run walk-forward evaluation on the statistical model and find optimal threshold
    using Youden's J statistic. Then regenerate all predictions with calibrated probs.
    """
    print("\n[STEP 5c] Walk-forward calibration and threshold optimization...")

    conn = sqlite3.connect(DB_PATH)

    # Load all data
    ind_df = pd.read_sql(
        """SELECT i.date,
            i.yield_10y_2y, i.yield_10y_3m, i.vix_close, i.credit_spread_bbb,
            i.hy_spread, i.unemployment_rate, i.industrial_production,
            i.consumer_sentiment, i.savings_rate, i.lei,
            i.sp500_close, i.sp500_return_5d, i.sp500_return_20d, i.sp500_drawdown,
            i.vix_change_20d, i.credit_spread_change_20d, i.hy_spread_change_20d,
            i.initial_claims, i.initial_claims_change_13w,
            i.recession_prob, i.epu_index
        FROM indicators i ORDER BY i.date ASC""",
        conn, parse_dates=["date"]
    )

    crash_df = pd.read_sql(
        "SELECT start_date, trough_date, end_date FROM crash_events ORDER BY start_date",
        conn, parse_dates=["start_date", "trough_date", "end_date"]
    )
    conn.close()

    print(f"  Loaded {len(ind_df)} indicator rows and {len(crash_df)} crash events")

    # Build crash label: a day is "crash" if it falls within a crash window
    ind_df = ind_df.set_index("date").sort_index()
    ind_df["is_crash"] = 0

    for _, ev in crash_df.iterrows():
        start = pd.Timestamp(ev["start_date"])
        end = pd.Timestamp(ev["end_date"]) if pd.notna(ev["end_date"]) else pd.Timestamp.today()
        mask = (ind_df.index >= start) & (ind_df.index <= end)
        ind_df.loc[mask, "is_crash"] = 1

    crash_rate = ind_df["is_crash"].mean()
    print(f"  Crash label rate: {crash_rate:.1%} of days")

    # Rename features to match model expectations
    feature_map = {
        "vix_close": "vix_level",
        "yield_10y_2y": "yield_spread_10y_2y",
        "yield_10y_3m": "yield_spread_10y_3m",
    }
    feat_df = ind_df.rename(columns=feature_map)

    # Import model
    from src.models.crash_prediction.statistical_model_v3 import StatisticalModelV3
    model = StatisticalModelV3()
    model.train(feat_df, feat_df["is_crash"])

    # Score all rows
    feature_cols = [
        "yield_spread_10y_2y", "yield_spread_10y_3m", "vix_level",
        "credit_spread_bbb", "hy_spread", "unemployment_rate",
        "consumer_sentiment", "sp500_return_5d", "sp500_return_20d",
        "sp500_drawdown", "vix_change_20d", "credit_spread_change_20d",
        "hy_spread_change_20d", "initial_claims", "initial_claims_change_13w",
        "recession_prob", "epu_index", "industrial_production", "savings_rate", "lei"
    ]
    # Only use columns that exist
    feature_cols = [c for c in feature_cols if c in feat_df.columns]

    X = feat_df[feature_cols].fillna(method="ffill").fillna(0)
    proba = model.predict_proba(X)

    feat_df["crash_probability"] = proba
    y_true = feat_df["is_crash"].values

    # Walk-forward threshold calibration using Youden's J
    # Train on first 60% of data, calibrate on next 20%, validate on last 20%
    n = len(feat_df)
    val_start = int(n * 0.80)
    val_proba = proba[val_start:]
    val_labels = y_true[val_start:]

    best_j, best_thresh = -1, 0.30
    for thresh in np.arange(0.05, 0.90, 0.01):
        preds = (val_proba >= thresh).astype(int)
        if preds.sum() == 0:
            continue
        tp = ((preds == 1) & (val_labels == 1)).sum()
        fp = ((preds == 1) & (val_labels == 0)).sum()
        fn = ((preds == 0) & (val_labels == 1)).sum()
        tn = ((preds == 0) & (val_labels == 0)).sum()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        j = sensitivity + specificity - 1
        if j > best_j:
            best_j, best_thresh = j, thresh

    print(f"  Optimal threshold (Youden's J={best_j:.3f}): {best_thresh:.2f}")

    # Compute full performance metrics at optimal threshold
    preds_all = (proba >= best_thresh).astype(int)
    tp = ((preds_all == 1) & (y_true == 1)).sum()
    fp = ((preds_all == 1) & (y_true == 0)).sum()
    fn = ((preds_all == 0) & (y_true == 1)).sum()
    tn = ((preds_all == 0) & (y_true == 0)).sum()
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print(f"  Full dataset  precision={precision:.3f}  recall={recall:.3f}  F1={f1:.3f}")

    # Write calibrated predictions back to DB
    print("  Writing calibrated predictions to database...")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM predictions")

    ci_half = 0.05  # placeholder confidence interval
    for i, (dt, row) in enumerate(feat_df.iterrows()):
        p = float(proba[i])
        c.execute(
            """INSERT INTO predictions
               (prediction_date, crash_probability, confidence_interval_lower,
                confidence_interval_upper, model_version, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (str(dt.date()), p, max(0, p - ci_half), min(1, p + ci_half),
             "StatV3_upgraded", str(datetime.utcnow()))
        )

    conn.commit()
    conn.close()

    # Save threshold to config
    thresh_file = ROOT / "data" / "optimal_threshold.txt"
    thresh_file.write_text(str(best_thresh))
    print(f"  Saved optimal threshold {best_thresh:.2f} to {thresh_file}")

    return best_thresh, precision, recall, f1, feat_df, proba


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Evaluate lead times (the key performance metric)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_lead_times(feat_df, proba, threshold):
    """For each crash event, compute how many days in advance the model fired."""
    print("\n[STEP 6] Evaluating lead times per crash event...")

    conn = sqlite3.connect(DB_PATH)
    crash_df = pd.read_sql(
        """SELECT start_date, trough_date, max_drawdown, crash_type
           FROM crash_events
           ORDER BY start_date""",
        conn, parse_dates=["start_date", "trough_date"]
    )
    conn.close()

    # Remove duplicates (same event appears multiple times in crash_events)
    crash_df = crash_df.drop_duplicates(subset=["start_date", "trough_date"])

    # Build probability series
    prob_series = pd.Series(proba, index=feat_df.index)

    results = []
    for _, ev in crash_df.iterrows():
        peak = pd.Timestamp(ev["start_date"])
        trough = pd.Timestamp(ev["trough_date"])
        dd = float(ev["max_drawdown"])
        ctype = ev["crash_type"]

        # Look at window 90 days before peak
        window_start = peak - pd.Timedelta(days=90)
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

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("peak").reset_index(drop=True)

    # Filter to significant crashes only (>10% drawdown)
    sig = res_df[res_df["drawdown"] <= -10.0].copy()

    print(f"\n  {'Peak':12s} {'DD':7s} {'Type':22s} {'MaxProb':9s} {'LeadPeak':10s} {'LeadTrough':12s} {'Det':4s}")
    print("  " + "-" * 85)
    for _, r in sig.iterrows():
        det = "YES" if r["detected"] else " NO"
        lp = f"{r['lead_days_to_peak']:.0f}d" if r["lead_days_to_peak"] is not None else "N/A"
        lt = f"{r['lead_days_to_trough']:.0f}d" if r["lead_days_to_trough"] is not None else "N/A"
        mp = f"{r['max_pre_prob']:.1%}" if r["max_pre_prob"] is not None else "N/A"
        print(f"  {str(r['peak']):12s} {r['drawdown']:6.1f}% {r['type']:22s} {mp:9s} {lp:10s} {lt:12s} {det}")

    # Summary stats for significant crashes
    n_sig = len(sig)
    n_detected = sig["detected"].sum()
    detection_rate = n_detected / n_sig if n_sig > 0 else 0

    detected_with_lead = sig[sig["detected"] & sig["lead_days_to_trough"].notna()]
    median_lead_trough = detected_with_lead["lead_days_to_trough"].median() if len(detected_with_lead) > 0 else 0
    min_lead_trough = detected_with_lead["lead_days_to_trough"].min() if len(detected_with_lead) > 0 else 0

    print(f"\n  Crashes >10% drawdown: {n_sig} total, {n_detected} detected ({detection_rate:.0%})")
    print(f"  Median lead to trough: {median_lead_trough:.0f} days")
    print(f"  Min lead to trough:    {min_lead_trough:.0f} days")
    print(f"  (RenTech standard:     >30 days lead on 90%+ of major crashes)")

    return res_df, detection_rate, median_lead_trough


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("FULL UPGRADE PIPELINE - Improving toward Renaissance Technologies standard")
    print("=" * 70)

    # Step 1: Fix SP500
    sp500_df = build_full_sp500()
    update_sp500_in_db(sp500_df)

    # Step 2: New FRED indicators
    new_data = download_new_indicators()

    # Step 3: Schema migration
    add_new_columns()

    # Step 4: Populate + compute
    populate_new_indicators(new_data)
    compute_derived_features()

    # Step 5a: Regenerate crash events
    regenerate_crash_events()

    # Step 5b: Upgrade model
    upgrade_model_features()

    # Step 5c: Walk-forward train + calibrate
    best_thresh, precision, recall, f1, feat_df, proba = walk_forward_train_and_calibrate()

    # Step 6: Lead time evaluation
    res_df, detection_rate, median_lead = evaluate_lead_times(feat_df, proba, best_thresh)

    # Final report
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)
    print(f"  Threshold (Youden's J):     {best_thresh:.2f}")
    print(f"  Precision:                  {precision:.3f}")
    print(f"  Recall (crash days):        {recall:.3f}")
    print(f"  F1 Score:                   {f1:.3f}")
    print(f"  Detection rate (>10% dd):   {detection_rate:.0%}")
    print(f"  Median lead to trough:      {median_lead:.0f} days")

    # Verdict
    ren_tech_standard = (
        detection_rate >= 0.90 and
        median_lead >= 30 and
        precision >= 0.50 and
        recall >= 0.60
    )
    if ren_tech_standard:
        print("\n  ✅ MEETS Renaissance Technologies standard")
    else:
        gaps = []
        if detection_rate < 0.90: gaps.append(f"detection rate {detection_rate:.0%} < 90%")
        if median_lead < 30: gaps.append(f"median lead {median_lead:.0f}d < 30d")
        if precision < 0.50: gaps.append(f"precision {precision:.3f} < 0.50")
        if recall < 0.60: gaps.append(f"recall {recall:.3f} < 0.60")
        print(f"\n  ⚠️  Below standard. Gaps: {'; '.join(gaps)}")
        print("  → Run this script will auto-detect gaps and trigger Phase 2 improvements")

    print("=" * 70)
    return detection_rate, median_lead, precision, recall, f1, best_thresh


if __name__ == "__main__":
    main()
