"""Refresh SP500 historical price data and fix the flat-line backfill bug.

PROBLEM:
  The sp500_cache.csv only covered 2016-03-28 onward.  When collect_data.py
  merged FRED indicator data (which starts 1982) with this cache and called
  .ffill().bfill(), the first cache value (2037.05, the 2016 SP500 level)
  was backfilled to every row from 1982-2015 — a flat line that makes crash
  detection blind to all pre-2016 market history.

WHAT THIS SCRIPT DOES:
  1. Downloads full SP500 history from Yahoo Finance (^GSPC, back to ~1927)
     with FRED (SP500 series, back to 2010) as fallback.
  2. Overwrites data/cache/sp500_cache.csv with the full history.
  3. Updates sp500_close in the indicators table for every row that currently
     has the stale constant value 2037.05 OR that could be enriched with
     actual price data.
  4. Re-runs populate_crash_events.py so that historical crashes (1987, 1990,
     2000-2002, 2008-2009) are now labelled in the crash_events table.

RUN THIS ONCE after restoring internet connectivity:
  python scripts/utils/refresh_sp500_history.py

Then re-run the full pipeline:
  python scripts/run_pipeline.sh       (or)
  python -m scripts.training.train_statistical_model_v3
"""

import sys
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

CACHE_FILE = project_root / "data" / "cache" / "sp500_cache.csv"
DB_PATH = project_root / "data" / "market_crash.db"

# Fetch window: 50 years
START_DATE = (datetime.now() - timedelta(days=50 * 365)).strftime("%Y-%m-%d")
END_DATE = datetime.now().strftime("%Y-%m-%d")


def fetch_yahoo(start: str, end: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf
        ticker = yf.Ticker("^GSPC")
        data = ticker.history(start=start, end=end, auto_adjust=True)
        if data is None or data.empty:
            return None
        df = pd.DataFrame({"date": data.index, "close": data["Close"].values})
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.dropna().sort_values("date").reset_index(drop=True)
        print(f"  Yahoo Finance: {len(df)} rows ({df['date'].min().date()} to {df['date'].max().date()})")
        return df
    except Exception as e:
        print(f"  Yahoo Finance failed: {e}")
        return None


def fetch_fred(start: str, end: str) -> pd.DataFrame | None:
    try:
        from fredapi import Fred
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            print("  FRED: no API key")
            return None
        fred = Fred(api_key=api_key)
        # FRED SP500 series starts 2010-01-04
        raw = fred.get_series("SP500", observation_start=start, observation_end=end)
        if raw is None or raw.empty:
            return None
        df = pd.DataFrame({"date": pd.to_datetime(raw.index), "close": raw.values})
        df = df.dropna().sort_values("date").reset_index(drop=True)
        print(f"  FRED SP500: {len(df)} rows ({df['date'].min().date()} to {df['date'].max().date()})")
        return df
    except Exception as e:
        print(f"  FRED failed: {e}")
        return None


def update_database(sp500_df: pd.DataFrame):
    """Update sp500_close in the indicators table with real historical prices."""
    import sqlite3

    sp500_map = {row["date"].date(): row["close"] for _, row in sp500_df.iterrows()}

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    updated = 0
    skipped = 0
    for dt, price in sp500_map.items():
        c.execute(
            "UPDATE indicators SET sp500_close = ? WHERE date = ?",
            (float(price), str(dt)),
        )
        if c.rowcount:
            updated += 1
        else:
            skipped += 1

    conn.commit()
    conn.close()
    print(f"  DB updated: {updated} rows changed, {skipped} dates not in indicators table")


def main():
    print("=" * 70)
    print("SP500 HISTORY REFRESH")
    print("=" * 70)

    print(f"\nFetching SP500 from {START_DATE} to {END_DATE}...")

    sp500_df = fetch_yahoo(START_DATE, END_DATE)
    if sp500_df is None or len(sp500_df) < 100:
        print("  Trying FRED fallback...")
        sp500_df = fetch_fred(START_DATE, END_DATE)

    if sp500_df is None or sp500_df.empty:
        print("\n❌ Could not fetch SP500 data — is internet connectivity available?")
        sys.exit(1)

    # Save to cache
    sp500_df.to_csv(CACHE_FILE, index=False)
    print(f"\n✅ Cache saved: {CACHE_FILE} ({len(sp500_df)} rows)")

    # Update database
    print("\nUpdating indicators table in database...")
    update_database(sp500_df)

    # Re-run crash event detection
    print("\nRe-detecting crash events from updated SP500 data...")
    result = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "data" / "populate_crash_events.py")],
        capture_output=True,
        text=True,
    )
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])
        print("⚠️  populate_crash_events.py exited with error — check output above")
    else:
        print("✅ Crash events refreshed")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Re-train the model to use the extended crash labels:")
    print("   python scripts/training/train_statistical_model_v3.py")
    print("2. Re-run the full pipeline:")
    print("   bash scripts/run_pipeline.sh")
    print("=" * 70)


if __name__ == "__main__":
    main()
