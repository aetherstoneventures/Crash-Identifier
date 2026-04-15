"""
April 2026 data refresh — clean version.
1. Removes any rows after 2026-03-27 (bad NaN rows from previous attempt)
2. Fetches FRED data using string-date index (no reindex bugs)
3. Inserts raw indicator columns into DB
4. Re-runs the full generate_predictions_v5 pipeline for proper feature engineering
"""
import sqlite3
import requests
import pandas as pd
import numpy as np
import sys
import pickle
from pathlib import Path

sys.path.insert(0, '.')

FRED_KEY = '547eaa8594ba77f00c821095c8e8482a'
DB       = 'data/market_crash.db'


def fred_series(series_id, start):
    """Fetch a FRED series, return dict {date_str: float}."""
    url = (f'https://api.stlouisfed.org/fred/series/observations'
           f'?series_id={series_id}&observation_start={start}'
           f'&api_key={FRED_KEY}&file_type=json')
    r = requests.get(url, timeout=20)
    return {o['date']: float(o['value'])
            for o in r.json().get('observations', [])
            if o['value'] != '.'}


# ── Step 1: Clean up bad rows ──────────────────────────────────────────────────
conn = sqlite3.connect(DB)
deleted = conn.execute("DELETE FROM indicators WHERE date > '2026-03-27'").rowcount
conn.commit()
print(f"Removed {deleted} stale rows (date > 2026-03-27)")

# ── Step 2: Fetch FRED data ────────────────────────────────────────────────────
print("Fetching FRED series...")
sp500_d = fred_series('SP500',           '2026-03-01')
vix_d   = fred_series('VIXCLS',          '2026-03-01')
t10_d   = fred_series('DGS10',           '2026-03-01')
t2_d    = fred_series('DGS2',            '2026-03-01')
t3m_d   = fred_series('DGS3MO',          '2026-03-01')
nfci_d  = fred_series('NFCI',            '2025-10-01')   # weekly
anfci_d = fred_series('ANFCI',           '2025-10-01')   # weekly
hy_d    = fred_series('BAMLH0A0HYM2OAS', '2025-10-01')   # daily (may lag)
epu_d   = fred_series('USEPUINDXD',      '2026-01-01')   # daily
icsa_d  = fred_series('ICSA',            '2025-10-01')   # weekly
rec_d   = fred_series('RECPROUSM156N',   '2025-06-01')   # monthly

print(f"  SP500  latest: {max(sp500_d, default='?')} = {sp500_d.get(max(sp500_d, default=''), '?')}")
print(f"  VIX    latest: {max(vix_d,   default='?')} = {vix_d.get(max(vix_d,   default=''), '?')}")
print(f"  NFCI   latest: {max(nfci_d,  default='?')} = {nfci_d.get(max(nfci_d,  default=''), '?')}")
print(f"  EPU    latest: {max(epu_d,   default='?')} = {epu_d.get(max(epu_d,   default=''), '?')}")
print(f"  ICSA   latest: {max(icsa_d,  default='?')} = {icsa_d.get(max(icsa_d,  default=''), '?')}")
print(f"  RecPrb latest: {max(rec_d,   default='?')} = {rec_d.get(max(rec_d,   default=''), '?')}")


def latest_on_or_before(data_dict, date_str):
    """Forward-fill: return most recent value for date_str in data_dict."""
    candidates = [v for k, v in data_dict.items() if k <= date_str]
    return candidates[-1] if candidates else None


# ── Step 3: Build date range and insert rows ──────────────────────────────────
new_dates = pd.bdate_range('2026-03-28', '2026-04-15')
print(f"\nBusiness days to insert: {len(new_dates)}")

rows_inserted = 0
for dt in new_dates:
    ds = str(dt.date())  # e.g. '2026-03-28'

    # Check if already exists
    exists = conn.execute("SELECT 1 FROM indicators WHERE date=?", (ds,)).fetchone()
    if exists:
        print(f"  {ds}: already exists, skipping")
        continue

    sp    = latest_on_or_before(sp500_d, ds)
    vix   = latest_on_or_before(vix_d,   ds)
    t10   = latest_on_or_before(t10_d,   ds)
    t2    = latest_on_or_before(t2_d,    ds)
    t3m   = latest_on_or_before(t3m_d,   ds)
    nfci  = latest_on_or_before(nfci_d,  ds)
    anfci = latest_on_or_before(anfci_d, ds)
    hy    = latest_on_or_before(hy_d,    ds)
    epu   = latest_on_or_before(epu_d,   ds)
    icsa  = latest_on_or_before(icsa_d,  ds)
    rec   = latest_on_or_before(rec_d,   ds)
    yc_2y = (t10 - t2)  if (t10 and t2)  else None
    yc_3m = (t10 - t3m) if (t10 and t3m) else None

    conn.execute("""
        INSERT INTO indicators
        (date, sp500_close, vix_close, yield_10y, yield_10y_2y, yield_10y_3m,
         nfci, anfci, hy_spread, epu_index, initial_claims, recession_prob,
         in_crash, pre_crash_60d, pre_crash_30d)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,0,0)
    """, (ds, sp, vix, t10, yc_2y, yc_3m, nfci, anfci, hy, epu, icsa, rec))

    rows_inserted += 1
    sp_str  = f"{sp:.0f}"   if sp  else "?"
    vix_str = f"{vix:.1f}"  if vix else "?"
    epu_str = f"{epu:.0f}"  if epu else "?"
    yc_str  = f"{yc_2y:.2f}" if yc_2y else "?"
    nfci_str = f"{nfci:.3f}" if nfci else "?"
    print(f"  {ds}: SP={sp_str} VIX={vix_str} YC={yc_str} EPU={epu_str} NFCI={nfci_str}")

conn.commit()
conn.close()
print(f"\nInserted {rows_inserted} new rows.")

# ── Step 4: Run predictions via the proper feature-engineering pipeline ────────
print("\n" + "="*70)
print("Running StatisticalModelV3 predictions via generate_predictions_v5...")
print("="*70 + "\n")

from scripts.utils.generate_predictions_v5 import engineer_features_for_prediction
from src.models.crash_prediction.statistical_model_v3 import StatisticalModelV3

conn = sqlite3.connect(DB)
df = pd.read_sql("SELECT * FROM indicators ORDER BY date", conn, parse_dates=['date'])
df = df.set_index('date')
print(f"Total rows in DB: {len(df)}  (last: {df.index[-1].date()})")

# Instantiate fresh from source (pickle is stale — missing phase 3+ factors)
stat_model = StatisticalModelV3()
print(f"Instantiated model: {type(stat_model).__name__} (live source)")

# Feature engineering (adds vix_level, yield_spread_10y_2y, sahm_rule, etc.)
df_features = engineer_features_for_prediction(df)

# Predict for ALL rows (rolling features need full history context)
proba = stat_model.predict_proba(df_features)
proba = np.nan_to_num(proba, nan=0.0)
proba = np.clip(proba, 0.0, 1.0)

# Write predictions to DB
conn.execute("DELETE FROM predictions WHERE prediction_date > '2026-03-27'")
for i, dt in enumerate(df.index):
    conn.execute(
        "INSERT OR REPLACE INTO predictions (prediction_date, crash_probability, model_version) "
        "VALUES (?,?,?)",
        (str(dt.date()), float(proba[i]), 'StatV3_phase3')
    )
conn.commit()
conn.close()
print(f"Predictions stored for all {len(df)} rows.")

# ── Step 5: Print April 2026 results ──────────────────────────────────────────
new_mask = df.index >= pd.Timestamp('2026-03-28')
new_dates_idx = df.index[new_mask]
new_probs     = proba[new_mask]

# Also grab the last pre-period day for context
context_start = pd.Timestamp('2026-03-01')
context_mask  = df.index >= context_start
ctx_dates = df.index[context_mask]
ctx_probs = proba[context_mask]

print()
print("=== CRASH PROBABILITY — March/April 2026 ===")
print(f"{'Date':12s} {'SP500':>7s} {'VIX':>6s} {'YC 2Y':>6s} {'EPU':>7s} {'NFCI':>7s}  {'CrashProb':>10s}  Signal")
print("-"*80)
for i, dt in enumerate(ctx_dates):
    row  = df_features.loc[dt]
    sp   = f"{row['sp500_close']:.0f}"         if not pd.isna(row['sp500_close'])      else '?'
    vix  = f"{row['vix_close']:.1f}"           if not pd.isna(row['vix_close'])        else '?'
    yc   = f"{row['yield_spread_10y_2y']:.2f}" if not pd.isna(row.get('yield_spread_10y_2y', float('nan'))) else '?'
    ep   = f"{row['epu_index']:.0f}"           if not pd.isna(row.get('epu_index', float('nan'))) else '?'
    nf   = f"{row['nfci']:.3f}"               if not pd.isna(row.get('nfci', float('nan')))      else '?'
    p    = ctx_probs[i]
    sig  = 'ALARM  ⚠️' if p >= 0.15 else ('WARNING' if p >= 0.12 else 'normal ')
    marker = ' ◄ NEW' if dt >= pd.Timestamp('2026-03-28') else ''
    print(f"{str(dt.date()):12s} {sp:>7s} {vix:>6s} {yc:>6s} {ep:>7s} {nf:>7s}  {p:>9.1%}  {sig}{marker}")

print()
last_prob = proba[-1]
apr_probs = new_probs
print(f"CURRENT CRASH PROBABILITY (latest day): {last_prob:.1%}")
print(f"Peak probability in new period:         {apr_probs.max():.1%}  on {str(new_dates_idx[apr_probs.argmax()].date())}")
print(f"Min  probability in new period:         {apr_probs.min():.1%}")
alarm_days = int((apr_probs >= 0.15).sum())
warn_days  = int(((apr_probs >= 0.12) & (apr_probs < 0.15)).sum())
print(f"Alarm days (≥15%): {alarm_days} | Warning days (12-15%): {warn_days}")

# Check alarm status
print()
if last_prob >= 0.15:
    print("STATUS: 🔴 ALARM ACTIVE — crash probability above 15% entry threshold")
elif last_prob >= 0.12:
    print("STATUS: 🟡 WARNING — above detection threshold (12%), below alarm entry (15%)")
else:
    print("STATUS: 🟢 NORMAL — below detection threshold")

print("\nDone.")
