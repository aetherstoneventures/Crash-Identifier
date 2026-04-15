"""April 2026 data refresh + live regime assessment.
Pulls FRED data since March 28, updates the indicators table, re-runs predictions.
"""
import sqlite3
import requests
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

FRED_KEY = '547eaa8594ba77f00c821095c8e8482a'
DB = 'data/market_crash.db'

def fred(series, start='2026-03-28'):
    url = (f'https://api.stlouisfed.org/fred/series/observations'
           f'?series_id={series}&observation_start={start}'
           f'&api_key={FRED_KEY}&file_type=json')
    r = requests.get(url, timeout=20)
    obs = r.json().get('observations', [])
    s = pd.Series({o['date']: o['value'] for o in obs if o['value'] != '.'})
    return pd.to_numeric(s, errors='coerce').dropna()

print("Fetching FRED data since March 28, 2026...")

sp500  = fred('SP500')
vix    = fred('VIXCLS')
t10    = fred('DGS10')
t2     = fred('DGS2')
t3m    = fred('DGS3MO')
nfci   = fred('NFCI',   '2025-10-01')
anfci  = fred('ANFCI',  '2025-10-01')
hy     = fred('BAMLH0A0HYM2OAS', '2025-10-01')
epu    = fred('USEPUINDXD', '2026-01-01')
icsa   = fred('ICSA',   '2025-10-01')
recprob= fred('RECPROUSM156N', '2025-01-01')

print(f"  SP500  : {len(sp500)} rows, latest {sp500.index[-1] if len(sp500) else 'N/A'} = {sp500.iloc[-1] if len(sp500) else 'N/A'}")
print(f"  VIX    : {len(vix)} rows, latest {vix.index[-1] if len(vix) else 'N/A'} = {vix.iloc[-1] if len(vix) else 'N/A'}")
print(f"  NFCI   : {len(nfci)} rows, latest {nfci.index[-1] if len(nfci) else 'N/A'} = {nfci.iloc[-1] if len(nfci) else 'N/A'}")
print(f"  EPU    : {len(epu)} rows, latest {epu.index[-1] if len(epu) else 'N/A'} = {epu.iloc[-1] if len(epu) else 'N/A'}")
print(f"  ICSA   : {len(icsa)} rows, latest {icsa.index[-1] if len(icsa) else 'N/A'} = {icsa.iloc[-1] if len(icsa) else 'N/A'}")
print()

# ── Build a daily date range of new business days ─────────────────────────────
new_dates = pd.bdate_range('2026-03-28', '2026-04-15')

# Forward-fill weekly series to daily
def ffill_to_daily(s, date_range):
    daily = s.reindex(pd.DatetimeIndex(s.index.astype('datetime64[ns]')))
    daily = daily.reindex(date_range).ffill()
    return daily

sp_d    = sp500.reindex(pd.DatetimeIndex(sp500.index.astype('datetime64[ns]'))).reindex(new_dates, method='ffill')
vix_d   = vix.reindex(pd.DatetimeIndex(vix.index.astype('datetime64[ns]'))).reindex(new_dates, method='ffill')
t10_d   = t10.reindex(pd.DatetimeIndex(t10.index.astype('datetime64[ns]'))).reindex(new_dates, method='ffill')
t2_d    = t2.reindex(pd.DatetimeIndex(t2.index.astype('datetime64[ns]'))).reindex(new_dates, method='ffill')
t3m_d   = t3m.reindex(pd.DatetimeIndex(t3m.index.astype('datetime64[ns]'))).reindex(new_dates, method='ffill')
nfci_d  = nfci.reindex(pd.DatetimeIndex(nfci.index.astype('datetime64[ns]'))).reindex(new_dates, method='ffill')
anfci_d = anfci.reindex(pd.DatetimeIndex(anfci.index.astype('datetime64[ns]'))).reindex(new_dates, method='ffill')
hy_d    = hy.reindex(pd.DatetimeIndex(hy.index.astype('datetime64[ns]'))).reindex(new_dates, method='ffill')
epu_d   = epu.reindex(pd.DatetimeIndex(epu.index.astype('datetime64[ns]'))).reindex(new_dates, method='ffill')
icsa_d  = icsa.reindex(pd.DatetimeIndex(icsa.index.astype('datetime64[ns]'))).reindex(new_dates, method='ffill')
rec_d   = recprob.reindex(pd.DatetimeIndex(recprob.index.astype('datetime64[ns]'))).reindex(new_dates, method='ffill')

yc_d    = t10_d - t2_d   # yield curve 10Y-2Y

# Pull existing indicators to compute rolling features
conn = sqlite3.connect(DB)
hist = pd.read_sql(
    "SELECT date, sp500_close, vix_close, yield_10y_2y, hy_spread, nfci, anfci, "
    "epu_index, initial_claims, sp500_return_5d, sp500_return_20d, sp500_drawdown, "
    "vix_change_20d, credit_spread_change_20d, hy_spread_change_20d, "
    "sp500_shock_5d, vix_momentum_5d, credit_momentum_5d, stress_composite, "
    "epu_ma_90d, epu_acceleration, yield_curve_velocity_63d, yield_curve_velocity_120d "
    "FROM indicators ORDER BY date",
    conn, parse_dates=['date']
)
hist = hist.set_index('date')

# ── Append new rows ────────────────────────────────────────────────────────────
existing_max = hist.index.max()
all_sp = hist['sp500_close'].copy()
all_vix = hist['vix_close'].copy()
all_yc  = hist['yield_10y_2y'].copy()
all_hy  = hist['hy_spread'].copy()
all_nfci= hist['nfci'].copy()
all_anfci=hist['anfci'].copy()
all_epu = hist['epu_index'].copy()

new_rows = []
for dt in new_dates:
    dt_str = str(dt.date())
    if dt <= existing_max:
        continue

    sp_val   = sp_d.get(dt, np.nan)
    vix_val  = vix_d.get(dt, np.nan)
    yc_val   = yc_d.get(dt, np.nan)
    hy_val   = hy_d.get(dt, np.nan)
    nfci_val = nfci_d.get(dt, np.nan)
    anfci_val= anfci_d.get(dt, np.nan)
    epu_val  = epu_d.get(dt, np.nan)
    icsa_val = icsa_d.get(dt, np.nan)
    rec_val  = rec_d.get(dt, np.nan)

    # Append to rolling series for derived features
    all_sp[dt]   = sp_val
    all_vix[dt]  = vix_val
    all_yc[dt]   = yc_val
    all_hy[dt]   = hy_val
    all_nfci[dt] = nfci_val
    all_anfci[dt]= anfci_val
    all_epu[dt]  = epu_val

    # Derived features
    sp_5d  = all_sp.iloc[-6:-1].mean() if len(all_sp) >= 6 else np.nan
    sp_20d = all_sp.iloc[-21:-1].mean() if len(all_sp) >= 21 else np.nan
    sp_ret5  = (sp_val / sp_5d  - 1) if not np.isnan(sp_5d)  and sp_5d > 0 else np.nan
    sp_ret20 = (sp_val / sp_20d - 1) if not np.isnan(sp_20d) and sp_20d > 0 else np.nan

    sp_peak = all_sp.expanding().max().iloc[-1]
    drawdown = (sp_val / sp_peak - 1) if not np.isnan(sp_val) and sp_peak > 0 else np.nan

    vix_20d = all_vix.iloc[-21:-1].mean() if len(all_vix) >= 21 else np.nan
    vix_chg  = (vix_val / vix_20d - 1) if not np.isnan(vix_20d) and vix_20d > 0 else np.nan

    hy_20d  = all_hy.iloc[-21:-1].mean() if len(all_hy) >= 21 else np.nan
    hy_chg  = (hy_val  / hy_20d  - 1) if not np.isnan(hy_20d)  and hy_20d > 0 else np.nan

    sp_shock = (sp_val / all_sp.iloc[-6] - 1) if len(all_sp) >= 6 and all_sp.iloc[-6] > 0 else np.nan
    vix_mom  = (vix_val / all_vix.iloc[-6] - 1) if len(all_vix) >= 6 and all_vix.iloc[-6] > 0 else np.nan
    hy_mom   = (hy_val  / all_hy.iloc[-6]  - 1) if len(all_hy)  >= 6 and all_hy.iloc[-6]  > 0 else np.nan

    stress = None
    if not any(np.isnan(x) for x in [nfci_val, anfci_val, vix_val]):
        stress = (nfci_val + anfci_val) / 2 + (vix_val / 20 - 1)

    epu_90d = all_epu.iloc[-91:-1].mean() if len(all_epu) >= 91 else all_epu.mean()
    epu_accel= (epu_val - epu_90d) if not np.isnan(epu_90d) else np.nan

    yc_vel63  = (yc_val - all_yc.iloc[-64]) if len(all_yc) >= 64 else np.nan
    yc_vel120 = (yc_val - all_yc.iloc[-121]) if len(all_yc) >= 121 else np.nan

    new_rows.append({
        'date': dt_str,
        'sp500_close': sp_val, 'vix_close': vix_val,
        'yield_10y_2y': yc_val, 'hy_spread': hy_val,
        'nfci': nfci_val, 'anfci': anfci_val,
        'epu_index': epu_val, 'initial_claims': icsa_val,
        'recession_prob': rec_val,
        'sp500_return_5d': sp_ret5, 'sp500_return_20d': sp_ret20,
        'sp500_drawdown': drawdown, 'vix_change_20d': vix_chg,
        'hy_spread_change_20d': hy_chg,
        'sp500_shock_5d': sp_shock, 'vix_momentum_5d': vix_mom,
        'credit_momentum_5d': hy_mom, 'stress_composite': stress,
        'epu_ma_90d': epu_90d, 'epu_acceleration': epu_accel,
        'yield_curve_velocity_63d': yc_vel63,
        'yield_curve_velocity_120d': yc_vel120,
    })

print(f"New rows to insert: {len(new_rows)}")

if new_rows:
    for row in new_rows:
        placeholders = ', '.join(['?' for _ in row])
        cols = ', '.join(row.keys())
        vals = list(row.values())
        conn.execute(
            f"INSERT OR REPLACE INTO indicators ({cols}) VALUES ({placeholders})",
            vals
        )
    conn.commit()
    print(f"Inserted {len(new_rows)} rows into indicators.")

# ── Re-run StatisticalModelV3 on new rows ──────────────────────────────────────
from src.models.crash_prediction.statistical_model_v3 import StatisticalModelV3
model = StatisticalModelV3()

new_preds = []
for row in new_rows:
    try:
        row_s = pd.Series(row)
        row_df = pd.DataFrame([row_s])
        prob = model.predict_proba(row_df)[0]
        new_preds.append({'date': row['date'], 'prob': prob, 'row': row})
    except Exception as e:
        print(f"  Prediction error {row['date']}: {e}")

print()
print("=== PREDICTIONS: March 28 – April 15, 2026 ===")
print(f"{'Date':12s} {'SP500':8s} {'VIX':6s} {'YC':6s} {'EPU':7s} {'CrashProb':10s} {'Signal'}")
print("-" * 70)
for p in new_preds:
    r = p['row']
    sp_v  = f"{r['sp500_close']:.0f}"  if not np.isnan(r['sp500_close']  or float('nan')) else '?'
    vix_v = f"{r['vix_close']:.1f}"    if not np.isnan(r['vix_close']    or float('nan')) else '?'
    yc_v  = f"{r['yield_10y_2y']:.2f}" if not np.isnan(r['yield_10y_2y'] or float('nan')) else '?'
    epu_v = f"{r['epu_index']:.0f}"    if r['epu_index'] and not np.isnan(r['epu_index']) else '?'
    prob  = p['prob']
    signal = 'ALARM' if prob >= 0.15 else ('WARNING' if prob >= 0.12 else 'NORMAL')
    print(f"{p['date']:12s} {sp_v:>8s} {vix_v:>6s} {yc_v:>6s} {epu_v:>7s}  {prob:.1%}   {signal}")

    # Write to predictions table
    conn.execute(
        "INSERT OR REPLACE INTO predictions (prediction_date, crash_probability, model_version) "
        "VALUES (?, ?, ?)",
        (p['date'], prob, 'StatV3_phase3')
    )
conn.commit()
conn.close()

if new_preds:
    probs = [p['prob'] for p in new_preds]
    latest = new_preds[-1]
    print()
    print(f"CURRENT CRASH PROBABILITY (April 15): {latest['prob']:.1%}")
    print(f"Peak probability in April:             {max(probs):.1%}")
    print(f"Min probability in April:              {min(probs):.1%}")
    print()
    alarm_days = sum(1 for p in probs if p >= 0.15)
    warn_days  = sum(1 for p in probs if 0.12 <= p < 0.15)
    print(f"Days in ALARM  (>=15%): {alarm_days}")
    print(f"Days in WARNING (12-15%): {warn_days}")
print("\nDone. Database updated.")
