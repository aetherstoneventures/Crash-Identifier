"""Fetch v5 feature set from FRED: long-history daily equity (NASDAQCOM) + credit/oil/dollar.

Adds new columns to indicators table (creates them if absent):
  nasdaq_close      — NASDAQ Composite daily since 1971 (replaces stale SP500 pre-2016)
  baa_10y_spread    — Moody's Baa minus 10Y (daily since 1986) — classic credit stress
  oil_wti           — WTI crude (daily since 1986) — geopolitical proxy
  dollar_twi        — Trade-weighted dollar broad index (daily since 2006)
  epu_daily         — Economic Policy Uncertainty daily (since 1985) — geopolitical
"""
import requests, sqlite3, time
import pandas as pd
from pathlib import Path

DB = 'data/market_crash.db'
KEY = '547eaa8594ba77f00c821095c8e8482a'

SERIES = {
    'nasdaq_close':    'NASDAQCOM',
    'baa_10y_spread':  'BAA10Y',
    'oil_wti':         'DCOILWTICO',
    'dollar_twi':      'DTWEXBGS',
    'epu_daily':       'USEPUINDXD',
}

def fetch(sid: str) -> pd.DataFrame:
    url = ('https://api.stlouisfed.org/fred/series/observations'
           f'?series_id={sid}&api_key={KEY}&file_type=json&observation_start=1980-01-01')
    for _ in range(3):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                obs = r.json()['observations']
                df = pd.DataFrame(obs)[['date','value']]
                df['date'] = pd.to_datetime(df['date'])
                df = df[df['value'] != '.'].copy()
                df['value'] = df['value'].astype(float)
                return df.set_index('date')['value']
        except Exception as e:
            print(f'  retry {sid}: {e}'); time.sleep(2)
    raise RuntimeError(f'Failed to fetch {sid}')

conn = sqlite3.connect(DB)

# Ensure columns exist
cols = [r[1] for r in conn.execute('PRAGMA table_info(indicators)').fetchall()]
for col in SERIES:
    if col not in cols:
        conn.execute(f'ALTER TABLE indicators ADD COLUMN {col} REAL')
        print(f'  added column: {col}')
conn.commit()

# Load existing dates
existing = pd.read_sql('SELECT date FROM indicators ORDER BY date', conn, parse_dates=['date'])['date']
min_d, max_d = existing.min(), existing.max()
print(f'Existing indicator rows span {min_d.date()} → {max_d.date()}')

# Fetch each series
series_data = {}
for col, sid in SERIES.items():
    print(f'Fetching {sid} → {col} ...')
    s = fetch(sid)
    print(f'  got {len(s)} obs  {s.index.min().date()} → {s.index.max().date()}')
    series_data[col] = s

# Build dense frame indexed by all existing indicator dates (preserve schema)
all_dates = pd.DatetimeIndex(sorted(existing.unique()))
frame = pd.DataFrame(index=all_dates)
for col, s in series_data.items():
    frame[col] = s.reindex(all_dates)

# Also reindex to business-day grid to fill gaps — but keep at existing dates only for now
# Upsert into DB
print('\nUpdating indicators table...')
rows_updated = 0
for dt, row in frame.iterrows():
    vals = {c: (None if pd.isna(v) else float(v)) for c, v in row.items()}
    conn.execute(
        f'UPDATE indicators SET {", ".join(c+"=?" for c in SERIES)} WHERE date=?',
        tuple(vals[c] for c in SERIES) + (dt.strftime('%Y-%m-%d'),)
    )
    rows_updated += 1
conn.commit()
print(f'Updated {rows_updated} rows')

# Sanity check: count non-null per column
for col in SERIES:
    n = conn.execute(f'SELECT COUNT({col}) FROM indicators').fetchone()[0]
    print(f'  {col}: {n} non-null rows')

# Verify NASDAQ daily resolution 1987, 2000, 2008, 2020
q = pd.read_sql("""SELECT date, nasdaq_close FROM indicators
                   WHERE nasdaq_close IS NOT NULL ORDER BY date""",
                conn, parse_dates=['date']).set_index('date')
print('\nNASDAQ unique values per year (spot-check):')
for y in [1987, 2000, 2008, 2020, 2025]:
    n = q[q.index.year == y]['nasdaq_close'].nunique()
    print(f'  {y}: {n} unique values  ({len(q[q.index.year == y])} rows)')

conn.close()
print('\nDone.')
