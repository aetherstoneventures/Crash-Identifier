"""Pull additional leading-indicator features for v5.1.

Adds:
  vxv_close         — 3-month VIX (term-structure denominator) [2007+]
  vxo_close         — old VIX (1986-2021), used to backfill before VIXCLS
  aaa_10y_spread    — Aaa - 10Y daily [1983+]
  nfci_leverage     — NFCI leverage subindex [1971+ weekly]
  nfci_risk         — NFCI risk subindex
  nfci_credit       — NFCI credit subindex
  kcfsi             — KC Financial Stress Index [1990+ monthly]
  umcsent           — UMich consumer sentiment [1952+ monthly]
  t10yie            — 10Y breakeven inflation [2003+]
  dfii10            — TIPS 10Y real yield [2003+]
"""
import requests, sqlite3, time
import pandas as pd

DB = 'data/market_crash.db'
KEY = '547eaa8594ba77f00c821095c8e8482a'

SERIES = {
    'vxv_close':       'VXVCLS',
    'vxo_close':       'VXOCLS',
    'aaa_10y_spread':  'AAA10Y',
    'nfci_leverage':   'NFCILEVERAGE',
    'nfci_risk':       'NFCIRISK',
    'nfci_credit':     'NFCICREDIT',
    'kcfsi':           'KCFSI',
    'umcsent':         'UMCSENT',
    't10yie':          'T10YIE',
    'dfii10':          'DFII10',
}

def fetch(sid):
    url = (f'https://api.stlouisfed.org/fred/series/observations?series_id={sid}'
           f'&api_key={KEY}&file_type=json&observation_start=1970-01-01')
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
    raise RuntimeError(sid)

conn = sqlite3.connect(DB)
existing_cols = [r[1] for r in conn.execute('PRAGMA table_info(indicators)').fetchall()]
for col in SERIES:
    if col not in existing_cols:
        conn.execute(f'ALTER TABLE indicators ADD COLUMN {col} REAL')
        print(f'  added column: {col}')
conn.commit()

dates = pd.read_sql('SELECT date FROM indicators ORDER BY date', conn, parse_dates=['date'])['date']
all_dates = pd.DatetimeIndex(sorted(dates.unique()))
frame = pd.DataFrame(index=all_dates)
for col, sid in SERIES.items():
    print(f'Fetching {sid} -> {col} ...')
    s = fetch(sid)
    print(f'  {len(s)} obs   {s.index.min().date()} -> {s.index.max().date()}')
    frame[col] = s.reindex(all_dates)

print('\nUpserting...')
cols_csv = ','.join(SERIES); ph = ','.join('?' * len(SERIES))
n = 0
for dt, row in frame.iterrows():
    vals = tuple(None if pd.isna(v) else float(v) for v in row)
    conn.execute(
        f'UPDATE indicators SET {", ".join(c+"=?" for c in SERIES)} WHERE date=?',
        vals + (dt.strftime('%Y-%m-%d'),))
    n += 1
conn.commit()
print(f'Updated {n} rows')

for col in SERIES:
    nn = conn.execute(f'SELECT COUNT({col}) FROM indicators').fetchone()[0]
    print(f'  {col:18s}: {nn} non-null')
conn.close()
print('Done.')
