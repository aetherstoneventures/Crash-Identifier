"""Surgical data cleanup — purge ONLY obvious spike anomalies.

Rule: A row is a "spike" if its sp500_close is >10% away from BOTH the previous valid
      close AND the next valid close, AND the neighbors agree with each other within 5%.
      No legitimate market move does this — even 1987 and 2008 are sustained moves.

Plus: strip SP500 values from fixed US market holidays (NYSE-closed days).
"""
import sqlite3
import pandas as pd
from pathlib import Path

DB = 'data/market_crash.db'
conn = sqlite3.connect(DB)

backup_path = Path('data/backups/indicators_prepurge.csv')
backup_path.parent.mkdir(parents=True, exist_ok=True)
pd.read_sql('SELECT * FROM indicators', conn).to_csv(backup_path, index=False)
print(f'Backed up → {backup_path}')

ind = pd.read_sql('SELECT date, sp500_close FROM indicators ORDER BY date', conn, parse_dates=['date'])
sp = ind['sp500_close'].copy()
prev = sp.shift(1)
nxt  = sp.shift(-1)
d_prev = (sp / prev - 1).abs()
d_next = (sp / nxt  - 1).abs()
prev_next_agree = (prev / nxt - 1).abs() < 0.05
spike = (d_prev > 0.10) & (d_next > 0.10) & prev_next_agree
spike_dates = ind.loc[spike, 'date']
print(f'\nRound-trip spike anomalies: {len(spike_dates)}')
for d in spike_dates:
    v = sp[ind.date == d].iloc[0]
    p = prev[ind.date == d].iloc[0]
    n = nxt[ind.date == d].iloc[0]
    print(f'  {d.date()}: sp500={v:.1f}  prev={p:.1f}  next={n:.1f}')

if len(spike_dates) > 0:
    dates_str = tuple(d.strftime('%Y-%m-%d') for d in spike_dates)
    ph = ','.join('?' * len(dates_str))
    conn.execute(f'UPDATE indicators SET sp500_close=NULL WHERE date IN ({ph})', dates_str)
    conn.commit()
    print(f'→ Purged sp500_close on {len(dates_str)} spike dates.')

# Fixed US market holidays
holidays = []
for year in range(2000, 2027):
    holidays += [f'{year}-01-01', f'{year}-07-04', f'{year}-12-25']
    if year >= 2022:
        holidays.append(f'{year}-06-19')
bad_hol = ind[ind.date.isin(pd.to_datetime(holidays)) & ind.sp500_close.notna()]
if len(bad_hol) > 0:
    print(f'\nFixed-holiday rows with SP500 data: {len(bad_hol)}')
    dates_str = tuple(d.strftime('%Y-%m-%d') for d in bad_hol.date)
    ph = ','.join('?' * len(dates_str))
    conn.execute(f'UPDATE indicators SET sp500_close=NULL WHERE date IN ({ph})', dates_str)
    conn.commit()
    print(f'→ Purged sp500_close on {len(dates_str)} fixed-holiday rows.')

# Verify
ind2 = pd.read_sql('SELECT date, sp500_close FROM indicators WHERE sp500_close IS NOT NULL ORDER BY date',
                   conn, parse_dates=['date']).set_index('date')
ret = ind2['sp500_close'].pct_change()
big = ret[ret.abs() > 0.15]
print(f'\nRemaining day-over-day moves > 15%: {len(big)}')
for dt, r in big.items():
    i = ind2.index.get_loc(dt)
    ctx_prev = ind2.iloc[max(0, i-2):i]['sp500_close'].tolist()
    ctx_next = ind2.iloc[i+1:i+3]['sp500_close'].tolist()
    print(f'  {dt.date()}  {r:+.1%}   prev2={ctx_prev}  now={ind2.iloc[i].sp500_close}  next2={ctx_next}')

conn.close()
print('\nData cleanup complete.')
