"""Final honest test: isolate bottom-finder's standalone value vs pure MA50."""
import sqlite3, json
import pandas as pd, numpy as np

DB='data/market_crash.db'
conn=sqlite3.connect(DB)
ind=pd.read_sql('SELECT * FROM indicators ORDER BY date', conn, parse_dates=['date']).set_index('date').sort_index()
preds=pd.read_sql('SELECT * FROM predictions', conn, parse_dates=['prediction_date'])
conn.close()
eq=ind['nasdaq_close'].ffill().dropna()
sig_v5=preds[preds.model_version=='v5'].set_index('prediction_date')['crash_probability'].sort_index().reindex(eq.index).ffill()
sig_bot=preds[preds.model_version=='v5.1_bot'].set_index('prediction_date')['crash_probability'].sort_index().reindex(eq.index).ffill()

v5cfg=json.loads(open('data/alarm_config_v5.json').read())
def hyst(p, idx, en, ex, mn, mx):
    a=[]; on=False; si=None; pv=p.values
    for i in range(len(pv)):
        if not on:
            if pv[i]>=en: on=True; si=i
        else:
            d=i-si
            if pv[i]<ex or d>=mx:
                if d>=mn: a.append((idx[si], idx[i-1])); 
                on=False
    if on and si is not None and (len(pv)-si)>=mn: a.append((idx[si], idx[-1]))
    return a
alms=hyst(sig_v5, sig_v5.index, v5cfg['entry'], v5cfg['exit'], v5cfg['min_dur'], v5cfg['max_dur'])
am=pd.Series(False, index=sig_v5.index)
for s,e in alms: am.loc[s:e]=True

def run(prices, am, mode, th=0.5, min_off=5, cost=5/10000):
    rets=prices.pct_change().fillna(0); ma50=prices.rolling(50, min_periods=20).mean()
    in_pos=True; days_off=0; eqv=[1.0]
    for i in range(1, len(prices)):
        prev=in_pos
        if am.iloc[i]: in_pos=False; days_off=0
        else:
            days_off=days_off+1 if not in_pos else days_off
            if not in_pos and days_off>=min_off:
                bs=sig_bot.iloc[i]
                if mode=='ma50_only':       reentry = prices.iloc[i] > ma50.iloc[i]
                elif mode=='bot_only':      reentry = (not pd.isna(bs)) and (bs >= th)
                elif mode=='bot_or_ma50':   reentry = ((not pd.isna(bs)) and (bs >= th)) or (prices.iloc[i] > ma50.iloc[i])
                elif mode=='bot_and_ma50':  reentry = ((not pd.isna(bs)) and (bs >= th)) and (prices.iloc[i] > ma50.iloc[i])
                if reentry: in_pos=True
        r=rets.iloc[i] if prev else 0.0
        if prev != in_pos: r -= cost
        eqv.append(eqv[-1]*(1+r))
    return pd.Series(eqv, index=prices.index)

def met(eqv, s, e):
    if s: eqv=eqv.loc[pd.Timestamp(s):]
    if e: eqv=eqv.loc[:pd.Timestamp(e)]
    eqv=eqv/eqv.iloc[0]
    yrs=(eqv.index[-1]-eqv.index[0]).days/365.25
    cagr=eqv.iloc[-1]**(1/max(yrs,1e-6))-1
    rets=eqv.pct_change().dropna()
    sh=rets.mean()/rets.std()*np.sqrt(252) if rets.std()>0 else 0
    mdd=(eqv/eqv.cummax()-1).min()
    return cagr, sh, mdd, eqv.iloc[-1]

print(f'{"Mode":>22} {"th":>5} | {"FULL CAGR":>9} {"Sharpe":>7} {"MaxDD":>7} {"Final":>8} | {"BLIND CAGR":>10} {"Sharpe":>7} {"MaxDD":>7}')
for mode in ['ma50_only','bot_only','bot_or_ma50','bot_and_ma50']:
    for th in ([0.0] if mode in ('ma50_only',) else [0.40,0.60,0.80]):
        eqv=run(eq, am, mode, th=th, min_off=5)
        f=met(eqv, '2000-01-01', None); b=met(eqv, '2021-01-01', None)
        print(f'{mode:>22} {th:>5.2f} | {f[0]*100:>8.1f}% {f[1]:>6.2f} {f[2]*100:>6.1f}% {f[3]:>7.2f}x | {b[0]*100:>9.1f}% {b[1]:>6.2f} {b[2]*100:>6.1f}%')
