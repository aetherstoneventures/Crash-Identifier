"""
Pareto sweep of v5.1 alarm entry threshold:
  For each ENTRY in {0.25..0.55}, compute (on TUNE only, 1999-2020):
    - lead_from_peak (median, days)
    - day_precision, day_recall
    - alarm-day pct
    - CAGR if used as backtest signal (with bottom-finder re-entry)
  Then validate the chosen threshold on BLIND (2021-2026).

This lets the user pick the honest tradeoff, with no overfitting:
threshold and bot_thresh chosen on TUNE; reported metrics on BLIND.
"""
import sqlite3, pickle, json
import pandas as pd, numpy as np

DB = 'data/market_crash.db'
TUNE_END   = pd.Timestamp('2020-12-31')
TEST_START = pd.Timestamp('2021-01-01')
COST_BPS = 5
EXIT, MIN_DUR, MAX_DUR, MF = 0.20, 10, 30, 1

conn = sqlite3.connect(DB)
ind = pd.read_sql('SELECT * FROM indicators ORDER BY date', conn, parse_dates=['date']).set_index('date').sort_index()
preds = pd.read_sql('SELECT * FROM predictions', conn, parse_dates=['prediction_date'])
conn.close()
eq = ind['nasdaq_close'].ffill().dropna()
sig = preds[preds.model_version=='v5.1'].set_index('prediction_date')['crash_probability'].sort_index().reindex(eq.index).ffill()
sig_bot = preds[preds.model_version=='v5.1_bot'].set_index('prediction_date')['crash_probability'].sort_index().reindex(eq.index).ffill()

# Crash labels
def compute_crash_labels(prices, dd=0.15, min_td=30):
    p = prices.ffill(); n = len(p); pv = p.values
    rm = p.rolling(252, min_periods=50).max().values
    below = (pv/rm - 1) <= -dd
    crash = np.zeros(n, dtype=bool); i=0
    while i<n:
        if not below[i]: i+=1; continue
        rs=i
        while i<n and below[i]: i+=1
        re=i-1
        tp = rs + int(np.argmin(pv[rs:re+1]))
        peak_val = rm[rs]; pp = rs
        for j in range(rs-1, max(0, rs-260), -1):
            if pv[j] >= peak_val * 0.998: pp = j; break
        if tp - pp >= min_td: crash[pp:tp+1] = True
    return pd.Series(crash, index=p.index)
crash = compute_crash_labels(eq)

def get_events(lbl, prices):
    out=[]; on=False; cs=None
    for d, v in lbl.items():
        if not on and v: on, cs = True, d
        elif on and not v: out.append({'start':cs,'end':d}); on=False
    if on: out.append({'start':cs,'end':lbl.index[-1]})
    df = pd.DataFrame(out)
    if df.empty: return df
    df['peak']     = df.apply(lambda r: prices.loc[r.start:r.end].idxmax(), axis=1)
    df['trough']   = df.apply(lambda r: prices.loc[r.start:r.end].idxmin(), axis=1)
    df['drawdown'] = df.apply(lambda r: prices.loc[r.trough]/prices.loc[r.peak]-1, axis=1)
    return df
events = get_events(crash, eq)

def hysteresis(p, idx, en, ex, mn, mx):
    a=[]; on=False; si=None; pv=p.values
    for i in range(len(pv)):
        if not on:
            if pv[i] >= en: on=True; si=i
        else:
            d=i-si
            if pv[i]<ex or d>=mx:
                if d>=mn: a.append((idx[si], idx[i-1]))
                on=False
    if on and si is not None and (len(pv)-si)>=mn: a.append((idx[si], idx[-1]))
    return a

def alarm_mask(alms, idx):
    m = pd.Series(False, index=idx)
    for s,e in alms: m.loc[s:e] = True
    return m

def eval_alarms(alms, evts, idx, lbl):
    if not alms: return dict(ev_det=0, ev_prec=0, day_prec=0, day_rec=0, adp=0, lead=0, n=0)
    am = alarm_mask(alms, idx)
    y = lbl.reindex(idx).fillna(False).astype(bool)
    tp=int((am&y).sum()); fp=int((am&~y).sum()); fn=int((~am&y).sum())
    dp=tp/max(tp+fp,1); dr=tp/max(tp+fn,1); adp=am.sum()/len(idx)
    n_conf=0
    for s,e in alms:
        if any(((s <= ce.peak) and (s >= ce.peak-pd.Timedelta(days=180))) or
               ((s <= ce.end) and (e >= ce.start)) for _, ce in evts.iterrows()):
            n_conf += 1
    ev_prec = n_conf/len(alms)
    leads=[]; n_det=0
    for _, ce in evts.iterrows():
        for s,e in alms:
            if (s <= ce.end+pd.Timedelta(days=30)) and (e >= ce.start-pd.Timedelta(days=180)):
                n_det += 1; leads.append((ce.peak - s).days); break
    return dict(ev_det=n_det/max(len(evts),1), ev_prec=ev_prec, day_prec=dp, day_rec=dr,
                adp=adp, lead=int(np.median(leads)) if leads else 0, n=len(alms))

def backtest(prices, am, bot, bot_th, cost_bps=COST_BPS):
    rets = prices.pct_change().fillna(0)
    ma50 = prices.rolling(50, min_periods=20).mean()
    in_pos=True; days_off=0; eqv=[1.0]; cost=cost_bps/10000.0
    for i in range(1, len(prices)):
        prev = in_pos
        if am.iloc[i]:
            in_pos=False; days_off=0
        else:
            days_off = days_off+1 if not in_pos else days_off
            if not in_pos and days_off >= 5:
                bs=bot.iloc[i]
                if (not pd.isna(bs) and bs>=bot_th) or (prices.iloc[i] > ma50.iloc[i]):
                    in_pos=True
        r = rets.iloc[i] if prev else 0.0
        if prev != in_pos: r -= cost
        eqv.append(eqv[-1]*(1+r))
    return pd.Series(eqv, index=prices.index)

def metrics(eqv, lbl, start=None, end=None):
    if start: eqv = eqv.loc[pd.Timestamp(start):]
    if end:   eqv = eqv.loc[:pd.Timestamp(end)]
    if len(eqv) < 2: return {'cagr':0,'sharpe':0,'mdd':0,'final':1}
    eqv = eqv / eqv.iloc[0]
    yrs=(eqv.index[-1]-eqv.index[0]).days/365.25
    cagr=(eqv.iloc[-1])**(1/max(yrs,1e-6))-1
    rets=eqv.pct_change().dropna()
    sh=rets.mean()/rets.std()*np.sqrt(252) if rets.std()>0 else 0
    mdd=(eqv/eqv.cummax()-1).min()
    return {'cagr':cagr,'sharpe':sh,'mdd':mdd,'final':eqv.iloc[-1]}

# ---- Sweep ----
print('='*120)
print(f'{"ENTRY":>6} | {"TUNE: lead":>10} {"ev_det":>7} {"ev_prec":>8} {"day_p":>6} {"adp":>6} {"CAGR":>6} {"Sharpe":>7} {"MaxDD":>7} | {"BLIND: lead":>11} {"ev_det":>7} {"day_p":>6} {"CAGR":>6} {"Sharpe":>7} {"MaxDD":>7}')
print('='*120)

ev_tune = events[events.peak <= TUNE_END]
ev_test = events[events.peak >= TEST_START]
sig_tune = sig[sig.index <= TUNE_END]
sig_test = sig[sig.index >= TEST_START]

# Test multiple bot_thresh too
results = []
for entry in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    alms_tune = hysteresis(sig_tune, sig_tune.index, entry, EXIT, MIN_DUR, MAX_DUR)
    alms_test = hysteresis(sig_test, sig_test.index, entry, EXIT, MIN_DUR, MAX_DUR)
    m_tune = eval_alarms(alms_tune, ev_tune, sig_tune.index, crash)
    m_test = eval_alarms(alms_test, ev_test, sig_test.index, crash)

    # find best bot_thresh on tune
    full_alms = hysteresis(sig, sig.index, entry, EXIT, MIN_DUR, MAX_DUR)
    am = alarm_mask(full_alms, sig.index)
    best_bt = None; best_tune_cagr = -1
    for bt in [0.30, 0.40, 0.50, 0.60]:
        eqv = backtest(eq, am, sig_bot, bt)
        m = metrics(eqv, crash, '2000-01-01', '2020-12-31')
        if m['cagr'] > best_tune_cagr: best_tune_cagr, best_bt = m['cagr'], bt
    eqv = backtest(eq, am, sig_bot, best_bt)
    m_t = metrics(eqv, crash, '2000-01-01', '2020-12-31')
    m_b = metrics(eqv, crash, '2021-01-01', None)
    results.append({'entry':entry,'bot':best_bt,'tune':m_tune,'test':m_test,'bt':m_t,'bb':m_b})
    print(f'{entry:>6.2f} | {m_tune["lead"]:>9}d {m_tune["ev_det"]:>6.0%} {m_tune["ev_prec"]:>7.0%} {m_tune["day_prec"]:>5.0%} {m_tune["adp"]:>5.0%} {m_t["cagr"]*100:>5.1f}% {m_t["sharpe"]:>6.2f} {m_t["mdd"]*100:>6.1f}% | {m_test["lead"]:>10}d {m_test["ev_det"]:>6.0%} {m_test["day_prec"]:>5.0%} {m_b["cagr"]*100:>5.1f}% {m_b["sharpe"]:>6.2f} {m_b["mdd"]*100:>6.1f}%')

# Buy-hold reference
bh = eq / eq.iloc[0]
m_bh_t = metrics(bh, crash, '2000-01-01', '2020-12-31')
m_bh_b = metrics(bh, crash, '2021-01-01', None)
print('-'*120)
print(f'{"BH":>6} |          --     --       --     --     -- {m_bh_t["cagr"]*100:>5.1f}% {m_bh_t["sharpe"]:>6.2f} {m_bh_t["mdd"]*100:>6.1f}% |          --     --     -- {m_bh_b["cagr"]*100:>5.1f}% {m_bh_b["sharpe"]:>6.2f} {m_bh_b["mdd"]*100:>6.1f}%')

# Choose best on tune Sharpe (must have ev_det>=80%)
elig = [r for r in results if r['tune']['ev_det'] >= 0.80]
if elig:
    best = max(elig, key=lambda r: r['bt']['sharpe'])
    print(f'\nChosen on TUNE Sharpe (ev_det>=80%): entry={best["entry"]}  bot_thresh={best["bot"]}')
    print(f'  TUNE  : CAGR={best["bt"]["cagr"]*100:.1f}% Sharpe={best["bt"]["sharpe"]:.2f} MaxDD={best["bt"]["mdd"]*100:.1f}% lead={best["tune"]["lead"]}d ev_det={best["tune"]["ev_det"]:.0%}')
    print(f'  BLIND : CAGR={best["bb"]["cagr"]*100:.1f}% Sharpe={best["bb"]["sharpe"]:.2f} MaxDD={best["bb"]["mdd"]*100:.1f}% lead={best["test"]["lead"]}d ev_det={best["test"]["ev_det"]:.0%}')

with open('data/v5_1_pareto.json','w') as f:
    json.dump([{'entry':r['entry'],'bot':r['bot'],
                'tune_lead':r['tune']['lead'],'tune_evdet':r['tune']['ev_det'],
                'tune_cagr':r['bt']['cagr'],'tune_sharpe':r['bt']['sharpe'],'tune_mdd':r['bt']['mdd'],
                'blind_lead':r['test']['lead'],'blind_evdet':r['test']['ev_det'],
                'blind_cagr':r['bb']['cagr'],'blind_sharpe':r['bb']['sharpe'],'blind_mdd':r['bb']['mdd']}
               for r in results], f, indent=2, default=float)
print('\nSaved data/v5_1_pareto.json')
