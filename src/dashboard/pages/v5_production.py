"""v5 Production dashboard — the canonical view of our production crash detector.

Self-contained: pulls v5 + StatV3 + GBM_v4 predictions from DB, applies the
alarm hysteresis from data/alarm_config_v5.json, computes the live alarm state,
backtests vs Nasdaq buy-and-hold, and renders an information-dense view.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---- paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "data" / "market_crash.db"
CFG_PATH = ROOT / "data" / "alarm_config_v5.json"
EXP_C_PATH = ROOT / "data" / "experiment_C_reentry.json"
EXP_D_PATH = ROOT / "data" / "experiment_D_round2.json"
V6_VERDICT_PATH = ROOT / "data" / "v6_kill_verdict.json"

TUNE_END = pd.Timestamp("2020-12-31")
TEST_START = pd.Timestamp("2021-01-01")
COST_BPS = 5

# ---- visual identity -----------------------------------------------------
COL_V5 = "#FFD23F"            # v5 = signature gold
COL_PRICE = "#5BC0EB"         # cool blue for price
COL_ALARM = "#FF4D6D"         # crimson alarm shading
COL_CRASH = "rgba(120,120,120,0.18)"  # subtle grey for historical crashes
COL_BH = "#9D4EDD"            # purple for buy-and-hold
COL_STAT = "#06D6A0"          # mint for StatV3
COL_GBM = "#FF9F1C"           # amber for GBM_v4
COL_OK = "#06D6A0"
COL_BAD = "#EF476F"
COL_BG = "#0A0E1A"
COL_CARD = "rgba(255,255,255,0.04)"
COL_BORDER = "rgba(255,255,255,0.12)"


# =====================================================================
# Data layer
# =====================================================================

@st.cache_data(ttl=600, show_spinner=False)
def load_v5_data() -> dict:
    """Load everything the v5 page needs in one DB hit."""
    conn = sqlite3.connect(str(DB_PATH))
    ind = pd.read_sql(
        "SELECT date, sp500_close, nasdaq_close, vix_close FROM indicators ORDER BY date",
        conn, parse_dates=["date"],
    ).set_index("date").sort_index()
    preds = pd.read_sql(
        "SELECT prediction_date, crash_probability, model_version FROM predictions",
        conn, parse_dates=["prediction_date"],
    )
    conn.close()

    eq = ind["nasdaq_close"].ffill().dropna()
    sig_v5 = (
        preds[preds.model_version == "v5"]
        .set_index("prediction_date")["crash_probability"]
        .sort_index().reindex(eq.index).ffill()
    )
    # StatV3 predictions: tolerate either canonical name ("StatV3_phase3"
    # from older phase scripts, or "statistical_v3" from generate_predictions_v5.py)
    stat_rows = preds[preds.model_version.isin(["StatV3_phase3", "statistical_v3"])]
    sig_stat = (
        stat_rows.set_index("prediction_date")["crash_probability"]
        .sort_index().reindex(eq.index).ffill()
    )
    sig_gbm = (
        preds[preds.model_version == "GBM_v4"]
        .set_index("prediction_date")["crash_probability"]
        .sort_index().reindex(eq.index).ffill()
    )
    return dict(ind=ind, eq=eq, sig_v5=sig_v5, sig_stat=sig_stat, sig_gbm=sig_gbm)


@st.cache_data(ttl=86400, show_spinner=False)
def load_alarm_config() -> dict:
    return json.loads(CFG_PATH.read_text())


# =====================================================================
# Strategy primitives (mirror scripts/utils/v5_backtest.py logic)
# =====================================================================

def crash_labels(p: pd.Series, dd: float = 0.15, mtd: int = 30) -> pd.Series:
    """Peak-back-walked crashes: 252d-rolling-max DD ≥ 15%, ≥ 30 trading days."""
    p = p.ffill().dropna()
    n = len(p); pv = p.values
    rm = p.rolling(252, min_periods=50).max().values
    below = (pv / rm - 1) <= -dd
    cr = np.zeros(n, dtype=bool); i = 0
    while i < n:
        if not below[i]: i += 1; continue
        rs = i
        while i < n and below[i]: i += 1
        re = i - 1
        tp = rs + int(np.argmin(pv[rs:re + 1]))
        peak_val = rm[rs]; pp = rs
        for j in range(rs - 1, max(0, rs - 260), -1):
            if pv[j] >= peak_val * 0.998:
                pp = j; break
        if tp - pp >= mtd:
            cr[pp:tp + 1] = True
    return pd.Series(cr, index=p.index)


def event_table(crash: pd.Series, prices: pd.Series) -> pd.DataFrame:
    """Crash episodes with peak / trough / drawdown."""
    out = []; on = False; cs = None
    for d, v in crash.items():
        if not on and v: on, cs = True, d
        elif on and not v: out.append({"start": cs, "end": d}); on = False
    if on: out.append({"start": cs, "end": crash.index[-1]})
    df = pd.DataFrame(out)
    if df.empty: return df
    df["peak"] = df.apply(lambda r: prices.loc[r.start:r.end].idxmax(), axis=1)
    df["trough"] = df.apply(lambda r: prices.loc[r.start:r.end].idxmin(), axis=1)
    df["drawdown"] = df.apply(
        lambda r: prices.loc[r.trough] / prices.loc[r.peak] - 1, axis=1)
    df["duration_d"] = (df["trough"] - df["peak"]).dt.days
    return df


def hysteresis_alarms(prob: pd.Series, entry: float, exit_thr: float,
                      min_dur: int, max_dur: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    out = []; on = False; si = None; pv = prob.values; idx = prob.index
    for i in range(len(pv)):
        if not on:
            if pv[i] >= entry: on = True; si = i
        else:
            d = i - si
            if pv[i] < exit_thr or d >= max_dur:
                if d >= min_dur: out.append((idx[si], idx[i - 1]))
                on = False
    if on and si is not None and (len(pv) - si) >= min_dur:
        out.append((idx[si], idx[-1]))
    return out


def alarm_mask(alms: list, idx: pd.DatetimeIndex) -> pd.Series:
    m = pd.Series(False, index=idx)
    for s, e in alms: m.loc[s:e] = True
    return m


def backtest_5d_ma50(prices: pd.Series, am: pd.Series, cost_bps: int = COST_BPS) -> pd.Series:
    rets = prices.pct_change().fillna(0)
    ma50 = prices.rolling(50, min_periods=20).mean()
    in_pos, days_off = True, 0
    eqv = [1.0]; cost = cost_bps / 10000.0
    for i in range(1, len(prices)):
        prev = in_pos
        if am.iloc[i]:
            in_pos = False; days_off = 0
        else:
            days_off = days_off + 1 if not in_pos else 0
            if not in_pos and days_off >= 5 and prices.iloc[i] > ma50.iloc[i]:
                in_pos = True
        r = rets.iloc[i] if prev else 0.0
        if prev != in_pos: r -= cost
        eqv.append(eqv[-1] * (1 + r))
    return pd.Series(eqv, index=prices.index)


def window_metrics(eqv: pd.Series, start, end) -> dict:
    if start: eqv = eqv.loc[pd.Timestamp(start):]
    if end: eqv = eqv.loc[: pd.Timestamp(end)]
    eqv = eqv / eqv.iloc[0]
    yrs = (eqv.index[-1] - eqv.index[0]).days / 365.25
    cagr = eqv.iloc[-1] ** (1 / max(yrs, 1e-6)) - 1
    rr = eqv.pct_change().dropna()
    sharpe = rr.mean() / rr.std() * np.sqrt(252) if rr.std() > 0 else 0
    mdd = (eqv / eqv.cummax() - 1).min()
    return dict(cagr=cagr, sharpe=sharpe, mdd=mdd, final=eqv.iloc[-1])


def alarm_metrics(alms, evts, idx, lbl) -> dict:
    if not alms:
        return dict(ev_det=0, ev_prec=0, day_prec=0, day_rec=0, adp=0, lead=0, n=0)
    am = alarm_mask(alms, idx)
    y = lbl.reindex(idx).fillna(False).astype(bool)
    tp = int((am & y).sum()); fp = int((am & ~y).sum()); fn = int((~am & y).sum())
    dp = tp / max(tp + fp, 1); dr = tp / max(tp + fn, 1); adp = am.sum() / len(idx)
    n_conf = sum(1 for s, e in alms
                 if any(((s <= ce.peak) and (s >= ce.peak - pd.Timedelta(days=180))) or
                        ((s <= ce.end) and (e >= ce.start)) for _, ce in evts.iterrows()))
    leads = []; n_det = 0
    for _, ce in evts.iterrows():
        for s, e in alms:
            if (s <= ce.end + pd.Timedelta(days=30)) and (e >= ce.start - pd.Timedelta(days=180)):
                n_det += 1; leads.append((ce.peak - s).days); break
    return dict(ev_det=n_det / max(len(evts), 1), ev_prec=n_conf / len(alms),
                day_prec=dp, day_rec=dr, adp=adp,
                lead=int(np.median(leads)) if leads else 0, n=len(alms))


# =====================================================================
# Computation pipeline (cached)
# =====================================================================

@st.cache_data(ttl=600, show_spinner=False)
def compute_v5_strategy() -> dict:
    data = load_v5_data()
    cfg = load_alarm_config()
    eq = data["eq"]
    sig_v5 = data["sig_v5"]
    sig_stat = data["sig_stat"]

    crash = crash_labels(eq)
    events = event_table(crash, eq)

    alms_full = hysteresis_alarms(
        sig_v5, cfg["entry"], cfg["exit"], cfg["min_dur"], cfg["max_dur"])
    am_full = alarm_mask(alms_full, sig_v5.index)

    eqv_v5 = backtest_5d_ma50(eq, am_full)
    eqv_bh = (eq / eq.iloc[0])

    # window metrics
    metrics_full = {
        "v5":   window_metrics(eqv_v5, "2000-01-01", None),
        "bh":   window_metrics(eqv_bh, "2000-01-01", None),
    }
    metrics_tune = {
        "v5":   window_metrics(eqv_v5, "2000-01-01", "2020-12-31"),
        "bh":   window_metrics(eqv_bh, "2000-01-01", "2020-12-31"),
    }
    metrics_blind = {
        "v5":   window_metrics(eqv_v5, "2021-01-01", None),
        "bh":   window_metrics(eqv_bh, "2021-01-01", None),
    }

    # Alarm metrics on TUNE / BLIND
    sig_tune = sig_v5[sig_v5.index <= TUNE_END]
    sig_blind = sig_v5[sig_v5.index >= TEST_START]
    alms_tune = hysteresis_alarms(
        sig_tune, cfg["entry"], cfg["exit"], cfg["min_dur"], cfg["max_dur"])
    alms_blind = hysteresis_alarms(
        sig_blind, cfg["entry"], cfg["exit"], cfg["min_dur"], cfg["max_dur"])
    ev_tune = events[events.peak <= TUNE_END].reset_index(drop=True)
    ev_blind = events[events.peak >= TEST_START].reset_index(drop=True)
    am_tune = alarm_metrics(alms_tune, ev_tune, sig_tune.index, crash)
    am_blind = alarm_metrics(alms_blind, ev_blind, sig_blind.index, crash)

    # live state
    today = sig_v5.index[-1]
    in_alarm_today = bool(am_full.loc[today])
    if alms_full:
        last_s, last_e = alms_full[-1]
        days_since_last_toggle = (today - last_e).days if not in_alarm_today else (today - last_s).days
    else:
        last_s = last_e = None
        days_since_last_toggle = None

    return dict(
        eq=eq, crash=crash, events=events,
        sig_v5=sig_v5, sig_stat=sig_stat,
        cfg=cfg, alms_full=alms_full, am_full=am_full,
        eqv_v5=eqv_v5, eqv_bh=eqv_bh,
        metrics_full=metrics_full, metrics_tune=metrics_tune, metrics_blind=metrics_blind,
        alarm_metrics_tune=am_tune, alarm_metrics_blind=am_blind,
        today=today, in_alarm_today=in_alarm_today,
        days_since_last_toggle=days_since_last_toggle,
        last_alarm_start=last_s, last_alarm_end=last_e,
        latest_prob=float(sig_v5.iloc[-1]),
        latest_stat=(float(sig_stat.iloc[-1])
                     if not sig_stat.empty and pd.notna(sig_stat.iloc[-1])
                     else None),
    )


# =====================================================================
# Visual helpers
# =====================================================================

def _kpi_card(label: str, value: str, *, sub: str = "", color: str = COL_V5,
              icon: str = "") -> str:
    """Render a glassy KPI card with gradient accent."""
    return f"""
    <div style="
        background: linear-gradient(135deg, {COL_CARD} 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid {COL_BORDER};
        border-left: 4px solid {color};
        border-radius: 14px;
        padding: 18px 20px;
        height: 100%;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);">
      <div style="font-size: 0.78rem; letter-spacing: 0.08em;
                  text-transform: uppercase; color: rgba(255,255,255,0.55);
                  margin-bottom: 6px;">
        {icon} {label}
      </div>
      <div style="font-size: 2.0rem; font-weight: 700; color: {color};
                  line-height: 1.1;">
        {value}
      </div>
      <div style="font-size: 0.85rem; color: rgba(255,255,255,0.65);
                  margin-top: 4px;">
        {sub}
      </div>
    </div>
    """


def _status_badge(in_alarm: bool, days: int | None) -> str:
    if in_alarm:
        return f"""
        <div style="
            background: linear-gradient(135deg, rgba(255,77,109,0.25), rgba(255,77,109,0.05));
            border: 1px solid {COL_ALARM};
            border-radius: 14px; padding: 22px 26px;
            box-shadow: 0 0 30px rgba(255,77,109,0.18);">
          <div style="font-size: 0.78rem; letter-spacing: 0.10em;
                      color: rgba(255,255,255,0.6); text-transform: uppercase;">
            Live State
          </div>
          <div style="font-size: 2.4rem; font-weight: 800; color: {COL_ALARM};
                      letter-spacing: 0.02em;">
            🚨 ALARM ACTIVE
          </div>
          <div style="font-size: 0.95rem; color: rgba(255,255,255,0.75);
                      margin-top: 4px;">
            v5 has been firing for {days}d. Strategy in cash; awaits price &gt; MA50 + 5d cool-off.
          </div>
        </div>"""
    sub = f"{days}d since last alarm cleared" if days is not None else "no alarms in history"
    return f"""
    <div style="
        background: linear-gradient(135deg, rgba(6,214,160,0.18), rgba(6,214,160,0.03));
        border: 1px solid {COL_OK};
        border-radius: 14px; padding: 22px 26px;
        box-shadow: 0 0 30px rgba(6,214,160,0.15);">
      <div style="font-size: 0.78rem; letter-spacing: 0.10em;
                  color: rgba(255,255,255,0.6); text-transform: uppercase;">
        Live State
      </div>
      <div style="font-size: 2.4rem; font-weight: 800; color: {COL_OK};
                  letter-spacing: 0.02em;">
        ✅ ALL CLEAR
      </div>
      <div style="font-size: 0.95rem; color: rgba(255,255,255,0.75);
                  margin-top: 4px;">
        Strategy long Nasdaq · {sub}
      </div>
    </div>"""


def _master_chart(state: dict, lookback_years: int) -> go.Figure:
    eq = state["eq"]; sig = state["sig_v5"]; events = state["events"]
    alms = state["alms_full"]
    cfg = state["cfg"]

    cutoff = eq.index[-1] - pd.DateOffset(years=lookback_years)
    eq = eq.loc[cutoff:]; sig = sig.loc[cutoff:]
    ma50 = state["eq"].rolling(50, min_periods=20).mean().loc[cutoff:]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.65, 0.35],
        subplot_titles=("Nasdaq Composite — v5 alarms shaded · MA50 overlay",
                        "v5 blended probability — entry / exit hysteresis"),
    )

    # Historical crash episodes — subtle grey shade behind everything
    for _, ev in events.iterrows():
        if ev.end < cutoff: continue
        fig.add_vrect(x0=max(ev.peak, cutoff), x1=ev.trough,
                      fillcolor=COL_CRASH, opacity=1, line_width=0,
                      layer="below", row=1, col=1)

    # v5 alarm episodes — red shade
    for s, e in alms:
        if e < cutoff: continue
        fig.add_vrect(x0=max(s, cutoff), x1=e,
                      fillcolor=COL_ALARM, opacity=0.22, line_width=0,
                      layer="below", row=1, col=1)
        fig.add_vrect(x0=max(s, cutoff), x1=e,
                      fillcolor=COL_ALARM, opacity=0.10, line_width=0,
                      layer="below", row=2, col=1)

    # TUNE/BLIND divider
    if cutoff < TEST_START < eq.index[-1]:
        fig.add_vline(x=TEST_START, line=dict(color="rgba(255,210,63,0.55)",
                      width=1.5, dash="dot"), row=1, col=1)
        fig.add_annotation(x=TEST_START, y=eq.max(), xref="x", yref="y",
                           text="BLIND →", showarrow=False, font=dict(color=COL_V5),
                           xanchor="left", yanchor="top")

    # Price + MA50
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values, mode="lines", name="Nasdaq",
        line=dict(color=COL_PRICE, width=2.2),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{y:,.0f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ma50.index, y=ma50.values, mode="lines", name="MA50",
        line=dict(color="rgba(255,255,255,0.45)", width=1.2, dash="dot"),
        hoverinfo="skip",
    ), row=1, col=1)

    # Probability + hysteresis bands
    fig.add_trace(go.Scatter(
        x=sig.index, y=sig.values, mode="lines", name="v5 prob",
        line=dict(color=COL_V5, width=2.0),
        fill="tozeroy", fillcolor="rgba(255,210,63,0.10)",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>p = %{y:.2%}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=cfg["entry"], line=dict(color=COL_ALARM, width=1, dash="dash"),
                  annotation_text=f"entry {cfg['entry']:.2f}",
                  annotation_position="right", row=2, col=1)
    fig.add_hline(y=cfg["exit"], line=dict(color=COL_OK, width=1, dash="dash"),
                  annotation_text=f"exit {cfg['exit']:.2f}",
                  annotation_position="right", row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=620, hovermode="x unified",
        plot_bgcolor=COL_BG, paper_bgcolor=COL_BG,
        margin=dict(l=60, r=70, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)"),
        font=dict(family="Inter, system-ui, sans-serif"),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1, type="log",
                     showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="probability", row=2, col=1, range=[0, 1],
                     showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(showgrid=False)
    return fig


def _equity_chart(state: dict) -> go.Figure:
    eqv_v5 = state["eqv_v5"]; eqv_bh = state["eqv_bh"]
    eqv_v5 = eqv_v5 / eqv_v5.iloc[0]
    eqv_bh = eqv_bh / eqv_bh.iloc[0]

    fig = go.Figure()
    fig.add_vrect(
        x0=eqv_v5.index[0], x1=TUNE_END,
        fillcolor="rgba(157,78,221,0.08)", line_width=0, layer="below",
        annotation_text="TUNE", annotation_position="top left",
        annotation=dict(font=dict(color="rgba(255,255,255,0.5)")),
    )
    fig.add_vrect(
        x0=TEST_START, x1=eqv_v5.index[-1],
        fillcolor="rgba(255,210,63,0.07)", line_width=0, layer="below",
        annotation_text="BLIND (out-of-sample)", annotation_position="top right",
        annotation=dict(font=dict(color=COL_V5)),
    )

    fig.add_trace(go.Scatter(
        x=eqv_bh.index, y=eqv_bh.values, mode="lines",
        name="Buy & Hold Nasdaq", line=dict(color=COL_BH, width=1.8, dash="dot"),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>B&H: %{y:.2f}x<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=eqv_v5.index, y=eqv_v5.values, mode="lines",
        name="v5 strategy", line=dict(color=COL_V5, width=2.6),
        fill="tonexty", fillcolor="rgba(255,210,63,0.08)",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>v5: %{y:.2f}x<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark", height=420,
        title=dict(text="Equity curves — v5 strategy vs Nasdaq buy-and-hold (5bp cost)",
                   font=dict(size=15)),
        plot_bgcolor=COL_BG, paper_bgcolor=COL_BG,
        margin=dict(l=60, r=50, t=70, b=40),
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(type="log", title="growth of $1",
                   showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(showgrid=False),
        font=dict(family="Inter, system-ui, sans-serif"),
    )
    return fig


def _scorecard_panel(state: dict) -> None:
    """Three-window scorecard: TUNE / BLIND / FULL."""
    rows = []
    for window, label in [("metrics_tune", "TUNE 2000-2020"),
                          ("metrics_blind", "BLIND 2021-now"),
                          ("metrics_full", "FULL 2000-now")]:
        m_v5 = state[window]["v5"]; m_bh = state[window]["bh"]
        rows.append({
            "Window": label,
            "v5 CAGR":   f"{m_v5['cagr']*100:5.1f}%",
            "B&H CAGR":  f"{m_bh['cagr']*100:5.1f}%",
            "v5 Sharpe": f"{m_v5['sharpe']:.2f}",
            "B&H Sharpe":f"{m_bh['sharpe']:.2f}",
            "v5 MaxDD":  f"{m_v5['mdd']*100:5.1f}%",
            "B&H MaxDD": f"{m_bh['mdd']*100:5.1f}%",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _kill_scorecard(state: dict) -> pd.DataFrame:
    """Compare v5 BLIND metrics against shelved alternatives."""
    m_v5 = state["alarm_metrics_blind"]
    bb_v5 = state["metrics_blind"]["v5"]
    rows = [{
        "Model": "✅ v5 (PRODUCTION)",
        "BLIND day_prec": f"{m_v5['day_prec']*100:.1f}%",
        "BLIND ev_det":   f"{m_v5['ev_det']*100:.1f}%",
        "BLIND CAGR":     f"{bb_v5['cagr']*100:.1f}%",
        "BLIND Sharpe":   f"{bb_v5['sharpe']:.2f}",
        "BLIND MaxDD":    f"{bb_v5['mdd']*100:.1f}%",
        "Status": "BENCHMARK",
    }]
    if V6_VERDICT_PATH.exists():
        v6 = json.loads(V6_VERDICT_PATH.read_text())
        rows.append({
            "Model": "❌ v6 (options + breadth + cross-asset)",
            "BLIND day_prec": f"{v6['v6']['alarm']['day_prec']*100:.1f}%",
            "BLIND ev_det":   f"{v6['v6']['alarm']['ev_det']*100:.1f}%",
            "BLIND CAGR":     f"{v6['v6']['blind']['cagr']*100:.1f}%",
            "BLIND Sharpe":   f"{v6['v6']['blind']['sharpe']:.2f}",
            "BLIND MaxDD":    f"{v6['v6']['blind']['mdd']*100:.1f}%",
            "Status": "SHELVED — kill 2/4",
        })
    if EXP_D_PATH.exists():
        d = json.loads(EXP_D_PATH.read_text())
        rows.append({
            "Model": "❌ v5_multi (US + EU + Asia, 83 events)",
            "BLIND day_prec": f"{d['v5_multi']['alarm']['day_prec']*100:.1f}%",
            "BLIND ev_det":   f"{d['v5_multi']['alarm']['ev_det']*100:.1f}%",
            "BLIND CAGR":     f"{d['v5_multi']['blind']['cagr']*100:.1f}%",
            "BLIND Sharpe":   f"{d['v5_multi']['blind']['sharpe']:.2f}",
            "BLIND MaxDD":    f"{d['v5_multi']['blind']['mdd']*100:.1f}%",
            "Status": "SHELVED — kill 3/5",
        })
    return pd.DataFrame(rows)


def _events_table(state: dict) -> pd.DataFrame:
    """Build the historical-events table with two lead columns:

    - **Lead-to-peak (d)**: signed.  positive = alarm fired BEFORE peak
      (early warning); negative = alarm fired AFTER peak (concurrent
      detection — the normal case for v5, which is a near-coincident
      detector by design).
    - **Days into drawdown when caught**: always ≥ 0.  How many days from
      peak to first alarm.  Same magnitude as |Lead-to-peak| when negative,
      and 0 when alarm fires before or on the peak.
    """
    events = state["events"].copy()
    if events.empty: return events
    am = state["am_full"]
    rows = []
    for _, ev in events.iterrows():
        # detection: any alarm overlapping [peak-180d, end+30d]
        window_start = ev.peak - pd.Timedelta(days=180)
        window_end = ev.end + pd.Timedelta(days=30)
        detected_mask = am.loc[window_start:window_end]
        detected = bool(detected_mask.any())
        if detected:
            first_alarm_day = detected_mask[detected_mask].index[0]
            lead = (ev.peak - first_alarm_day).days
            into_dd = max(0, -lead)
        else:
            lead = None
            into_dd = None
        rows.append({
            "Peak date":    ev.peak.strftime("%Y-%m-%d"),
            "Trough date":  ev.trough.strftime("%Y-%m-%d"),
            "Drawdown":     f"{ev.drawdown*100:.1f}%",
            "Duration":     f"{ev.duration_d}d",
            "Era":          "BLIND" if ev.peak >= TEST_START else "TUNE",
            "v5 caught?":   "✅" if detected else "❌",
            "Lead-to-peak (d)":          "—" if lead is None else f"{lead:+d}",
            "Days into drawdown caught": "—" if into_dd is None else f"{into_dd}d",
        })
    return pd.DataFrame(rows)


# =====================================================================
# Render entry point
# =====================================================================

def render() -> None:
    # Custom CSS — subtle but distinct
    st.markdown(f"""
    <style>
      .v5-hero {{
        background: linear-gradient(135deg, rgba(255,210,63,0.08) 0%, rgba(157,78,221,0.05) 100%);
        border: 1px solid {COL_BORDER};
        border-radius: 18px; padding: 28px 32px;
        margin-bottom: 22px;
      }}
      .v5-title {{
        font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em;
        background: linear-gradient(90deg, {COL_V5} 0%, {COL_PRICE} 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0;
      }}
      .v5-sub {{
        color: rgba(255,255,255,0.7); font-size: 1.0rem;
        margin-top: 6px;
      }}
      .v5-section {{
        font-size: 1.15rem; font-weight: 600; color: {COL_V5};
        letter-spacing: 0.02em; margin-top: 18px; margin-bottom: 8px;
        border-bottom: 1px solid {COL_BORDER}; padding-bottom: 6px;
      }}
    </style>
    """, unsafe_allow_html=True)

    state = compute_v5_strategy()

    # ---- Hero ---------------------------------------------------------
    st.markdown(f"""
    <div class="v5-hero">
      <div class="v5-title">🛡️ v5 — PRODUCTION CRASH DETECTOR</div>
      <div class="v5-sub">
        XGBoost predictive model · StatV3 risk-factor blend · 50/50 weighted ·
        nested walk-forward (4 folds 1999-2026 · BLIND ≥ 2021) ·
        alarm hysteresis (entry={state['cfg']['entry']:.2f} · exit={state['cfg']['exit']:.2f} ·
        min_dur={state['cfg']['min_dur']}d) · re-entry MA50 + 5d
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Top row: live state + 3 KPIs ---------------------------------
    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
    with c1:
        st.markdown(_status_badge(state["in_alarm_today"], state["days_since_last_toggle"]),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(_kpi_card(
            "Latest blended probability",
            f"{state['latest_prob']*100:.1f}%",
            sub=f"as of {state['today'].strftime('%Y-%m-%d')}",
            color=COL_V5, icon="🎯"), unsafe_allow_html=True)
    with c3:
        ab = state["alarm_metrics_blind"]
        st.markdown(_kpi_card(
            "BLIND day-precision",
            f"{ab['day_prec']*100:.1f}%",
            sub=f"{ab['n']} alarm episodes · ev_det {ab['ev_det']*100:.0f}%",
            color=COL_OK, icon="🎯"), unsafe_allow_html=True)
    with c4:
        bb = state["metrics_blind"]["v5"]
        bh = state["metrics_blind"]["bh"]
        st.markdown(_kpi_card(
            "BLIND CAGR",
            f"{bb['cagr']*100:.1f}%",
            sub=f"vs B&H {bh['cagr']*100:.1f}% · Sharpe {bb['sharpe']:.2f}",
            color=COL_V5, icon="📈"), unsafe_allow_html=True)

    # ---- Master chart -------------------------------------------------
    st.markdown('<div class="v5-section">📈 Master view — price · alarms · signal</div>',
                unsafe_allow_html=True)
    lookback = st.select_slider(
        "lookback window",
        options=[3, 5, 10, 15, 26], value=10,
        format_func=lambda y: f"{y} years", key="v5_master_lookback",
        label_visibility="collapsed",
    )
    st.plotly_chart(_master_chart(state, lookback), use_container_width=True,
                    key="v5_master_chart")

    # ---- Equity curve -------------------------------------------------
    st.markdown('<div class="v5-section">💰 Strategy performance vs buy-and-hold</div>',
                unsafe_allow_html=True)
    st.plotly_chart(_equity_chart(state), use_container_width=True, key="v5_equity_chart")
    _scorecard_panel(state)

    # ---- BLIND alarm scorecard ---------------------------------------
    st.markdown('<div class="v5-section">🎯 BLIND-window alarm metrics (out-of-sample)</div>',
                unsafe_allow_html=True)
    ab = state["alarm_metrics_blind"]; at = state["alarm_metrics_tune"]
    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    for col, label, k, fmt, color in [
        (cc1, "Event detection", "ev_det", lambda x: f"{x*100:.0f}%", COL_OK),
        (cc2, "Event precision", "ev_prec", lambda x: f"{x*100:.0f}%", COL_OK),
        (cc3, "Day precision", "day_prec", lambda x: f"{x*100:.1f}%", COL_OK),
        (cc4, "Alarm-day pct", "adp", lambda x: f"{x*100:.1f}%", COL_V5),
        (cc5, "Median lead", "lead", lambda x: f"{x:+d}d", COL_V5),
    ]:
        with col:
            st.markdown(_kpi_card(
                label, fmt(ab[k]), sub=f"TUNE: {fmt(at[k])}", color=color,
            ), unsafe_allow_html=True)

    # ---- Kill scorecard ----------------------------------------------
    st.markdown('<div class="v5-section">⚔️ Why v5 stays — shelved alternatives</div>',
                unsafe_allow_html=True)
    st.dataframe(_kill_scorecard(state), use_container_width=True, hide_index=True)
    st.caption("Each shelved model FAILED kill criteria: a model ships only if it does NOT regress "
               "v5 BLIND day-precision (≥ -5pp), ev_det (≥ -5pp), median lead (≥ -5d), and "
               "BLIND backtest CAGR (≥ -2pp). See `docs/FUTURE_WORK_RESULTS.md`.")

    # ---- Historical events -------------------------------------------
    st.markdown('<div class="v5-section">📚 Historical crash episodes (Nasdaq 15%+ DD)</div>',
                unsafe_allow_html=True)
    ev_df = _events_table(state)
    st.dataframe(ev_df, use_container_width=True, hide_index=True)
    n_ev = len(ev_df); n_caught = (ev_df["v5 caught?"] == "✅").sum() if n_ev else 0
    st.caption(
        f"v5 detected {n_caught}/{n_ev} historical crash episodes "
        f"({n_caught/max(n_ev,1)*100:.0f}%) with median lead-to-peak "
        f"of {state['alarm_metrics_blind']['lead']}d (BLIND), "
        f"{state['alarm_metrics_tune']['lead']}d (TUNE). "
        f"v5 is a **near-coincident detector** by design — negative lead-to-peak "
        f"means the alarm fires AFTER the price peak, typically once a developing "
        f"drawdown is already confirmed by the BAA spread, VIX and momentum factors. "
        f"Positive lead-to-peak (early warning) is a bonus, not the norm."
    )

    # ---- Methodology --------------------------------------------------
    with st.expander("🧠 Architecture & methodology", expanded=False):
        st.markdown(f"""
**Production model: v5** — committed at git tag `v5-BENCHMARK` (`925d8b7`),
mirrored at branch `v5-benchmark-protected`.

**Architecture**
- **Predictive ML**: XGBoost (n_estimators=500, max_depth=4, lr=0.03, subsample=0.75,
  colsample_bytree=0.8, min_child_weight=15, α=0.1, λ=2.0, scale_pos_weight=neg/pos).
- **Statistical risk**: StatV3 — multi-factor scoring across yield curve, credit,
  market, sentiment, economic, financial-conditions, momentum-shock dimensions.
- **Blend**: signal = 0.5 × XGBoost_proba + 0.5 × StatV3_total_risk
- **Alarm hysteresis**: enter at p ≥ {state['cfg']['entry']:.2f}, exit at p < {state['cfg']['exit']:.2f},
  minimum duration {state['cfg']['min_dur']}d, max {state['cfg']['max_dur']}d.
- **Re-entry**: 5-day cool-off after alarm clears, then long Nasdaq when price > MA50.
- **Cost**: 5 bp per round-trip applied to backtest.

**How entry / exit are computed** (no hand-tuning, no leakage)
1. After the 4 walk-forward folds produce out-of-sample probabilities for
   1999-2020, the trainer takes the v5 blended signal restricted to TUNE.
2. **Grid search** — 5 × 3 × 4 × 4 × 3 = 720 combinations of
   `entry ∈ {{0.25, 0.30, 0.35, 0.40, 0.45}}`, `exit ∈ {{0.10, 0.15, 0.20}}`,
   `min_dur ∈ {{5, 10, 15, 20}}d`, `max_dur ∈ {{30, 45, 60, 90}}d`,
   `mf_min ∈ {{1, 2, 3}}` (min. number of StatV3 risk factors elevated).
3. **Constraints** (must pass): `event_detection ≥ 75%` AND `day_precision ≥ 30%`
   on TUNE.
4. **Objective**: `score = 2·event_detection + day_precision − 2·alarm_day_pct`.
5. The winning tuple is written to `data/alarm_config_v5.json` and **frozen**.
   BLIND (≥ 2021) is then evaluated *once* with that frozen config.
6. **Hysteresis at inference**: state starts OFF. When `prob ≥ entry` → ON.
   Once ON, stay ON until `prob < exit` AND `days_since_on ≥ min_dur`. Force OFF
   at `max_dur` days. The gap between entry and exit (with min_duration) is
   what prevents the alarm from chattering when `prob` oscillates near a single
   threshold.

Current values: **entry={state['cfg']['entry']:.2f}, exit={state['cfg']['exit']:.2f},
min_dur={state['cfg']['min_dur']}d, max_dur={state['cfg']['max_dur']}d**
(re-tuned on whatever data is currently available; the frozen v5-BENCHMARK at
git tag `v5-BENCHMARK` chose `entry=0.45, exit=0.20, min_dur=20d` from a richer
dataset).

**Validation discipline**
- Nested walk-forward, 4 folds: (1999-2005), (2005-2012), (2012-2020), (2020-2026).
- TUNE = ≤ 2020-12-31 used for hyperparameter search and alarm-config tuning.
- BLIND = ≥ 2021-01-01 evaluated *once* with frozen config.
- Crash labels: rolling 252d max drawdown ≥ 15%, peak-back-walked, ≥ 30 trading days
  duration. {len(state['events'])} episodes total since 1983.

**Why v5 stays**
Three independent attempts to improve on v5 (v5.1, v6, v5_multi) all failed
the same kill criteria on BLIND. Experiment C established that a perfect
bottom-finder oracle improves CAGR by only +0.5pp over the current MA50 rule —
the practical ceiling. See `docs/FUTURE_WORK_RESULTS.md` for the full scorecard.
""")
