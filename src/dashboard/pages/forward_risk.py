"""Forward Risk dashboard tab — partial-ship: 1-month horizon ONLY.

Scope (per BLIND audit on 2026-04-29):
  Only h=21d (1 trading month) ret + maxdd cells passed the strict
  [85%, 95%] BLIND-coverage kill criterion after split-conformal CQR
  recalibration. Horizons 63/126/252 days were SHELVED — over-coverage on
  BLIND would mean shipping vacuously wide intervals (e.g. +/-113%
  log-return for h=252).

Honesty surfaces ON THIS PAGE (cannot be hidden):
  * Pre-conformal vs post-conformal coverage on BLIND (the actual numbers).
  * What was shelved and why.
  * v5 vs Forward-Risk side-by-side.
  * Median CI90 width — sharpness IS reported.

Source artifact: data/processed/forward_risk_predictions_conformal.parquet
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sqlite3

ROOT = Path(__file__).resolve().parents[3]
PRED_PATH = ROOT / "data" / "processed" / "forward_risk_predictions_conformal.parquet"
SUMMARY_PATH = ROOT / "data" / "processed" / "forward_risk_conformal_summary.csv"
DB_PATH = ROOT / "data" / "market_crash.db"

BLIND_START = pd.Timestamp("2021-01-01")
H_SHIP = 21  # only horizon that passes strict [85%, 95%] BLIND coverage.

# Match the dashboard's signature dark theme (same as v5_production tab).
COL_BG       = "#0A0E1A"
COL_PRICE    = "#5BC0EB"          # cool blue
COL_GBM      = "#FF9F1C"          # amber (90% band)
COL_BAND_50  = "rgba(91,192,235,0.30)"
COL_BAND_90  = "rgba(255,159,28,0.18)"
COL_REAL     = "#FFD23F"          # signature gold for realized
COL_TEMPLATE = "plotly_dark"


@st.cache_data(show_spinner=False)
def _load_predictions() -> pd.DataFrame:
    df = pd.read_parquet(PRED_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df[df["horizon"] == H_SHIP].copy()


@st.cache_data(show_spinner=False)
def _load_summary() -> pd.DataFrame:
    return pd.read_csv(SUMMARY_PATH)


@st.cache_data(show_spinner=False)
def _load_nasdaq() -> pd.DataFrame:
    with sqlite3.connect(str(DB_PATH)) as conn:
        df = pd.read_sql(
            "SELECT date, nasdaq_close FROM indicators "
            "WHERE nasdaq_close IS NOT NULL ORDER BY date",
            conn, parse_dates=["date"],
        )
    df["nasdaq_close"] = pd.to_numeric(df["nasdaq_close"], errors="coerce")
    return df.dropna().set_index("date")


def _latest_forecast(preds: pd.DataFrame, target: str) -> pd.Series:
    g = preds[preds["target"] == target].sort_values("date")
    return g.iloc[-1] if len(g) else pd.Series(dtype=float)


def render() -> None:
    st.markdown("## Forward Risk — 1-month NASDAQ forecast")
    st.caption(
        "Partial-ship. Only the 21-trading-day (1-month) horizon passed the "
        "strict [85%, 95%] BLIND-coverage kill criterion after split-conformal "
        "recalibration. 3-month / 6-month / 12-month horizons were SHELVED — "
        "see methodology below."
    )

    if not PRED_PATH.exists():
        st.error(f"Prediction artifact missing: {PRED_PATH}. Run "
                 "`scripts/forward_risk/conformal_recalibrate.py`.")
        return

    preds = _load_predictions()
    summary = _load_summary()
    nasdaq = _load_nasdaq()

    # ---------------- header KPIs (latest forecast) ----------------
    latest_ret = _latest_forecast(preds, "ret")
    latest_dd  = _latest_forecast(preds, "maxdd")
    asof = max(latest_ret["date"], latest_dd["date"])
    st.markdown(f"**As of {asof.date().isoformat()}** — forecast for the next 21 trading days:")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Median return (1m)",
            f"{(np.exp(latest_ret['q50']) - 1) * 100:+.2f}%",
            help="Conditional median forecast. Log-return converted to simple return.",
        )
    with c2:
        # 90% CI in simple-return space
        lo = (np.exp(latest_ret["q05_conf"]) - 1) * 100
        hi = (np.exp(latest_ret["q95_conf"]) - 1) * 100
        st.metric(
            "90% CI on return",
            f"{lo:+.1f}% .. {hi:+.1f}%",
            help="Post-conformal calibrated 90% interval (simple return).",
        )
    with c3:
        st.metric(
            "Median max drawdown (1m)",
            f"{(np.exp(latest_dd['q50']) - 1) * 100:.2f}%",
            help="Median worst-case peak-to-trough log return within the next 21 days.",
        )
    with c4:
        # Tail probability: P(maxdd worse than -10%). Use post-conformal
        # quantiles consistently (mixing pre/post would imply a non-monotone CDF).
        # The median q50 is not conformal-shifted by construction (alpha=0.5
        # applies to interval endpoints, not the point estimate); we treat it
        # as the q=0.50 anchor.
        cutoff = np.log(1 - 0.10)  # log(0.90), -10% simple-return threshold
        qs = np.array([0.05, 0.25, 0.50, 0.75, 0.95])
        xs = np.array([latest_dd["q05_conf"], latest_dd["q25_conf"],
                       latest_dd["q50"],      latest_dd["q75_conf"],
                       latest_dd["q95_conf"]])
        order = np.argsort(xs); xs, ys = xs[order], qs[order]
        if cutoff <= xs[0]:
            p = 0.0
        elif cutoff >= xs[-1]:
            p = 1.0
        else:
            p = float(np.interp(cutoff, xs, ys))
        st.metric(
            "P(maxdd worse than -10%)",
            f"{p * 100:.0f}%",
            help="Probability the worst peak-to-trough log return exceeds -10% over the next 21 days. Computed via linear interpolation of the post-conformal CDF through (q05, q25, q50, q75, q95).",
        )

    # ---------------- Honesty band on calibration ----------------
    cell_ret = summary[(summary["horizon"] == H_SHIP) & (summary["target"] == "ret")].iloc[0]
    cell_dd  = summary[(summary["horizon"] == H_SHIP) & (summary["target"] == "maxdd")].iloc[0]
    st.markdown("### BLIND calibration audit (the truth, on the page)")
    st.caption("Measured ONCE on out-of-sample data >= 2021-01-01. n_blind shown per row.")
    audit = pd.DataFrame([
        {"target": "ret",   "n_blind": int(cell_ret["n_blind"]),
         "cov90 pre-conformal": f"{cell_ret['cov_pre_90']:.3f}",
         "cov90 post-conformal": f"{cell_ret['cov_post_90']:.3f}",
         "median CI90 width": f"{cell_ret['width_post_90']:.3f}",
         "kill criterion [0.85, 0.95]": "PASS" if 0.85 <= cell_ret["cov_post_90"] <= 0.95 else "FAIL"},
        {"target": "maxdd", "n_blind": int(cell_dd["n_blind"]),
         "cov90 pre-conformal": f"{cell_dd['cov_pre_90']:.3f}",
         "cov90 post-conformal": f"{cell_dd['cov_post_90']:.3f}",
         "median CI90 width": f"{cell_dd['width_post_90']:.3f}",
         "kill criterion [0.85, 0.95]": "PASS" if 0.85 <= cell_dd["cov_post_90"] <= 0.95 else "FAIL"},
    ])
    st.dataframe(audit, hide_index=True, use_container_width=True)

    # ---------------- Fan chart over BLIND period ----------------
    st.markdown("### BLIND fan vs realized (2021-01-01 onward)")
    target = st.radio("Target", ["ret", "maxdd"], horizontal=True, key="fr_target")
    g = preds[preds["target"] == target].sort_values("date")
    if g.empty:
        st.warning("No predictions to plot.")
        return

    fig = go.Figure()
    # 90% band (post-conformal)
    fig.add_trace(go.Scatter(
        x=g["date"], y=g["q95_conf"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=g["date"], y=g["q05_conf"], line=dict(width=0), fill="tonexty",
        fillcolor=COL_BAND_90, name="90% CI (post-conformal)",
        hovertemplate="q05: %{y:.4f}<extra></extra>"))
    # 50% band (post-conformal — same calibration source as the 90% band).
    fig.add_trace(go.Scatter(
        x=g["date"], y=g["q75_conf"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=g["date"], y=g["q25_conf"], line=dict(width=0), fill="tonexty",
        fillcolor=COL_BAND_50, name="50% CI (post-conformal)"))
    # Median
    fig.add_trace(go.Scatter(
        x=g["date"], y=g["q50"], line=dict(color=COL_PRICE, width=1.4), name="median"))
    # Actual (only where observed)
    obs = g.dropna(subset=["y_true"])
    fig.add_trace(go.Scatter(
        x=obs["date"], y=obs["y_true"], line=dict(color=COL_REAL, width=1.2),
        name="realized", mode="lines"))
    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=20, b=20),
        legend=dict(orientation="h", yanchor="top", y=1.08),
        xaxis_title=None,
        yaxis_title=("forward 21d log-return" if target == "ret"
                     else "forward 21d max log-drawdown"),
        template=COL_TEMPLATE,
        plot_bgcolor=COL_BG, paper_bgcolor=COL_BG,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- v5 vs Forward Risk side-by-side ----------------
    st.markdown("### Compare with v5 (current state)")
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            v5 = pd.read_sql(
                "SELECT date, ensemble_v5_proba FROM model_predictions "
                "WHERE ensemble_v5_proba IS NOT NULL "
                "ORDER BY date DESC LIMIT 1",
                conn, parse_dates=["date"],
            )
        if len(v5):
            v5_p = float(v5["ensemble_v5_proba"].iloc[0])
            v5_d = pd.to_datetime(v5["date"].iloc[0]).date().isoformat()
        else:
            v5_p, v5_d = float("nan"), "n/a"
    except Exception:
        v5_p, v5_d = float("nan"), "n/a"

    cA, cB = st.columns(2)
    with cA:
        st.markdown("**v5 (near-coincident detector)**")
        st.write(f"- as of: `{v5_d}`")
        st.write(f"- crash probability: `{v5_p:.3f}`" if not np.isnan(v5_p) else "- crash probability: `n/a`")
        st.caption("v5 answers: *is a crash happening RIGHT NOW?*")
    with cB:
        st.markdown("**Forward Risk (probabilistic, 1m ahead)**")
        st.write(f"- as of: `{asof.date().isoformat()}`")
        st.write(f"- median 1m return: `{(np.exp(latest_ret['q50']) - 1) * 100:+.2f}%`")
        st.write(f"- 90% CI: `[{(np.exp(latest_ret['q05_conf']) - 1) * 100:+.1f}%, "
                 f"{(np.exp(latest_ret['q95_conf']) - 1) * 100:+.1f}%]`")
        st.caption("Forward Risk answers: *what's the distribution of returns over the next month?*")

    # ---------------- methodology + shelved horizons ----------------
    with st.expander("Methodology & shelved horizons", expanded=False):
        st.markdown("""
**Method.** LightGBM quantile regression at q in {0.05, 0.25, 0.50, 0.75, 0.95}, two
targets (forward 21-day log-return; forward 21-day worst peak-to-trough log
drawdown), 32 disciplined features (yield curve, credit, equity momentum, vol,
VIX, macro Δ, sentiment, EPU, dollar, oil). Walk-forward over 4 folds with
the same boundaries as v5; BLIND >= 2021-01-01 evaluated **once**.

**Calibration.** Split-conformal CQR (Romano, Patterson, Candès, 2019)
using folds 1–3 OOF predictions (1999–2020, n=5,284 per cell) as the
calibration set. **No BLIND peeking at any stage.**

**Shelved horizons.** 3-month / 6-month / 12-month horizons all
over-cover on BLIND (96–100% vs nominal 90%) because the calibration
set's tail (1999–2020 contains dotcom −78% + GFC −55%) is heavier than the
test period's tail (2021–2026 worst NASDAQ drawdown was 2022 at −33%).
Conformal sized $\\tau$ to cover the heavier cal tail, so on BLIND the
intervals are too wide. Example: post-conformal 12-month CI90 width is
$\\pm$113% in log-return — honest, but vacuously wide. Per the
honesty contract, those horizons are NOT shipped.

**What v5 is and isn't.** v5 detects crashes that are already underway
(near-coincident). Forward Risk gives a calibrated 1-month forward
distribution. They answer different questions and are intentionally not
combined into a single number.
        """)


if __name__ == "__main__":
    render()
