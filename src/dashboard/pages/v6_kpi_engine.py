"""Streamlit page: v6 Crash KPI Engine — Tunable x% / horizon.

Run from project root:
    streamlit run src/dashboard/pages/v6_kpi_engine.py

What you see
------------
- Two inputs: x% drawdown threshold AND horizon (trading days)
- Posterior P̂(maxDD ≥ x% in next h days) over time, with the L1/L2/L3
  gate fires overlaid on the price chart
- Per-engine pressure decomposition so you can see *why* the aggregator
  is moving
- Gate-reason breakdown (which condition is blocking)

Honesty note
------------
This dashboard surfaces the v6.0.0-alpha engine. Per
`docs/V6_HONEST_SCORECARD.md`, the alpha **failed the pre-declared
BLIND kill-criteria** (0 % recall on 2021-2026). Treat fires as
informational, not actionable, until v6.1 recalibration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v6.config import SUPPORTED_X_PCT, SUPPORTED_HORIZON_DAYS, DEFAULT_X_PCT, DEFAULT_HORIZON_DAYS
from src.v6.pipeline import CrashKPIPipeline


st.set_page_config(page_title="v6 Crash KPI Engine", layout="wide")
st.title("v6 Crash KPI Engine — Tunable Crash Detector")

st.warning(
    "**ALPHA — Failed BLIND kill-criteria.** This is the honest v6.0.0-alpha "
    "engine. See `docs/V6_HONEST_SCORECARD.md`. Treat fires as informational."
)

# --- Controls --------------------------------------------------------------
c1, c2, c3 = st.columns(3)
with c1:
    x_pct = st.selectbox("Crash threshold x %", options=list(SUPPORTED_X_PCT),
                         index=list(SUPPORTED_X_PCT).index(int(DEFAULT_X_PCT)))
with c2:
    horizon = st.selectbox("Horizon (trading days)", options=list(SUPPORTED_HORIZON_DAYS),
                           index=list(SUPPORTED_HORIZON_DAYS).index(DEFAULT_HORIZON_DAYS))
with c3:
    fit_through = st.text_input("Fit through (ISO date)", value="2019-12-31")


# --- Caching ---------------------------------------------------------------
@st.cache_resource(show_spinner="Fitting v6 pipeline...")
def fit_pipeline(fit_through_: str) -> CrashKPIPipeline:
    p = CrashKPIPipeline()
    p.fit_until(fit_through_)
    return p


@st.cache_data(show_spinner="Scoring engines + aggregator...")
def score(fit_through_: str, x: float, h: int) -> pd.DataFrame:
    p = fit_pipeline(fit_through_)
    return p.score(x_pct=float(x), horizon_td=int(h))


with st.spinner("Loading..."):
    pipe = fit_pipeline(fit_through)
    scores = score(fit_through, float(x_pct), int(horizon))
    prices = pipe.features_["_price"].loc[scores.index]

# --- Date range slider -----------------------------------------------------
min_d, max_d = scores.index.min().date(), scores.index.max().date()
date_range = st.slider("Date range", min_value=min_d, max_value=max_d,
                       value=(pd.Timestamp(fit_through).date(), max_d))
mask = (scores.index.date >= date_range[0]) & (scores.index.date <= date_range[1])
sc = scores.loc[mask]
px = prices.loc[mask]

# --- Headline metrics ------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
fires = sc["gate_fires"].sum()
m1.metric("Trading days", len(sc))
m2.metric("Gate fires", int(fires))
m3.metric("Fire rate %", f"{100 * fires / max(len(sc), 1):.2f}")
m4.metric("Latest posterior", f"{sc['posterior_mean'].iloc[-1]:.3f}",
          delta=f"conf {sc['confidence'].iloc[-1]:.2f}")

# --- Main chart: price + fires --------------------------------------------
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.3, 0.3],
                    vertical_spacing=0.04,
                    subplot_titles=("S&P 500 with gate fires",
                                    "Aggregator posterior + confidence band",
                                    "Per-engine pressure"))

fig.add_trace(go.Scatter(x=px.index, y=px.values, name="S&P 500",
                         line=dict(color="black", width=1)), row=1, col=1)
fire_dates = sc.index[sc["gate_fires"]]
if len(fire_dates) > 0:
    fig.add_trace(go.Scatter(x=fire_dates, y=px.loc[fire_dates],
                             mode="markers", name="Gate fire",
                             marker=dict(color="red", size=10, symbol="triangle-down")),
                  row=1, col=1)

# Posterior with ±1σ band
upper = sc["posterior_mean"] + sc["posterior_std"]
lower = (sc["posterior_mean"] - sc["posterior_std"]).clip(lower=0)
fig.add_trace(go.Scatter(x=sc.index, y=upper, line=dict(width=0), showlegend=False, hoverinfo="skip"), row=2, col=1)
fig.add_trace(go.Scatter(x=sc.index, y=lower, fill="tonexty", fillcolor="rgba(70,130,180,0.2)",
                         line=dict(width=0), name="±1σ band"), row=2, col=1)
fig.add_trace(go.Scatter(x=sc.index, y=sc["posterior_mean"], name="Posterior",
                         line=dict(color="steelblue", width=2)), row=2, col=1)
fig.add_hline(y=0.60, line=dict(color="red", dash="dash"), row=2, col=1)

for col, color in [("pressure_anomaly", "purple"), ("pressure_regime", "orange"),
                   ("pressure_analog", "green"), ("pressure_causal", "brown")]:
    fig.add_trace(go.Scatter(x=sc.index, y=sc[col], name=col.replace("pressure_", ""),
                             line=dict(color=color, width=1)), row=3, col=1)

fig.update_layout(height=900, hovermode="x unified", margin=dict(t=40, b=10))
st.plotly_chart(fig, use_container_width=True)

# --- Gate reason breakdown -------------------------------------------------
st.subheader("Gate-reason frequency (non-firing days)")
reason_counts = sc.loc[~sc["gate_fires"], "gate_reason"].value_counts()
st.bar_chart(reason_counts)

# --- Recent table ----------------------------------------------------------
st.subheader("Latest 20 trading days")
display = sc.tail(20)[[
    "posterior_mean", "confidence",
    "pressure_anomaly", "pressure_regime", "pressure_analog", "pressure_causal",
    "layer1_z", "layer2_z", "layer3_dd", "gate_fires", "gate_reason",
]].round(3)
st.dataframe(display, use_container_width=True)
