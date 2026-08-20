"""Streamlit page: v6 Crash KPI Engine — tunable x% / horizon.

Run from project root:
    streamlit run src/dashboard/pages/v6_kpi_engine.py

What you see
------------
- Two inputs: x% drawdown threshold AND horizon (trading days), both applied
  at inference against a single fitted model
- Posterior P(maxDD >= x% in next h days) over time, with gate fires
  overlaid on the price chart
- Per-engine pressure decomposition, plus the calibrated log-odds
  contribution each engine actually made to the posterior
- The crash **archetype** behind each fire (credit-led / rate-led /
  valuation-led / shock-led)
- Gate-reason breakdown showing which condition is blocking

Honesty note
------------
See `docs/V6_HONEST_SCORECARD.md` for the current verdict, including the
walk-forward folds that still fail their kill criteria and the disclosure
that the 2021+ window is no longer a clean holdout.
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

from src.v6.config import (
    SUPPORTED_X_PCT, SUPPORTED_HORIZON_DAYS, DEFAULT_X_PCT, DEFAULT_HORIZON_DAYS,
)
from src.v6.pipeline import CrashKPIPipeline
from src.v6.engines.aggregator import ENGINE_NAMES


st.set_page_config(page_title="v6 Crash KPI Engine", layout="wide")
st.title("v6 Crash KPI Engine — Tunable Crash Detector")

st.info(
    "**v6.1.** BLIND (2021-2026) passes all five kill criteria, but several "
    "walk-forward folds do not, and the 2021+ window was inspected during "
    "development so it is no longer a clean holdout. Read "
    "`docs/V6_HONEST_SCORECARD.md` before acting on anything here."
)

# --- Controls --------------------------------------------------------------
c1, c2, c3 = st.columns(3)
with c1:
    x_pct = st.selectbox("Crash threshold x %", options=list(SUPPORTED_X_PCT),
                         index=list(SUPPORTED_X_PCT).index(DEFAULT_X_PCT))
with c2:
    horizon = st.selectbox("Horizon (trading days)", options=list(SUPPORTED_HORIZON_DAYS),
                           index=list(SUPPORTED_HORIZON_DAYS).index(DEFAULT_HORIZON_DAYS))
with c3:
    fit_through = st.text_input("Fit through (ISO date)", value="2020-12-31")


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

index_name = (pipe.price_col_ or "index").replace("_close", "").upper()
gate = pipe.aggregator.gate_summary()

# --- Date range slider -----------------------------------------------------
min_d, max_d = scores.index.min().date(), scores.index.max().date()
default_start = max(pd.Timestamp(fit_through).date(), min_d)
date_range = st.slider("Date range", min_value=min_d, max_value=max_d,
                       value=(default_start, max_d))
mask = (scores.index.date >= date_range[0]) & (scores.index.date <= date_range[1])
sc = scores.loc[mask]
px = prices.loc[mask]

# --- Headline metrics ------------------------------------------------------
m1, m2, m3, m4, m5 = st.columns(5)
fires = int(sc["gate_fires"].sum())
m1.metric("Trading days", len(sc))
m2.metric("Gate fires", fires)
m3.metric("Fire rate %", f"{100 * fires / max(len(sc), 1):.2f}")
m4.metric("Latest posterior", f"{sc['posterior_mean'].iloc[-1]:.3f}",
          delta=f"conf {sc['confidence'].iloc[-1]:.2f}")
m5.metric("Base rate (train)", f"{pipe.aggregator.base_rate_:.3f}",
          help="Unconditional P(maxDD >= x%) in the training window. The "
               "posterior is only informative relative to this.")

# --- Engine weights + gate operating point ---------------------------------
with st.expander("Model audit — engine weights, skill, and the tuned gate"):
    a, b = st.columns(2)
    with a:
        st.caption("Per-engine skill and pooling weight (learned on training)")
        st.dataframe(pipe.aggregator.explain().round(4), use_container_width=True)
    with b:
        st.caption("Gate operating point (tuned on the training window)")
        st.json(gate)
    if pipe.engine_errors_:
        st.warning(f"Engines that could not fit or score: {pipe.engine_errors_}")

# --- Main chart: price + fires --------------------------------------------
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.3, 0.3],
    vertical_spacing=0.05,
    subplot_titles=(f"{index_name} with gate fires",
                    "Aggregator posterior (vs gate threshold and base rate)",
                    "Per-engine pressure"),
)

fig.add_trace(go.Scatter(x=px.index, y=px.values, name=index_name,
                         line=dict(color="black", width=1)), row=1, col=1)
fire_dates = sc.index[sc["gate_fires"].fillna(False)]
if len(fire_dates) > 0:
    tags = sc.loc[fire_dates, "archetype"].fillna("n/a")
    fig.add_trace(go.Scatter(
        x=fire_dates, y=px.loc[fire_dates], mode="markers", name="Gate fire",
        marker=dict(color="red", size=9, symbol="triangle-down"),
        text=tags, hovertemplate="%{x|%Y-%m-%d}<br>archetype: %{text}<extra></extra>",
    ), row=1, col=1)
fig.update_yaxes(type="log", row=1, col=1)

fig.add_trace(go.Scatter(x=sc.index, y=sc["posterior_mean"], name="Posterior",
                         line=dict(color="steelblue", width=2)), row=2, col=1)
post_thr = gate.get("posterior")
if post_thr is not None:
    fig.add_hline(y=float(post_thr), line=dict(color="red", dash="dash"),
                  annotation_text="gate threshold", row=2, col=1)
fig.add_hline(y=float(pipe.aggregator.base_rate_),
              line=dict(color="grey", dash="dot"),
              annotation_text="base rate", row=2, col=1)

for col, color in [("pressure_anomaly", "purple"), ("pressure_regime", "orange"),
                   ("pressure_analog", "green"), ("pressure_causal", "brown")]:
    if col in sc.columns:
        fig.add_trace(go.Scatter(x=sc.index, y=sc[col], name=col.replace("pressure_", ""),
                                 line=dict(color=color, width=1)), row=3, col=1)

fig.update_layout(height=900, hovermode="x unified", margin=dict(t=40, b=10))
st.plotly_chart(fig, use_container_width=True)

# --- Archetype + gate reason breakdown -------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Crash archetype when the gate fired")
    if fires:
        st.bar_chart(sc.loc[sc["gate_fires"].fillna(False), "archetype"].value_counts())
    else:
        st.caption("No fires in this window.")
with right:
    st.subheader("What blocked the gate (non-firing days)")
    st.bar_chart(sc.loc[~sc["gate_fires"].fillna(False), "gate_reason"].value_counts().head(10))

# --- Recent table ----------------------------------------------------------
st.subheader("Latest 20 trading days")
cols = ["posterior_mean", "confidence", "archetype"]
cols += [f"pressure_{n}" for n in ENGINE_NAMES if f"pressure_{n}" in sc.columns]
cols += [f"contribution_{n}" for n in ENGINE_NAMES if f"contribution_{n}" in sc.columns]
cols += ["layer1_z", "layer2_z", "layer3_dd", "gate_fires", "gate_reason"]
st.dataframe(sc.tail(20)[cols].round(3), use_container_width=True)
st.caption(
    "`contribution_*` is each engine's weighted log-odds contribution to the "
    "posterior: positive pushes toward a crash, negative away from one. They "
    "sum to the total shift from the base rate."
)
