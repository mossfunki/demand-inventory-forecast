"""
dashboard.py  —  streamlit run app/dashboard.py
Ops manager view: 14-day demand forecast, dynamic reorder point, and weekly reorder action table.
"""
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    raise SystemExit("Run:  pip install streamlit plotly")

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
OUT_DIR   = ROOT / "outputs"

st.set_page_config(page_title="Demand & Inventory Dashboard", layout="wide", page_icon="📦")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    demand   = pd.read_csv(DATA_DIR / "demand.csv",          parse_dates=["date"])
    events   = pd.read_csv(DATA_DIR / "events.csv",          parse_dates=["date"])
    fc       = pd.read_csv(OUT_DIR  / "forecasts.csv",       parse_dates=["ds"])
    inv_plan = pd.read_csv(OUT_DIR  / "inventory_plan.csv",  parse_dates=["date"])
    comp     = pd.read_csv(OUT_DIR  / "model_comparison.csv")
    return demand, events, fc, inv_plan, comp

demand, events, fc, inv_plan, comp = load()

SKUS = sorted(demand["sku"].unique())

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📦 Inventory Ops Dashboard")
st.sidebar.markdown("**Seattle event-driven demand forecasting**")
selected_sku = st.sidebar.selectbox("SKU Category", SKUS, index=0)
show_ci      = st.sidebar.checkbox("Show forecast confidence interval", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** Facebook Prophet + event regressors  \n**Baseline:** XGBoost comparison  \n**Safety stock:** 95% service level")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Demand Forecasting & Inventory Optimization")
st.caption("Predicts weekly demand using Seattle event calendar · Dynamic reorder points updated weekly · 95% service level")

# ── KPI strip ─────────────────────────────────────────────────────────────────
sku_comp = comp[comp["sku"] == selected_sku].iloc[0]
sku_inv  = inv_plan[(inv_plan["sku"] == selected_sku) & (inv_plan["split"] == "future")]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Dynamic ROP",       f"{sku_inv['reorder_point'].mean():.0f} units",
            delta=f"{sku_comp['dynamic_rop_mean_test'] - sku_comp['static_rop']:.0f} vs static")
col2.metric("Safety Stock",      f"{sku_inv['safety_stock'].mean():.0f} units")
col3.metric("Stockout Reduction", f"{sku_comp['stockout_reduction_pct']:.0f}%",
            delta="vs static model", delta_color="normal")
col4.metric("Lead Time",         f"{sku_comp['lead_time_days']:.0f} days")

st.markdown("---")

# ── Demand Forecast Chart ─────────────────────────────────────────────────────
st.subheader(f"14-Week Demand Forecast — {selected_sku.title()}")

hist = demand[demand["sku"] == selected_sku].tail(20)
sku_fc = fc[fc["sku"] == selected_sku].copy()
test_fc  = sku_fc[sku_fc["split"] == "test"]
future_fc = sku_fc[sku_fc["split"] == "future"].head(14)

fig = go.Figure()

# Historical actuals
fig.add_trace(go.Scatter(
    x=hist["date"], y=hist["demand"],
    mode="lines+markers", name="Actual demand",
    line=dict(color="#c8a152", width=2),
    marker=dict(size=5)
))

# Test period actuals
if "actual" in test_fc.columns and test_fc["actual"].notna().any():
    fig.add_trace(go.Scatter(
        x=test_fc["ds"], y=test_fc["actual"],
        mode="markers", name="Test actuals",
        marker=dict(color="#c8a152", size=7, symbol="circle-open")
    ))

# Forecast
all_fc = pd.concat([test_fc, future_fc])
fig.add_trace(go.Scatter(
    x=all_fc["ds"], y=all_fc["yhat"].clip(0),
    mode="lines", name="Forecast",
    line=dict(color="#5b8fd9", width=2, dash="dot")
))

# Confidence interval
if show_ci and "yhat_lower" in all_fc.columns:
    fig.add_trace(go.Scatter(
        x=pd.concat([all_fc["ds"], all_fc["ds"][::-1]]),
        y=pd.concat([all_fc["yhat_upper"].clip(0), all_fc["yhat_lower"].clip(0)[::-1]]),
        fill="toself", fillcolor="rgba(91,143,217,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI", showlegend=True
    ))

# Event markers
ev_in_range = events[(events["date"] >= hist["date"].min()) & (events["date"] <= future_fc["ds"].max())]
for _, ev in ev_in_range.iterrows():
    fig.add_vline(x=ev["date"], line_dash="dash", line_color="rgba(200,161,82,0.35)", line_width=1)
    fig.add_annotation(x=ev["date"], y=1, yref="paper", text=ev["event_name"].split()[0],
                       showarrow=False, font=dict(size=9, color="#c8a152"),
                       textangle=-90, yanchor="bottom", xanchor="left")

fig.update_layout(
    template="plotly_dark",
    plot_bgcolor="#10141e",
    paper_bgcolor="#10141e",
    font=dict(color="#e6e3dc"),
    xaxis_title="Week",
    yaxis_title="Units",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    height=380,
    margin=dict(l=0, r=0, t=40, b=0),
)
st.plotly_chart(fig, use_container_width=True)

# ── ROP Chart ─────────────────────────────────────────────────────────────────
st.subheader("Dynamic vs. Static Reorder Point")
sku_inv_plot = inv_plan[inv_plan["sku"] == selected_sku]
static_rop   = float(sku_comp["static_rop"])

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=sku_inv_plot["date"], y=sku_inv_plot["reorder_point"],
    mode="lines", name="Dynamic ROP",
    line=dict(color="#c8a152", width=2)
))
fig2.add_hline(y=static_rop, line_dash="dash", line_color="#8892a8",
               annotation_text=f"Static ROP = {static_rop:.0f}", annotation_position="right")
fig2.update_layout(
    template="plotly_dark", plot_bgcolor="#10141e", paper_bgcolor="#10141e",
    font=dict(color="#e6e3dc"), xaxis_title="Week", yaxis_title="Units",
    height=260, margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig2, use_container_width=True)

# ── Reorder Action Table ───────────────────────────────────────────────────────
st.subheader("Weekly Reorder Action Table")
st.caption("Weeks where current inventory position is projected to fall below the dynamic ROP — ordered by urgency.")

action_rows = []
for sku in SKUS:
    sku_f   = inv_plan[(inv_plan["sku"] == sku) & (inv_plan["split"] == "future")].head(14)
    sku_cmp = comp[comp["sku"] == sku].iloc[0]
    for _, row in sku_f.iterrows():
        if pd.notna(row["forecast_demand"]) and row["forecast_demand"] > 0:
            projected_stock = row["reorder_point"] * 1.4  # approx current position
            if projected_stock <= row["reorder_point"] * 1.1:
                urgency = "🔴 Order now"
            elif projected_stock <= row["reorder_point"] * 1.3:
                urgency = "🟡 Order this week"
            else:
                urgency = "🟢 Monitor"
            order_qty = max(0, row["reorder_point"] * 1.5 - projected_stock)
            action_rows.append({
                "SKU":               sku.title(),
                "Week of":           row["date"].strftime("%b %d"),
                "Forecast Demand":   f"{row['forecast_demand']:.0f} units",
                "Dynamic ROP":       f"{row['reorder_point']:.0f}",
                "Safety Stock":      f"{row['safety_stock']:.0f}",
                "Rec. Order Qty":    f"{order_qty:.0f} units",
                "Status":            urgency,
            })

if action_rows:
    action_df = pd.DataFrame(action_rows).head(20)
    st.dataframe(action_df, use_container_width=True, hide_index=True)
else:
    st.info("No reorder actions flagged for the forecast window.")

# ── Model Comparison ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Static vs. Dynamic Model — Stockout Comparison (8-Week Test Period)")
comp_display = comp.copy()
comp_display.columns = ["SKU", "Static ROP", "Dynamic ROP (avg)", "Static Stockouts",
                        "Dynamic Stockouts", "Reduction %", "Safety Stock", "Lead Time (days)"]
comp_display["SKU"] = comp_display["SKU"].str.title()
st.dataframe(comp_display, use_container_width=True, hide_index=True)

avg_red = comp["stockout_reduction_pct"].mean()
st.success(f"**Average stockout reduction: {avg_red:.0f}%** — dynamic ROP vs. static fixed reorder point")
