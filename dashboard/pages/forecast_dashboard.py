"""
dashboard/pages/forecast_dashboard.py
=====================================
Module 3 — Forecast Dashboard

30-day forecast with confidence bands, actual vs predicted comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dashboard.components.theme import render_page_header
from dashboard.components.kpi_cards import render_kpi_row, render_section_header
from utils.helpers import FORECAST_DATA_PATH, FEATURED_DATA_PATH, format_number


def _plotly_dark(fig, title="", height=400):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=height,
        title=dict(text=title, font=dict(size=14, color="#f1f5f9")),
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    return fig


def render(df: pd.DataFrame):
    """Render the Forecast Dashboard."""
    render_page_header("Forecast Dashboard", "AI-powered operational forecasting with confidence intervals")

    # Load forecast data
    forecast_available = FORECAST_DATA_PATH.exists()

    if not forecast_available:
        st.info("🔮 Forecast data not yet generated. Run the pipeline first: `python run_pipeline.py`")
        return

    forecast_df = pd.read_csv(FORECAST_DATA_PATH, parse_dates=["date"])

    # ── Forecast KPI Summary ──
    avg_forecast = forecast_df["forecast"].mean()
    max_forecast = forecast_df["forecast"].max()
    min_forecast = forecast_df["forecast"].min()
    ci_width = (forecast_df["upper_bound"] - forecast_df["lower_bound"]).mean()

    render_kpi_row([
        {"label": "Avg Forecast", "value": format_number(avg_forecast), "icon": "📊", "status": ""},
        {"label": "Peak Forecast", "value": format_number(max_forecast), "icon": "📈", "status": "warning"},
        {"label": "Min Forecast", "value": format_number(min_forecast), "icon": "📉", "status": ""},
        {"label": "Avg CI Width", "value": format_number(ci_width, decimals=1), "icon": "📐", "status": ""},
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main Forecast Chart ──
    render_section_header("Discharge Forecast — Next 30 Days")

    # Get historical tail for context
    historical_tail = df.tail(60) if len(df) > 60 else df

    fig = go.Figure()

    # Historical actuals
    fig.add_trace(go.Scatter(
        x=historical_tail["date"], y=historical_tail["discharged"],
        name="Historical (Actual)",
        line=dict(color="#10b981", width=2),
        mode="lines",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["forecast"],
        name="Forecast",
        line=dict(color="#06b6d4", width=3),
        mode="lines+markers",
        marker=dict(size=4),
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
        y=pd.concat([forecast_df["upper_bound"], forecast_df["lower_bound"][::-1]]),
        fill="toself",
        fillcolor="rgba(6, 182, 212, 0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Interval",
        showlegend=True,
    ))

    # Transition marker
    if not historical_tail.empty:
        last_actual_date = historical_tail["date"].iloc[-1]
        fig.add_vline(
            x=last_actual_date.timestamp() * 1000,
            line=dict(color="rgba(245,158,11,0.5)", width=2, dash="dash"),
            annotation_text="Forecast Start",
            annotation_font=dict(color="#f59e0b", size=11),
        )

    fig = _plotly_dark(fig, "Discharge Forecast with 95% Confidence Interval", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # ── Forecast Details Table ──
    col1, col2 = st.columns(2)

    with col1:
        render_section_header("Forecast Data")
        display_df = forecast_df.copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        display_df.columns = ["Date", "Forecast", "Lower Bound", "Upper Bound"]
        for c in ["Forecast", "Lower Bound", "Upper Bound"]:
            display_df[c] = display_df[c].round(1)
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)

    with col2:
        render_section_header("Confidence Interval Width Over Horizon")
        ci_fig = go.Figure()
        ci_fig.add_trace(go.Bar(
            x=forecast_df["date"],
            y=forecast_df["upper_bound"] - forecast_df["lower_bound"],
            marker_color="rgba(6, 182, 212, 0.5)",
            name="CI Width",
        ))
        ci_fig = _plotly_dark(ci_fig, "Uncertainty Grows with Forecast Horizon", height=400)
        st.plotly_chart(ci_fig, use_container_width=True)

    # ── Download ──
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Forecast Data", csv, "forecast_results.csv", "text/csv")
