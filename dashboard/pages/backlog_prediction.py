"""
dashboard/pages/backlog_prediction.py
=====================================
Module 4 — Backlog Prediction Panel

Backlog accumulation forecast, capacity alerts, trend decomposition.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dashboard.components.theme import render_page_header
from dashboard.components.kpi_cards import render_kpi_row, render_section_header
from dashboard.components.filters import render_date_filter, filter_dataframe_by_date
from utils.helpers import format_number


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
    """Render the Backlog Prediction panel."""
    render_page_header("Backlog Prediction", "Forecasting pipeline backlog and capacity pressure")

    start_date, end_date = render_date_filter(df, key_prefix="backlog")
    df_f = filter_dataframe_by_date(df, start_date, end_date)
    if df_f.empty:
        st.warning("No data for selected range.")
        return

    # ── Backlog KPIs ──
    if "backlog_accumulation" in df_f.columns:
        latest_bl = df_f["backlog_accumulation"].iloc[-1]
        avg_bl = df_f["backlog_accumulation"].mean()
        cum_bl = df_f.get("cumulative_backlog", df_f["backlog_accumulation"].cumsum())
        total_bl = cum_bl.iloc[-1] if len(cum_bl) > 0 else 0
        positive_days = (df_f["backlog_accumulation"] > 0).sum()

        render_kpi_row([
            {"label": "Today's Backlog", "value": format_number(latest_bl),
             "icon": "📦", "status": "danger" if latest_bl > 0 else "success"},
            {"label": "Avg Daily Backlog", "value": format_number(avg_bl, decimals=1),
             "icon": "📊", "status": ""},
            {"label": "Cumulative Backlog", "value": format_number(total_bl),
             "icon": "📈", "status": "warning" if total_bl > 0 else ""},
            {"label": "Positive Backlog Days", "value": f"{positive_days}/{len(df_f)}",
             "icon": "📅", "status": "warning" if positive_days > len(df_f)//2 else ""},
        ])
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Backlog Timeline ──
    render_section_header("Backlog Accumulation Timeline")

    if "backlog_accumulation" in df_f.columns:
        fig = go.Figure()

        colors = ["#10b981" if v <= 0 else "#f43f5e" for v in df_f["backlog_accumulation"]]
        fig.add_trace(go.Bar(
            x=df_f["date"], y=df_f["backlog_accumulation"],
            marker_color=colors, name="Daily Backlog",
            opacity=0.7,
        ))

        # Rolling average
        rolling = df_f["backlog_accumulation"].rolling(7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df_f["date"], y=rolling,
            line=dict(color="#f59e0b", width=2, dash="dash"),
            name="7-Day Average",
        ))

        fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", width=1))
        fig = _plotly_dark(fig, "Daily Backlog (Apprehensions − Discharges)", height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ── Cumulative Backlog ──
    col1, col2 = st.columns(2)
    with col1:
        render_section_header("Cumulative Backlog Growth")
        if "cumulative_backlog" in df_f.columns:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df_f["date"], y=df_f["cumulative_backlog"],
                fill="tozeroy",
                fillcolor="rgba(244, 63, 94, 0.1)",
                line=dict(color="#f43f5e", width=2),
            ))
            fig2 = _plotly_dark(fig2, height=350)
            st.plotly_chart(fig2, use_container_width=True)

    with col2:
        render_section_header("Apprehensions vs Discharges")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df_f["date"], y=df_f["apprehended"],
            name="Apprehended", line=dict(color="#06b6d4", width=2),
        ))
        fig3.add_trace(go.Scatter(
            x=df_f["date"], y=df_f["discharged"],
            name="Discharged", line=dict(color="#10b981", width=2),
        ))
        fig3 = _plotly_dark(fig3, height=350)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Monthly Backlog Summary ──
    render_section_header("Monthly Backlog Summary")
    if "backlog_accumulation" in df_f.columns:
        monthly = df_f.set_index("date").resample("ME")["backlog_accumulation"].agg(["sum", "mean", "max", "min"])
        monthly = monthly.reset_index()
        monthly["date"] = monthly["date"].dt.strftime("%Y-%m")
        monthly.columns = ["Month", "Total", "Daily Avg", "Peak", "Trough"]
        for c in ["Total", "Daily Avg", "Peak", "Trough"]:
            monthly[c] = monthly[c].round(1)
        st.dataframe(monthly, use_container_width=True, hide_index=True)
