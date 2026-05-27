"""
dashboard/pages/executive_overview.py
=====================================
Module 1 — Executive Overview Dashboard

High-level summary with KPI cards, trend sparklines, pipeline summary,
and latest operational metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dashboard.components.theme import render_page_header
from dashboard.components.kpi_cards import render_kpi_row, render_section_header, render_glass_container
from dashboard.components.filters import render_date_filter, filter_dataframe_by_date
from utils.helpers import format_number, format_delta, OPERATIONAL_COLS


def _plotly_dark_layout(fig, title="", height=400):
    """Apply consistent dark theme to plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, font=dict(size=16, color="#f1f5f9")),
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        height=height,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.1)",
            font=dict(size=11),
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    )
    return fig


def render(df: pd.DataFrame):
    """Render the Executive Overview page."""
    render_page_header(
        "Executive Overview",
        "Real-time operational intelligence for the UAC Program pipeline"
    )

    # Date filter
    start_date, end_date = render_date_filter(df, key_prefix="exec")
    df_filtered = filter_dataframe_by_date(df, start_date, end_date)

    if df_filtered.empty:
        st.warning("No data available for the selected date range.")
        return

    # ── KPI Cards ──
    latest = df_filtered.iloc[-1]
    prev = df_filtered.iloc[-2] if len(df_filtered) > 1 else latest

    kpis = [
        {
            "label": "Children in CBP Custody",
            "value": format_number(latest["cbp_custody"]),
            "delta": format_delta(latest["cbp_custody"], prev["cbp_custody"]),
            "delta_direction": "negative" if latest["cbp_custody"] > prev["cbp_custody"] else "positive",
            "status": "warning" if latest["cbp_custody"] > df_filtered["cbp_custody"].mean() else "",
            "icon": "🔒",
        },
        {
            "label": "Children in HHS Care",
            "value": format_number(latest["hhs_care"]),
            "delta": format_delta(latest["hhs_care"], prev["hhs_care"]),
            "delta_direction": "neutral",
            "status": "",
            "icon": "🏥",
        },
        {
            "label": "Discharged Today",
            "value": format_number(latest["discharged"]),
            "delta": format_delta(latest["discharged"], prev["discharged"]),
            "delta_direction": "positive" if latest["discharged"] >= prev["discharged"] else "negative",
            "status": "success" if latest["discharged"] > df_filtered["discharged"].mean() else "",
            "icon": "✅",
        },
        {
            "label": "Apprehended Today",
            "value": format_number(latest["apprehended"]),
            "delta": format_delta(latest["apprehended"], prev["apprehended"]),
            "delta_direction": "negative" if latest["apprehended"] > prev["apprehended"] else "positive",
            "status": "",
            "icon": "📋",
        },
    ]
    render_kpi_row(kpis)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pipeline Overview Chart ──
    render_section_header("Pipeline Overview", "Daily operational flow across all stages")

    fig = go.Figure()
    colors = {"apprehended": "#06b6d4", "cbp_custody": "#3b82f6", "transferred_out": "#6366f1",
              "hhs_care": "#10b981", "discharged": "#f59e0b"}
    for col in OPERATIONAL_COLS:
        fig.add_trace(go.Scatter(
            x=df_filtered["date"], y=df_filtered[col],
            name=col.replace("_", " ").title(),
            line=dict(color=colors.get(col, "#94a3b8"), width=2),
            mode="lines",
        ))
    fig = _plotly_dark_layout(fig, "UAC Pipeline — Daily Operations", height=450)
    st.plotly_chart(fig, use_container_width=True)

    # ── Two-column layout ──
    col1, col2 = st.columns(2)

    with col1:
        render_section_header("HHS Care Census Trend")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_filtered["date"], y=df_filtered["hhs_care"],
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.15)",
            line=dict(color="#10b981", width=2),
            name="HHS Care",
        ))
        fig2 = _plotly_dark_layout(fig2, height=350)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        render_section_header("CBP Custody vs Transfers")
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=df_filtered["date"], y=df_filtered["cbp_custody"],
            name="CBP Custody", marker_color="rgba(59, 130, 246, 0.6)",
        ))
        fig3.add_trace(go.Bar(
            x=df_filtered["date"], y=df_filtered["transferred_out"],
            name="Transferred Out", marker_color="rgba(99, 102, 241, 0.6)",
        ))
        fig3 = _plotly_dark_layout(fig3, height=350)
        fig3.update_layout(barmode="overlay")
        st.plotly_chart(fig3, use_container_width=True)

    # ── Summary Statistics Table ──
    render_section_header("Period Summary Statistics")
    stats_data = []
    for col in OPERATIONAL_COLS:
        stats_data.append({
            "Metric": col.replace("_", " ").title(),
            "Latest": format_number(latest[col]),
            "Mean": format_number(df_filtered[col].mean()),
            "Median": format_number(df_filtered[col].median()),
            "Min": format_number(df_filtered[col].min()),
            "Max": format_number(df_filtered[col].max()),
            "Std Dev": format_number(df_filtered[col].std(), decimals=1),
        })
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
