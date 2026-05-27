"""
dashboard/pages/kpi_monitoring.py
=================================
Module 2 — KPI Monitoring System

Gauge charts, threshold alerts, and daily/weekly/monthly KPI tracking.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dashboard.components.theme import render_page_header
from dashboard.components.kpi_cards import render_kpi_row, render_section_header
from dashboard.components.filters import render_date_filter, filter_dataframe_by_date
from utils.helpers import format_number, format_percentage


def _plotly_dark(fig, title="", height=400):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=height,
        title=dict(text=title, font=dict(size=14, color="#f1f5f9")),
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        margin=dict(l=30, r=30, t=50, b=30),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    return fig


def render(df: pd.DataFrame):
    """Render the KPI Monitoring page."""
    render_page_header("KPI Monitoring", "Track operational performance metrics with threshold-based alerts")

    start_date, end_date = render_date_filter(df, key_prefix="kpi")
    df_f = filter_dataframe_by_date(df, start_date, end_date)
    if df_f.empty:
        st.warning("No data for selected range.")
        return

    # ── Derived KPIs ──
    latest = df_f.iloc[-1]
    ter = latest.get("transfer_efficiency_ratio", 0)
    de = latest.get("discharge_effectiveness", 0)
    pt = latest.get("pipeline_throughput", 0)
    ba = latest.get("backlog_accumulation", 0)

    # ── Gauge Charts ──
    render_section_header("Operational Gauges", "Current efficiency metrics")

    cols = st.columns(4)

    # Transfer Efficiency Gauge
    with cols[0]:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ter * 100,
            title={"text": "Transfer Efficiency", "font": {"size": 13, "color": "#94a3b8"}},
            number={"suffix": "%", "font": {"color": "#f1f5f9", "size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                "bar": {"color": "#06b6d4"},
                "bgcolor": "rgba(255,255,255,0.03)",
                "steps": [
                    {"range": [0, 30], "color": "rgba(244,63,94,0.15)"},
                    {"range": [30, 70], "color": "rgba(245,158,11,0.15)"},
                    {"range": [70, 100], "color": "rgba(16,185,129,0.15)"},
                ],
                "threshold": {"line": {"color": "#f59e0b", "width": 2}, "thickness": 0.8, "value": 70},
            },
        ))
        fig = _plotly_dark(fig, height=250)
        st.plotly_chart(fig, use_container_width=True)

    # Discharge Effectiveness Gauge
    with cols[1]:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=de * 100,
            title={"text": "Discharge Rate", "font": {"size": 13, "color": "#94a3b8"}},
            number={"suffix": "%", "font": {"color": "#f1f5f9", "size": 28}},
            gauge={
                "axis": {"range": [0, 10], "tickcolor": "#64748b"},
                "bar": {"color": "#10b981"},
                "bgcolor": "rgba(255,255,255,0.03)",
                "steps": [
                    {"range": [0, 2], "color": "rgba(244,63,94,0.15)"},
                    {"range": [2, 5], "color": "rgba(245,158,11,0.15)"},
                    {"range": [5, 10], "color": "rgba(16,185,129,0.15)"},
                ],
            },
        ))
        fig = _plotly_dark(fig, height=250)
        st.plotly_chart(fig, use_container_width=True)

    # Pipeline Throughput Gauge
    with cols[2]:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=min(pt * 100, 999),
            title={"text": "Pipeline Throughput", "font": {"size": 13, "color": "#94a3b8"}},
            number={"suffix": "%", "font": {"color": "#f1f5f9", "size": 28}},
            gauge={
                "axis": {"range": [0, 500], "tickcolor": "#64748b"},
                "bar": {"color": "#6366f1"},
                "bgcolor": "rgba(255,255,255,0.03)",
            },
        ))
        fig = _plotly_dark(fig, height=250)
        st.plotly_chart(fig, use_container_width=True)

    # Backlog Indicator
    with cols[3]:
        ba_color = "#10b981" if ba <= 0 else "#f43f5e"
        fig = go.Figure(go.Indicator(
            mode="number+delta",
            value=ba,
            title={"text": "Daily Backlog", "font": {"size": 13, "color": "#94a3b8"}},
            number={"font": {"color": ba_color, "size": 36}},
            delta={"reference": 0, "increasing": {"color": "#f43f5e"}, "decreasing": {"color": "#10b981"}},
        ))
        fig = _plotly_dark(fig, height=250)
        st.plotly_chart(fig, use_container_width=True)

    # ── KPI Trends ──
    render_section_header("KPI Trends Over Time")

    kpi_cols = {
        "transfer_efficiency_ratio": ("Transfer Efficiency", "#06b6d4"),
        "discharge_effectiveness": ("Discharge Effectiveness", "#10b981"),
        "pipeline_throughput": ("Pipeline Throughput", "#6366f1"),
        "backlog_accumulation": ("Backlog Accumulation", "#f43f5e"),
    }

    tabs = st.tabs(list(v[0] for v in kpi_cols.values()))
    for tab, (col, (name, color)) in zip(tabs, kpi_cols.items()):
        with tab:
            if col in df_f.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_f["date"], y=df_f[col],
                    fill="tozeroy", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)",
                    line=dict(color=color, width=2),
                    name=name,
                ))
                # Add rolling average
                rolling = df_f[col].rolling(7, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=df_f["date"], y=rolling,
                    line=dict(color="#f59e0b", width=2, dash="dash"),
                    name="7-Day Average",
                ))
                fig = _plotly_dark(fig, f"{name} — Daily Trend", height=380)
                st.plotly_chart(fig, use_container_width=True)

    # ── Threshold Alerts ──
    render_section_header("⚠️ Threshold Alerts")

    alerts = []
    if ter < 0.3:
        alerts.append(("🔴", "Transfer Efficiency CRITICAL", f"Current: {ter*100:.1f}% (threshold: 30%)"))
    elif ter < 0.5:
        alerts.append(("🟡", "Transfer Efficiency LOW", f"Current: {ter*100:.1f}% (threshold: 50%)"))

    if de < 0.005:
        alerts.append(("🔴", "Discharge Rate CRITICAL", f"Current: {de*100:.2f}% (very low)"))

    if ba > 0:
        alerts.append(("🟡", "Positive Backlog", f"Apprehensions exceed discharges by {int(ba)}"))

    if not alerts:
        st.success("✅ All KPIs within normal operational parameters.")
    else:
        for icon, title, desc in alerts:
            st.markdown(f"""
            <div class="glass-card" style="border-left: 3px solid {'#f43f5e' if icon == '🔴' else '#f59e0b'};">
                <strong>{icon} {title}</strong><br>
                <span style="color: #94a3b8; font-size: 0.9rem;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)
