"""
dashboard/pages/anomaly_dashboard.py
====================================
Module 5 — Anomaly Detection System

Anomaly scatter plots, timeline, severity classification.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dashboard.components.theme import render_page_header
from dashboard.components.kpi_cards import render_kpi_row, render_section_header
from utils.helpers import ANOMALY_DATA_PATH, format_number


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
    """Render the Anomaly Detection dashboard."""
    render_page_header("Anomaly Detection", "Isolation Forest-based operational anomaly identification")

    if not ANOMALY_DATA_PATH.exists():
        st.info("🔍 Anomaly data not yet generated. Run the pipeline first.")
        return

    adf = pd.read_csv(ANOMALY_DATA_PATH, parse_dates=["date"])

    anomalies = adf[adf["is_anomaly"] == 1]
    n_total = len(adf)
    n_anomalies = len(anomalies)

    # ── KPIs ──
    type_counts = anomalies["anomaly_type"].value_counts() if "anomaly_type" in anomalies.columns else pd.Series()

    render_kpi_row([
        {"label": "Total Anomalies", "value": str(n_anomalies), "icon": "🚨",
         "status": "danger" if n_anomalies > 20 else "warning"},
        {"label": "Anomaly Rate", "value": f"{n_anomalies/n_total*100:.1f}%", "icon": "📊", "status": ""},
        {"label": "Backlog Spikes", "value": str(type_counts.get("backlog_spike", 0)), "icon": "📦", "status": ""},
        {"label": "Custody Surges", "value": str(type_counts.get("custody_surge", 0)), "icon": "🔒", "status": ""},
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Anomaly Timeline ──
    render_section_header("Anomaly Timeline", "Operational anomalies detected across the observation period")

    metric_choice = st.selectbox(
        "Select metric to visualize",
        ["cbp_custody", "hhs_care", "discharged", "apprehended", "transferred_out"],
        key="anomaly_metric",
    )

    if metric_choice in adf.columns:
        fig = go.Figure()

        # Normal points
        normal = adf[adf["is_anomaly"] == 0]
        fig.add_trace(go.Scatter(
            x=normal["date"], y=normal[metric_choice],
            mode="lines", name="Normal",
            line=dict(color="rgba(148, 163, 184, 0.5)", width=1.5),
        ))

        # Anomaly points
        if len(anomalies) > 0 and metric_choice in anomalies.columns:
            color_map = {
                "backlog_spike": "#f43f5e",
                "transfer_collapse": "#f59e0b",
                "custody_surge": "#3b82f6",
                "reunification_slowdown": "#a855f7",
                "operational_anomaly": "#64748b",
            }
            for atype in anomalies["anomaly_type"].unique():
                subset = anomalies[anomalies["anomaly_type"] == atype]
                fig.add_trace(go.Scatter(
                    x=subset["date"], y=subset[metric_choice],
                    mode="markers", name=atype.replace("_", " ").title(),
                    marker=dict(
                        color=color_map.get(atype, "#f43f5e"),
                        size=10, symbol="diamond",
                        line=dict(width=1, color="white"),
                    ),
                ))

        fig = _plotly_dark(fig, f"Anomalies in {metric_choice.replace('_', ' ').title()}", height=450)
        st.plotly_chart(fig, use_container_width=True)

    # ── Anomaly Score Distribution ──
    col1, col2 = st.columns(2)

    with col1:
        render_section_header("Anomaly Score Distribution")
        if "anomaly_score" in adf.columns:
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=adf["anomaly_score"],
                nbinsx=50,
                marker_color="rgba(6, 182, 212, 0.6)",
                name="All Data",
            ))
            if len(anomalies) > 0:
                fig2.add_trace(go.Histogram(
                    x=anomalies["anomaly_score"],
                    nbinsx=20,
                    marker_color="rgba(244, 63, 94, 0.8)",
                    name="Anomalies",
                ))
            fig2 = _plotly_dark(fig2, "Isolation Forest Anomaly Scores", height=350)
            fig2.update_layout(barmode="overlay")
            st.plotly_chart(fig2, use_container_width=True)

    with col2:
        render_section_header("Anomaly Type Breakdown")
        if len(type_counts) > 0:
            fig3 = go.Figure(go.Pie(
                labels=type_counts.index.str.replace("_", " ").str.title(),
                values=type_counts.values,
                hole=0.5,
                marker=dict(colors=["#f43f5e", "#f59e0b", "#3b82f6", "#a855f7", "#64748b"]),
                textfont=dict(color="white"),
            ))
            fig3 = _plotly_dark(fig3, "Anomaly Categories", height=350)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No anomalies detected to categorize.")

    # ── Anomaly Detail Table ──
    render_section_header("Anomaly Details")
    if len(anomalies) > 0:
        display_cols = ["date", "anomaly_type", "anomaly_score", "apprehended", "cbp_custody", "hhs_care", "discharged"]
        display_cols = [c for c in display_cols if c in anomalies.columns]
        display = anomalies[display_cols].copy()
        display["date"] = display["date"].dt.strftime("%Y-%m-%d")
        display["anomaly_score"] = display["anomaly_score"].round(4)
        st.dataframe(display, use_container_width=True, hide_index=True, height=300)
