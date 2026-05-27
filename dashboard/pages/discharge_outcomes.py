"""
dashboard/pages/discharge_outcomes.py
=====================================
Module 10 — Discharge Outcome Analytics

Discharge rate trends, effectiveness metrics, reunification analysis.
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
from utils.helpers import format_number, format_percentage


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
    """Render the Discharge Outcomes page."""
    render_page_header("Discharge Outcomes", "Reunification performance and discharge effectiveness analysis")

    start_date, end_date = render_date_filter(df, key_prefix="discharge")
    df_f = filter_dataframe_by_date(df, start_date, end_date)
    if df_f.empty:
        st.warning("No data for selected range.")
        return

    # ── KPIs ──
    total_discharged = df_f["discharged"].sum()
    avg_daily = df_f["discharged"].mean()
    latest_d = df_f["discharged"].iloc[-1]
    de = df_f.get("discharge_effectiveness", pd.Series([0]))
    avg_de = de.mean() if not de.empty else 0

    render_kpi_row([
        {"label": "Total Discharged", "value": format_number(total_discharged),
         "icon": "✅", "status": "success"},
        {"label": "Daily Average", "value": format_number(avg_daily, decimals=1),
         "icon": "📊", "status": ""},
        {"label": "Latest Day", "value": format_number(latest_d),
         "icon": "📅", "status": ""},
        {"label": "Avg Effectiveness", "value": format_percentage(avg_de),
         "icon": "🎯", "status": "success" if avg_de > 0.02 else "warning"},
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Discharge Volume Trend ──
    render_section_header("Discharge Volume Over Time")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=df_f["date"], y=df_f["discharged"],
        name="Daily Discharges",
        marker_color="rgba(16,185,129,0.5)",
    ), secondary_y=False)

    rolling_d = df_f["discharged"].rolling(14, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=df_f["date"], y=rolling_d,
        name="14-Day Average",
        line=dict(color="#f59e0b", width=2.5),
    ), secondary_y=False)

    if "hhs_care" in df_f.columns:
        fig.add_trace(go.Scatter(
            x=df_f["date"], y=df_f["hhs_care"],
            name="HHS Care Census",
            line=dict(color="#3b82f6", width=1.5, dash="dot"),
        ), secondary_y=True)

    fig.update_yaxes(title_text="Discharges", secondary_y=False,
                     gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="HHS Census", secondary_y=True,
                     gridcolor="rgba(255,255,255,0.05)")
    fig = _plotly_dark(fig, "Discharge Volume vs HHS Care Census", height=450)
    st.plotly_chart(fig, use_container_width=True)

    # ── Discharge Effectiveness ──
    col1, col2 = st.columns(2)

    with col1:
        render_section_header("Discharge Effectiveness Rate")
        if "discharge_effectiveness" in df_f.columns:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df_f["date"], y=df_f["discharge_effectiveness"] * 100,
                fill="tozeroy", fillcolor="rgba(16,185,129,0.1)",
                line=dict(color="#10b981", width=1.5),
                name="Daily Rate",
            ))
            rolling_de = (df_f["discharge_effectiveness"] * 100).rolling(7, min_periods=1).mean()
            fig2.add_trace(go.Scatter(
                x=df_f["date"], y=rolling_de,
                line=dict(color="#06b6d4", width=2.5),
                name="7-Day Average",
            ))
            fig2 = _plotly_dark(fig2, "Discharge / HHS Care (%)", height=350)
            st.plotly_chart(fig2, use_container_width=True)

    with col2:
        render_section_header("Discharge Distribution")
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=df_f["discharged"],
            nbinsx=30,
            marker_color="rgba(16,185,129,0.6)",
            name="Discharge Counts",
        ))
        mean_val = df_f["discharged"].mean()
        fig3.add_vline(x=mean_val, line=dict(color="#f59e0b", width=2, dash="dash"),
                       annotation_text=f"Mean: {mean_val:.0f}")
        fig3 = _plotly_dark(fig3, "Distribution of Daily Discharges", height=350)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Monthly Performance Table ──
    render_section_header("Monthly Discharge Performance")

    monthly = df_f.set_index("date").resample("ME").agg({
        "discharged": ["sum", "mean", "max", "min"],
        "hhs_care": "mean",
    })
    monthly.columns = ["Total", "Daily Avg", "Peak Day", "Lowest Day", "Avg HHS Census"]
    monthly = monthly.reset_index()
    monthly["date"] = monthly["date"].dt.strftime("%Y-%m")
    monthly["Effectiveness"] = monthly.apply(
        lambda r: f"{r['Daily Avg']/r['Avg HHS Census']*100:.2f}%" if r['Avg HHS Census'] > 0 else "N/A",
        axis=1,
    )
    for c in ["Total", "Daily Avg", "Peak Day", "Lowest Day", "Avg HHS Census"]:
        monthly[c] = monthly[c].round(0).astype(int)
    monthly.columns = ["Month", "Total", "Daily Avg", "Peak", "Lowest", "Avg Census", "Effectiveness"]
    st.dataframe(monthly, use_container_width=True, hide_index=True)

    # ── Insights ──
    render_section_header("💡 Discharge Insights")

    # Compute weekly pattern
    df_f_copy = df_f.copy()
    df_f_copy["dow"] = df_f_copy["date"].dt.day_name()
    dow_avg = df_f_copy.groupby("dow")["discharged"].mean()
    best_day = dow_avg.idxmax()
    worst_day = dow_avg.idxmin()

    st.markdown(f"""
    <div class="glass-card">
        <h4 style="color: #10b981; margin-top:0;">Reunification Performance Summary</h4>
        <ul style="color: #f1f5f9;">
            <li><strong>Best day for discharges:</strong> {best_day} (avg: {dow_avg[best_day]:.1f})</li>
            <li><strong>Lowest discharge day:</strong> {worst_day} (avg: {dow_avg[worst_day]:.1f})</li>
            <li><strong>Total discharged in period:</strong> {format_number(total_discharged)}</li>
            <li><strong>Average daily effectiveness:</strong> {format_percentage(avg_de)}</li>
        </ul>
        <p style="color: #94a3b8; font-size: 0.85rem;">
            Weekend discharges tend to be lower due to staffing patterns.
            Increasing weekend processing capacity could improve reunification timelines.
        </p>
    </div>
    """, unsafe_allow_html=True)
