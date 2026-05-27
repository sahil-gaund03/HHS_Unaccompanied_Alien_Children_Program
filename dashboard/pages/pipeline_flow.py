"""
dashboard/pages/pipeline_flow.py
================================
Module 6 — Pipeline Flow Visualization

Sankey diagram showing the flow: Apprehension → CBP Custody → Transfer → HHS Care → Discharge
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dashboard.components.theme import render_page_header
from dashboard.components.kpi_cards import render_section_header, render_kpi_row
from dashboard.components.filters import render_date_filter, filter_dataframe_by_date
from utils.helpers import format_number


def _plotly_dark(fig, title="", height=400):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=height,
        title=dict(text=title, font=dict(size=14, color="#f1f5f9")),
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def render(df: pd.DataFrame):
    """Render the Pipeline Flow Visualization."""
    render_page_header("Pipeline Flow", "End-to-end operational flow from apprehension to discharge")

    start_date, end_date = render_date_filter(df, key_prefix="pipeline")
    df_f = filter_dataframe_by_date(df, start_date, end_date)
    if df_f.empty:
        st.warning("No data for selected range.")
        return

    # Aggregate totals for the period
    total_apprehended = df_f["apprehended"].sum()
    total_custody = df_f["cbp_custody"].sum()  # average daily census
    total_transferred = df_f["transferred_out"].sum()
    total_hhs = df_f["hhs_care"].sum()  # average daily census
    total_discharged = df_f["discharged"].sum()

    # ── KPIs ──
    render_kpi_row([
        {"label": "Total Apprehended", "value": format_number(total_apprehended), "icon": "📋", "status": ""},
        {"label": "Total Transferred", "value": format_number(total_transferred), "icon": "🔄", "status": ""},
        {"label": "Total Discharged", "value": format_number(total_discharged), "icon": "✅", "status": "success"},
        {"label": "Period Days", "value": str(len(df_f)), "icon": "📅", "status": ""},
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sankey Diagram ──
    render_section_header("Pipeline Sankey Diagram", "Flow of children through the operational pipeline")

    # Nodes: 0=Apprehended, 1=CBP Custody, 2=Transferred, 3=HHS Care, 4=Discharged
    node_labels = ["Apprehended", "CBP Custody", "Transferred Out", "HHS Care", "Discharged"]
    node_colors = ["#06b6d4", "#3b82f6", "#6366f1", "#10b981", "#f59e0b"]

    # Links
    links_source = [0, 1, 2, 3]
    links_target = [1, 2, 3, 4]
    links_value = [
        total_apprehended,
        total_transferred,
        total_transferred,  # into HHS care
        total_discharged,
    ]
    link_colors = [
        "rgba(6, 182, 212, 0.3)",
        "rgba(59, 130, 246, 0.3)",
        "rgba(99, 102, 241, 0.3)",
        "rgba(16, 185, 129, 0.3)",
    ]

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=30,
            thickness=25,
            line=dict(color="rgba(255,255,255,0.1)", width=1),
            label=node_labels,
            color=node_colors,
        ),
        link=dict(
            source=links_source,
            target=links_target,
            value=links_value,
            color=link_colors,
        ),
    ))

    fig = _plotly_dark(fig, "UAC Operational Pipeline Flow", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # ── Daily Flow Stacked Area ──
    render_section_header("Daily Flow Breakdown")

    fig2 = go.Figure()
    flow_cols = ["apprehended", "transferred_out", "discharged"]
    flow_colors = ["#06b6d4", "#6366f1", "#10b981"]
    flow_labels = ["Apprehended", "Transferred Out", "Discharged"]

    for col, color, label in zip(flow_cols, flow_colors, flow_labels):
        fig2.add_trace(go.Scatter(
            x=df_f["date"], y=df_f[col],
            name=label, stackgroup="one",
            line=dict(width=0.5, color=color),
            fillcolor=color.replace(")", ",0.3)").replace("rgb", "rgba") if "rgb" in color else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.3)",
        ))

    fig2 = _plotly_dark(fig2, "Daily Operational Flow (Stacked)", height=420)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Stage-to-Stage Efficiency ──
    render_section_header("Stage-to-Stage Efficiency")
    col1, col2 = st.columns(2)

    with col1:
        eff_data = {
            "Stage Transition": [
                "Apprehension → Custody",
                "Custody → Transfer",
                "Transfer → HHS Care",
                "HHS Care → Discharge",
            ],
            "Volume": [
                format_number(total_apprehended),
                format_number(total_transferred),
                format_number(total_transferred),
                format_number(total_discharged),
            ],
        }
        st.dataframe(pd.DataFrame(eff_data), use_container_width=True, hide_index=True)

    with col2:
        # Efficiency ratios
        ratios = [100]  # base
        if total_apprehended > 0:
            ratios.append(total_transferred / total_apprehended * 100)
        else:
            ratios.append(0)
        ratios.append(ratios[-1])  # transfer to HHS
        if total_transferred > 0:
            ratios.append(total_discharged / total_transferred * 100)
        else:
            ratios.append(0)

        fig3 = go.Figure(go.Funnel(
            y=["Apprehended", "Transferred", "In HHS Care", "Discharged"],
            x=[total_apprehended, total_transferred, total_transferred, total_discharged],
            textinfo="value+percent initial",
            marker=dict(color=["#06b6d4", "#3b82f6", "#10b981", "#f59e0b"]),
            connector=dict(line=dict(color="rgba(255,255,255,0.1)")),
        ))
        fig3 = _plotly_dark(fig3, "Pipeline Funnel", height=350)
        st.plotly_chart(fig3, use_container_width=True)
