"""
dashboard/pages/transfer_efficiency.py
======================================
Module 9 — Transfer Efficiency Analytics

Transfer ratio trends, efficiency heatmaps, bottleneck identification.
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
    """Render the Transfer Efficiency Analytics page."""
    render_page_header("Transfer Efficiency", "Analyzing CBP → HHS transfer pipeline performance")

    start_date, end_date = render_date_filter(df, key_prefix="transfer")
    df_f = filter_dataframe_by_date(df, start_date, end_date)
    if df_f.empty:
        st.warning("No data for selected range.")
        return

    # ── KPIs ──
    if "transfer_efficiency_ratio" in df_f.columns:
        avg_ter = df_f["transfer_efficiency_ratio"].mean()
        latest_ter = df_f["transfer_efficiency_ratio"].iloc[-1]
        max_ter = df_f["transfer_efficiency_ratio"].max()
        min_ter = df_f["transfer_efficiency_ratio"].min()

        render_kpi_row([
            {"label": "Current Transfer Rate", "value": format_percentage(latest_ter),
             "icon": "🔄", "status": "success" if latest_ter > avg_ter else "warning"},
            {"label": "Period Average", "value": format_percentage(avg_ter),
             "icon": "📊", "status": ""},
            {"label": "Peak Efficiency", "value": format_percentage(max_ter),
             "icon": "📈", "status": "success"},
            {"label": "Lowest Efficiency", "value": format_percentage(min_ter),
             "icon": "📉", "status": "danger"},
        ])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Transfer Efficiency Trend ──
    render_section_header("Transfer Efficiency Over Time")

    if "transfer_efficiency_ratio" in df_f.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_f["date"], y=df_f["transfer_efficiency_ratio"],
            name="Daily Rate",
            line=dict(color="rgba(6,182,212,0.4)", width=1),
        ))
        # Rolling average
        rolling = df_f["transfer_efficiency_ratio"].rolling(7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df_f["date"], y=rolling,
            name="7-Day Average",
            line=dict(color="#06b6d4", width=2.5),
        ))
        # Add threshold lines
        fig.add_hline(y=avg_ter, line=dict(color="#f59e0b", width=1, dash="dot"),
                      annotation_text="Period Average")
        fig = _plotly_dark(fig, "Transfer Efficiency Ratio (Transferred / CBP Custody)", height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ── CBP Custody vs Transfers ──
    col1, col2 = st.columns(2)

    with col1:
        render_section_header("CBP Custody Load")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_f["date"], y=df_f["cbp_custody"],
            fill="tozeroy", fillcolor="rgba(59,130,246,0.12)",
            line=dict(color="#3b82f6", width=2),
            name="CBP Custody",
        ))
        rolling_custody = df_f["cbp_custody"].rolling(14, min_periods=1).mean()
        fig2.add_trace(go.Scatter(
            x=df_f["date"], y=rolling_custody,
            line=dict(color="#f59e0b", width=2, dash="dash"),
            name="14-Day Average",
        ))
        fig2 = _plotly_dark(fig2, height=350)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        render_section_header("Daily Transfers Out")
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=df_f["date"], y=df_f["transferred_out"],
            marker_color="rgba(99,102,241,0.6)",
            name="Transferred Out",
        ))
        fig3 = _plotly_dark(fig3, height=350)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Monthly Efficiency Heatmap ──
    render_section_header("Monthly Transfer Efficiency Heatmap")
    if "transfer_efficiency_ratio" in df_f.columns:
        df_copy = df_f.copy()
        df_copy["month"] = df_copy["date"].dt.month
        df_copy["year"] = df_copy["date"].dt.year

        pivot = df_copy.pivot_table(
            values="transfer_efficiency_ratio", index="year", columns="month", aggfunc="mean"
        )
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig4 = go.Figure(go.Heatmap(
            z=pivot.values * 100,
            x=[month_names[m-1] for m in pivot.columns],
            y=pivot.index.astype(str),
            colorscale=[[0, "#f43f5e"], [0.5, "#f59e0b"], [1, "#10b981"]],
            text=np.round(pivot.values * 100, 1),
            texttemplate="%{text}%",
            textfont=dict(size=10, color="white"),
            colorbar=dict(title="%", tickfont=dict(color="#94a3b8")),
        ))
        fig4 = _plotly_dark(fig4, "Monthly Average Transfer Efficiency (%)", height=300)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Bottleneck Identification ──
    render_section_header("⚠️ Bottleneck Detection")

    if "transfer_efficiency_ratio" in df_f.columns:
        threshold = df_f["transfer_efficiency_ratio"].quantile(0.1)
        bottleneck_days = df_f[df_f["transfer_efficiency_ratio"] <= threshold]

        if len(bottleneck_days) > 0:
            st.markdown(f"""
            <div class="glass-card" style="border-left: 3px solid #f43f5e;">
                <strong style="color: #f43f5e;">🚨 {len(bottleneck_days)} Low-Efficiency Days Detected</strong><br>
                <span style="color: #94a3b8;">
                    Days where transfer efficiency fell below the 10th percentile 
                    ({threshold*100:.1f}%). These represent potential operational bottlenecks
                    where custody-to-transfer pipeline was significantly constrained.
                </span>
            </div>
            """, unsafe_allow_html=True)

            display = bottleneck_days[["date", "cbp_custody", "transferred_out", "transfer_efficiency_ratio"]].copy()
            display["date"] = display["date"].dt.strftime("%Y-%m-%d")
            display["transfer_efficiency_ratio"] = (display["transfer_efficiency_ratio"] * 100).round(1).astype(str) + "%"
            display.columns = ["Date", "CBP Custody", "Transferred", "Efficiency"]
            st.dataframe(display.head(20), use_container_width=True, hide_index=True)
        else:
            st.success("✅ No significant bottlenecks detected in the selected period.")
