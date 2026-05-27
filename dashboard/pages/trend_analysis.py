"""
dashboard/pages/trend_analysis.py
=================================
Module 8 — Trend Analysis Dashboard

Rolling averages, seasonal decomposition, YoY comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dashboard.components.theme import render_page_header
from dashboard.components.kpi_cards import render_section_header
from dashboard.components.filters import (
    render_date_filter, render_rolling_window_selector,
    render_metric_selector, filter_dataframe_by_date,
)
from utils.helpers import OPERATIONAL_COLS


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
    """Render the Trend Analysis dashboard."""
    render_page_header("Trend Analysis", "Rolling averages, seasonal patterns, and year-over-year comparison")

    start_date, end_date = render_date_filter(df, key_prefix="trend")
    window = render_rolling_window_selector(key="trend_window")
    metrics = render_metric_selector(OPERATIONAL_COLS, default=["hhs_care", "discharged"], key="trend_metrics")

    df_f = filter_dataframe_by_date(df, start_date, end_date)
    if df_f.empty:
        st.warning("No data for selected range.")
        return

    # ── Rolling Average Trends ──
    render_section_header(f"{window}-Day Rolling Averages")

    colors = {"apprehended": "#06b6d4", "cbp_custody": "#3b82f6", "transferred_out": "#6366f1",
              "hhs_care": "#10b981", "discharged": "#f59e0b"}

    fig = make_subplots(rows=1, cols=1)
    for col in metrics:
        raw = df_f[col]
        rolling = raw.rolling(window, min_periods=1).mean()

        # Raw as faded line
        fig.add_trace(go.Scatter(
            x=df_f["date"], y=raw, name=f"{col} (raw)",
            line=dict(color=colors.get(col, "#94a3b8"), width=0.8),
            opacity=0.3,
        ))
        # Rolling as solid line
        fig.add_trace(go.Scatter(
            x=df_f["date"], y=rolling, name=f"{col} ({window}d avg)",
            line=dict(color=colors.get(col, "#94a3b8"), width=2.5),
        ))

    fig = _plotly_dark(fig, f"Rolling {window}-Day Average Trends", height=480)
    st.plotly_chart(fig, use_container_width=True)

    # ── Seasonal Decomposition (Monthly) ──
    render_section_header("Monthly Seasonal Patterns")

    col1, col2 = st.columns(2)

    with col1:
        # Monthly average heatmap-style
        df_f_copy = df_f.copy()
        df_f_copy["month"] = df_f_copy["date"].dt.month
        df_f_copy["year"] = df_f_copy["date"].dt.year

        metric_for_season = metrics[0] if metrics else "discharged"
        monthly_pivot = df_f_copy.pivot_table(
            values=metric_for_season, index="year", columns="month", aggfunc="mean"
        )
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig2 = go.Figure(go.Heatmap(
            z=monthly_pivot.values,
            x=[month_names[m-1] for m in monthly_pivot.columns],
            y=monthly_pivot.index.astype(str),
            colorscale=[[0, "#0a0e1a"], [0.5, "#06b6d4"], [1, "#10b981"]],
            text=np.round(monthly_pivot.values, 1),
            texttemplate="%{text}",
            textfont=dict(size=10, color="white"),
            colorbar=dict(title="Avg", tickfont=dict(color="#94a3b8")),
        ))
        fig2 = _plotly_dark(fig2, f"Monthly Avg: {metric_for_season.replace('_', ' ').title()}", height=350)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Day-of-week pattern
        df_f_copy["dow"] = df_f_copy["date"].dt.day_name()
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_avg = df_f_copy.groupby("dow")[metric_for_season].mean().reindex(dow_order)

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=dow_avg.index, y=dow_avg.values,
            marker=dict(
                color=dow_avg.values,
                colorscale=[[0, "#3b82f6"], [1, "#06b6d4"]],
            ),
        ))
        fig3 = _plotly_dark(fig3, f"Day-of-Week Pattern: {metric_for_season.replace('_', ' ').title()}", height=350)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Year-over-Year Comparison ──
    render_section_header("Year-over-Year Comparison")

    years = sorted(df_f["date"].dt.year.unique())
    if len(years) >= 2:
        metric_yoy = st.selectbox("Metric for YoY comparison", metrics, key="yoy_metric")

        fig4 = go.Figure()
        yoy_colors = ["#06b6d4", "#10b981", "#f59e0b", "#6366f1", "#f43f5e"]
        for idx, year in enumerate(years):
            year_data = df_f[df_f["date"].dt.year == year].copy()
            year_data["day_of_year"] = year_data["date"].dt.dayofyear
            fig4.add_trace(go.Scatter(
                x=year_data["day_of_year"], y=year_data[metric_yoy],
                name=str(year),
                line=dict(color=yoy_colors[idx % len(yoy_colors)], width=2),
            ))
        fig4 = _plotly_dark(fig4, f"Year-over-Year: {metric_yoy.replace('_', ' ').title()}", height=420)
        fig4.update_layout(xaxis_title="Day of Year")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Need at least 2 years of data for YoY comparison.")

    # ── Correlation Heatmap ──
    render_section_header("Correlation Matrix")
    available_ops = [c for c in OPERATIONAL_COLS if c in df_f.columns]
    kpi_cols = ["transfer_efficiency_ratio", "discharge_effectiveness", "backlog_accumulation"]
    corr_cols = available_ops + [c for c in kpi_cols if c in df_f.columns]

    if len(corr_cols) > 1:
        corr_matrix = df_f[corr_cols].corr()
        fig5 = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=[c.replace("_", " ").title() for c in corr_matrix.columns],
            y=[c.replace("_", " ").title() for c in corr_matrix.index],
            colorscale=[[0, "#f43f5e"], [0.5, "#0a0e1a"], [1, "#10b981"]],
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=10, color="white"),
            zmin=-1, zmax=1,
        ))
        fig5 = _plotly_dark(fig5, "Feature Correlation Matrix", height=500)
        st.plotly_chart(fig5, use_container_width=True)
