"""
dashboard/components/filters.py
===============================
Sidebar filter components for date range, metrics, and download controls.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List, Optional


def render_date_filter(
    df: pd.DataFrame,
    date_col: str = "date",
    key_prefix: str = "main",
) -> Tuple[datetime, datetime]:
    """
    Render a date range filter in the sidebar.
    Returns (start_date, end_date).
    """
    min_date = df[date_col].min().date()
    max_date = df[date_col].max().date()

    st.sidebar.markdown("### 📅 Date Range")

    date_range = st.sidebar.date_input(
        "Select range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key=f"{key_prefix}_date_range",
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        return date_range[0], date_range[1]
    return min_date, max_date


def render_metric_selector(
    options: List[str],
    default: Optional[List[str]] = None,
    key: str = "metrics",
    label: str = "📊 Metrics",
) -> List[str]:
    """Render a multi-select metric filter in the sidebar."""
    st.sidebar.markdown(f"### {label}")
    selected = st.sidebar.multiselect(
        "Choose metrics",
        options=options,
        default=default or options[:3],
        key=key,
    )
    return selected if selected else options[:1]


def render_rolling_window_selector(key: str = "rolling") -> int:
    """Render a rolling window size selector."""
    st.sidebar.markdown("### 📈 Rolling Window")
    return st.sidebar.select_slider(
        "Window size (days)",
        options=[3, 7, 14, 30, 60, 90],
        value=7,
        key=key,
    )


def render_download_button(df: pd.DataFrame, filename: str = "data.csv", label: str = "📥 Download Data"):
    """Render a CSV download button."""
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


def filter_dataframe_by_date(
    df: pd.DataFrame,
    start_date,
    end_date,
    date_col: str = "date",
) -> pd.DataFrame:
    """Filter DataFrame to the selected date range."""
    mask = (df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)
    return df[mask].copy()
