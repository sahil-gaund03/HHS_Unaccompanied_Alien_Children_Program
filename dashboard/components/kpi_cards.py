"""
dashboard/components/kpi_cards.py
=================================
Reusable KPI card components with delta indicators and status coloring.
"""

import streamlit as st
from typing import Optional


def render_kpi_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_direction: str = "neutral",  # "positive", "negative", "neutral"
    status: str = "",  # "success", "warning", "danger"
    icon: str = "",
):
    """
    Render a glassmorphism KPI card.

    Args:
        label: KPI name/label
        value: Formatted value string
        delta: Optional delta string (e.g., "+12.5%")
        delta_direction: Color direction for delta
        status: Card accent color
        icon: Optional emoji icon
    """
    status_class = f" {status}" if status else ""
    delta_html = ""
    if delta:
        delta_html = f'<div class="kpi-delta {delta_direction}">{delta}</div>'

    icon_html = f'<div style="font-size:1.5rem;margin-bottom:4px">{icon}</div>' if icon else ""

    st.markdown(f"""
    <div class="kpi-card{status_class}">
        {icon_html}
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_kpi_row(kpis: list):
    """
    Render a row of KPI cards.

    Args:
        kpis: List of dicts with keys: label, value, delta, delta_direction, status, icon
    """
    cols = st.columns(len(kpis))
    for col, kpi in zip(cols, kpis):
        with col:
            render_kpi_card(**kpi)


def render_metric_badge(label: str, status: str = "info"):
    """Render a status badge."""
    st.markdown(
        f'<span class="badge badge-{status}">{label}</span>',
        unsafe_allow_html=True,
    )


def render_glass_container(content_html: str):
    """Wrap content in a glassmorphism card."""
    st.markdown(
        f'<div class="glass-card">{content_html}</div>',
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str = ""):
    """Render a styled section header."""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<div class="section-subheader">{subtitle}</div>',
            unsafe_allow_html=True,
        )
