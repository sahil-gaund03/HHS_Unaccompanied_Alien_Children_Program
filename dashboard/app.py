"""
dashboard/app.py
================
Main Streamlit dashboard application with sidebar navigation
and page routing across all 10 analytics modules.
"""

import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.theme import apply_theme, render_sidebar_brand
from utils.helpers import FEATURED_DATA_PATH, CLEANED_DATA_PATH, load_dataframe, logger


# Page configuration
st.set_page_config(
    page_title="UAC Analytics Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=300)
def load_data():
    """Load the featured dataset with caching."""
    if FEATURED_DATA_PATH.exists():
        return load_dataframe(FEATURED_DATA_PATH, parse_dates=["date"])
    elif CLEANED_DATA_PATH.exists():
        return load_dataframe(CLEANED_DATA_PATH, parse_dates=["date"])
    else:
        return None


def main():
    # Apply theme
    apply_theme()

    # Sidebar branding
    render_sidebar_brand()

    # Load data
    df = load_data()

    if df is None:
        st.error("""
        ### ⚠️ No processed data found
        
        Please run the pipeline first to generate the analytics data:
        
        ```bash
        python run_pipeline.py --skip-tuning
        ```
        
        This will clean the data, engineer features, train models, and generate all 
        analytics artifacts needed by this dashboard.
        """)
        return

    # Navigation
    st.sidebar.markdown("### 🧭 Navigation")

    pages = {
        "📊 Executive Overview": "executive_overview",
        "📈 KPI Monitoring": "kpi_monitoring",
        "🔮 Forecast Dashboard": "forecast_dashboard",
        "📦 Backlog Prediction": "backlog_prediction",
        "🚨 Anomaly Detection": "anomaly_dashboard",
        "🔄 Pipeline Flow": "pipeline_flow",
        "🔬 SHAP Explainability": "shap_dashboard",
        "📉 Trend Analysis": "trend_analysis",
        "⚡ Transfer Efficiency": "transfer_efficiency",
        "✅ Discharge Outcomes": "discharge_outcomes",
        "🤖 AI Policy Copilot": "ai_assistant",
    }

    selected_page = st.sidebar.radio(
        "Select Module",
        list(pages.keys()),
        label_visibility="collapsed",
    )

    page_module = pages[selected_page]

    # Sidebar metadata
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Data Summary")
    st.sidebar.markdown(f"""
    - **Records**: {len(df):,}
    - **Features**: {len(df.columns)}
    - **Date Range**: {df['date'].min().strftime('%b %d, %Y')} — {df['date'].max().strftime('%b %d, %Y')}
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<p style="text-align:center; color:#64748b; font-size:0.7rem;">'
        'UAC Operational Intelligence Platform v1.0<br>'
        'Powered by AI/ML Analytics Engine</p>',
        unsafe_allow_html=True,
    )

    # Route to the selected page
    if page_module == "executive_overview":
        from dashboard.pages.executive_overview import render
        render(df)
    elif page_module == "kpi_monitoring":
        from dashboard.pages.kpi_monitoring import render
        render(df)
    elif page_module == "forecast_dashboard":
        from dashboard.pages.forecast_dashboard import render
        render(df)
    elif page_module == "backlog_prediction":
        from dashboard.pages.backlog_prediction import render
        render(df)
    elif page_module == "anomaly_dashboard":
        from dashboard.pages.anomaly_dashboard import render
        render(df)
    elif page_module == "pipeline_flow":
        from dashboard.pages.pipeline_flow import render
        render(df)
    elif page_module == "shap_dashboard":
        from dashboard.pages.shap_dashboard import render
        render(df)
    elif page_module == "trend_analysis":
        from dashboard.pages.trend_analysis import render
        render(df)
    elif page_module == "transfer_efficiency":
        from dashboard.pages.transfer_efficiency import render
        render(df)
    elif page_module == "discharge_outcomes":
        from dashboard.pages.discharge_outcomes import render
        render(df)
    elif page_module == "ai_assistant":
        from dashboard.pages.ai_assistant import render
        render(df)


if __name__ == "__main__":
    main()
