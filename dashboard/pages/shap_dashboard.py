"""
dashboard/pages/shap_dashboard.py
=================================
Module 7 — SHAP Explainability Dashboard

SHAP summary plots, feature importance, waterfall, dependence plots.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dashboard.components.theme import render_page_header
from dashboard.components.kpi_cards import render_section_header
from utils.helpers import SHAP_VALUES_PATH, DATA_DIR, SAVED_MODELS_DIR, load_json


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
    """Render the SHAP Explainability dashboard."""
    render_page_header("SHAP Explainability", "Understanding what drives operational outcomes")

    # Check data availability
    imp_path = DATA_DIR / "shap_feature_importance.csv"
    if not imp_path.exists():
        st.info("🔬 SHAP data not yet generated. Run the pipeline first: `python run_pipeline.py`")
        return

    importance_df = pd.read_csv(imp_path)

    # Load SHAP values if available
    shap_values = None
    feature_names = []
    if SHAP_VALUES_PATH.exists():
        try:
            data = np.load(SHAP_VALUES_PATH, allow_pickle=True)
            shap_values = data["shap_values"]
            meta_path = SAVED_MODELS_DIR / "meta_discharged.json"
            if meta_path.exists():
                meta = load_json(meta_path)
                feature_names = meta.get("feature_names", [])
        except Exception:
            pass

    # ── Top Feature Importance Bar Chart ──
    render_section_header("🏆 Top 20 Most Important Features", "Ranked by mean absolute SHAP value")

    top_n = min(20, len(importance_df))
    top_features = importance_df.head(top_n).iloc[::-1]  # Reverse for horizontal bar

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top_features["feature"].str.replace("_", " ").str.title(),
        x=top_features["importance"],
        orientation="h",
        marker=dict(
            color=top_features["importance"],
            colorscale=[[0, "#3b82f6"], [0.5, "#06b6d4"], [1, "#10b981"]],
        ),
        text=top_features["importance"].round(3),
        textposition="outside",
        textfont=dict(color="#94a3b8", size=10),
    ))
    fig = _plotly_dark(fig, "Feature Importance (SHAP)", height=max(400, top_n * 28))
    fig.update_layout(
        xaxis_title="Mean |SHAP Value|",
        yaxis=dict(tickfont=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── SHAP Value Distribution ──
    if shap_values is not None and len(feature_names) > 0:
        col1, col2 = st.columns(2)

        with col1:
            render_section_header("SHAP Value Distribution")

            # Select top features for bee swarm-style visualization
            top_feat_names = importance_df["feature"].head(10).tolist()
            top_indices = [i for i, f in enumerate(feature_names) if f in top_feat_names]

            if top_indices:
                shap_subset = shap_values[:, top_indices]
                feat_labels = [feature_names[i].replace("_", " ").title() for i in top_indices]

                fig2 = go.Figure()
                for idx, (col_idx, label) in enumerate(zip(top_indices, feat_labels)):
                    vals = shap_values[:, col_idx]
                    fig2.add_trace(go.Box(
                        y=[label] * len(vals),
                        x=vals,
                        name=label,
                        orientation="h",
                        marker=dict(color=px.colors.qualitative.Set2[idx % 8], opacity=0.6),
                        line=dict(width=1),
                        boxmean=True,
                    ))
                fig2 = _plotly_dark(fig2, "SHAP Value Spread (Top 10 Features)", height=450)
                fig2.update_layout(showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

        with col2:
            render_section_header("Feature Impact Direction")

            # Show positive vs negative SHAP contributions
            if len(top_indices) > 0:
                pos_impact = []
                neg_impact = []
                labels = []
                for col_idx in top_indices[:10]:
                    vals = shap_values[:, col_idx]
                    pos_impact.append(np.mean(vals[vals > 0]) if (vals > 0).any() else 0)
                    neg_impact.append(np.mean(vals[vals < 0]) if (vals < 0).any() else 0)
                    labels.append(feature_names[col_idx].replace("_", " ").title())

                fig3 = go.Figure()
                fig3.add_trace(go.Bar(
                    y=labels, x=pos_impact, name="Positive Impact",
                    orientation="h", marker_color="rgba(16,185,129,0.7)",
                ))
                fig3.add_trace(go.Bar(
                    y=labels, x=neg_impact, name="Negative Impact",
                    orientation="h", marker_color="rgba(244,63,94,0.7)",
                ))
                fig3 = _plotly_dark(fig3, "Average Positive vs Negative SHAP Impact", height=450)
                fig3.update_layout(barmode="relative")
                st.plotly_chart(fig3, use_container_width=True)

    # ── SHAP Dependence Plot ──
    if shap_values is not None and len(feature_names) > 0:
        render_section_header("SHAP Dependence Plot", "How a single feature affects the prediction")

        available_features = importance_df["feature"].head(20).tolist()
        selected_feature = st.selectbox(
            "Select feature for dependence plot",
            available_features,
            key="shap_dep_feature",
        )

        if selected_feature in feature_names:
            feat_idx = feature_names.index(selected_feature)
            feat_values_col = selected_feature

            # Get feature values from df if available
            if feat_values_col in df.columns:
                n_points = min(len(df), shap_values.shape[0])
                x_vals = df[feat_values_col].tail(n_points).values
                y_vals = shap_values[-n_points:, feat_idx]

                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    x=x_vals, y=y_vals,
                    mode="markers",
                    marker=dict(
                        color=y_vals, colorscale="RdBu_r",
                        size=6, opacity=0.7,
                        colorbar=dict(title="SHAP", tickfont=dict(color="#94a3b8")),
                    ),
                    name=selected_feature,
                ))
                fig4 = _plotly_dark(fig4, f"Dependence: {selected_feature.replace('_', ' ').title()}", height=400)
                fig4.update_layout(
                    xaxis_title=selected_feature.replace("_", " ").title(),
                    yaxis_title="SHAP Value",
                )
                st.plotly_chart(fig4, use_container_width=True)

    # ── Insights ──
    render_section_header("💡 Key Insights from SHAP Analysis")
    top3 = importance_df.head(3)["feature"].tolist()
    st.markdown(f"""
    <div class="glass-card">
        <h4 style="color: #06b6d4; margin-top:0">Top Drivers of Discharge Predictions</h4>
        <ol style="color: #f1f5f9;">
            <li><strong>{top3[0].replace('_', ' ').title() if len(top3) > 0 else 'N/A'}</strong> — Most influential feature</li>
            <li><strong>{top3[1].replace('_', ' ').title() if len(top3) > 1 else 'N/A'}</strong> — Second most influential</li>
            <li><strong>{top3[2].replace('_', ' ').title() if len(top3) > 2 else 'N/A'}</strong> — Third most influential</li>
        </ol>
        <p style="color: #94a3b8; font-size: 0.9rem;">
            These features explain the majority of model decisions. Policy interventions
            targeting these areas will have the greatest impact on operational outcomes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Full Feature Importance Table ──
    render_section_header("Full Feature Importance Table")
    display = importance_df.copy()
    display["importance"] = display["importance"].round(5)
    display.columns = ["Feature", "Mean |SHAP Value|"]
    st.dataframe(display, use_container_width=True, hide_index=True, height=400)
