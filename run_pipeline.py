"""
run_pipeline.py
===============
Master orchestrator that executes the full analytics pipeline:

  1. Data Cleaning
  2. Feature Engineering
  3. (Optional) Optuna Hyperparameter Tuning
  4. Stacking Ensemble Training
  5. Model Evaluation
  6. Anomaly Detection
  7. SHAP Explainability
  8. Time-Series Forecasting
  9. Report Generation

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-tuning
    python run_pipeline.py --target hhs_care --horizon 60
"""

import argparse
import time
import sys
import os
from pathlib import Path
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.helpers import (
    logger, ensure_dir, REPORTS_DIR, save_json,
    FEATURED_DATA_PATH, EVALUATION_METRICS_PATH,
)
from preprocessing.data_cleaner import DataCleaner
from feature_engineering.feature_builder import FeatureBuilder
from training.stacking_model import StackingEnsemble
from training.hyperparameter_tuning import OptunaOptimizer
from training.time_series_forecaster import TimeSeriesForecaster
from evaluation.evaluator import ModelEvaluator
from anomaly_detection.anomaly_detector import AnomalyDetector
from explainability.shap_explainer import ShapExplainer


def parse_args():
    parser = argparse.ArgumentParser(description="UAC Analytics Pipeline")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip Optuna tuning")
    parser.add_argument("--target", default="discharged", help="Target column")
    parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon (days)")
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials per model")
    return parser.parse_args()


def generate_reports(
    evaluation_df, anomaly_summary, shap_importance, forecast_df, target_col, elapsed
):
    """Generate markdown reports in the reports/ directory."""
    ensure_dir(REPORTS_DIR)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---- Executive Summary ----
    exec_summary = f"""# Executive Summary — UAC Operational Intelligence Platform
*Generated: {timestamp}*

## Overview
This report presents AI-driven analytics for the HHS Unaccompanied Alien Children (UAC) Program
operational pipeline, covering apprehension through discharge.

## Key Findings

### Model Performance
{evaluation_df.to_markdown(index=False) if evaluation_df is not None else 'Model not yet evaluated.'}

### Anomalies Detected
- **Total Anomalies**: {anomaly_summary.get('total_anomalies', 'N/A')}
- **Anomaly Rate**: {anomaly_summary.get('anomaly_rate', 0)*100:.1f}%
- **Types**: {anomaly_summary.get('type_counts', {})}

### Forecast
- **Horizon**: {len(forecast_df) if forecast_df is not None else 0} days ahead
- **Target**: {target_col}

### Top Predictive Features (SHAP)
{shap_importance.head(10).to_markdown(index=False) if shap_importance is not None else 'SHAP not computed.'}

## Pipeline Execution
- **Total Runtime**: {elapsed:.1f} seconds
"""
    with open(REPORTS_DIR / "executive_summary.md", "w", encoding="utf-8") as f:
        f.write(exec_summary)

    # ---- Technical Report ----
    tech_report = f"""# Technical Report — UAC Analytics Platform
*Generated: {timestamp}*

## Architecture
- **Base Models**: XGBoost, Random Forest, Extra Trees, Gradient Boosting, AdaBoost
- **Meta Model**: Ridge Regression (alpha=1.0)
- **Validation**: TimeSeriesSplit (5 folds)
- **Anomaly Detection**: Isolation Forest (n_estimators=300, contamination=0.03)
- **Explainability**: SHAP TreeExplainer

## Model Comparison
{evaluation_df.to_markdown(index=False) if evaluation_df is not None else 'N/A'}

## Feature Engineering
- Date features: 12
- Lag features: 25 (5 columns × 5 lag periods)
- Rolling features: 80 (5 columns × 4 windows × 4 aggregations)
- Trend features: 25 (5 columns × 5 metrics)
- KPI features: 9
- **Total**: ~151 engineered features

## Data Summary
- Time period: Jan 2023 – Dec 2025
- Observations: ~691 (after warmup row removal)
- Temporal split: 85% train / 15% test
"""
    with open(REPORTS_DIR / "technical_report.md", "w", encoding="utf-8") as f:
        f.write(tech_report)

    # ---- Policy Recommendations ----
    policy_report = f"""# Policy Recommendations — UAC Program
*Generated: {timestamp}*

## Operational Insights

### 1. Capacity Planning
The dramatic decline in HHS care occupancy (from ~11,000 to ~2,000) between 2024-2025
indicates significant operational changes. Predictive models should be recalibrated
quarterly to account for policy-driven regime changes.

### 2. Bottleneck Detection
Anomaly detection identified {anomaly_summary.get('total_anomalies', 0)} operational anomalies.
Transfer collapses and backlog spikes should trigger automatic alerts for resource reallocation.

### 3. Discharge Optimization
SHAP analysis reveals the key drivers of discharge timing. Focusing on the top 5 predictive
features could improve reunification workflow efficiency.

### 4. Weekend/Holiday Gaps
Data gaps during weekends and holidays suggest reporting cadence issues.
Continuous monitoring would improve forecasting accuracy.

### 5. Early Warning System
The Isolation Forest anomaly detection system can serve as an early warning
for operational disruptions when integrated into real-time dashboards.
"""
    with open(REPORTS_DIR / "policy_recommendations.md", "w", encoding="utf-8") as f:
        f.write(policy_report)

    logger.info(f"  Reports generated → {REPORTS_DIR}")


def main():
    args = parse_args()
    start_time = time.time()

    logger.info("+" + "=" * 58 + "+")
    logger.info("|   UAC OPERATIONAL INTELLIGENCE PLATFORM - PIPELINE RUN   |")
    logger.info("+" + "=" * 58 + "+")
    logger.info(f"  Target: {args.target} | Horizon: {args.horizon} days")
    logger.info(f"  Tuning: {'Enabled' if not args.skip_tuning else 'Skipped'}")
    logger.info("")

    # ====================================================================
    # STEP 1: Data Cleaning
    # ====================================================================
    cleaner = DataCleaner()
    df_clean = cleaner.run()

    # ====================================================================
    # STEP 2: Feature Engineering
    # ====================================================================
    builder = FeatureBuilder(df=df_clean)
    df_featured = builder.run()

    # ====================================================================
    # STEP 6 (optional): Optuna Tuning
    # ====================================================================
    if not args.skip_tuning:
        from utils.helpers import OPERATIONAL_COLS
        exclude = {"date"} | set(OPERATIONAL_COLS)
        feat_cols = [c for c in df_featured.columns if c not in exclude]
        X_tune = df_featured[feat_cols].values
        y_tune = df_featured[args.target].values

        optimizer = OptunaOptimizer(
            target_col=args.target,
            n_trials=args.n_trials,
        )
        optimizer.run(X_tune, y_tune)

    # ====================================================================
    # STEPS 4-5: Stacking Ensemble Training
    # ====================================================================
    ensemble = StackingEnsemble(
        target_col=args.target,
        use_tuned_params=not args.skip_tuning,
    )
    results = ensemble.train(df=df_featured)

    # ====================================================================
    # STEP 7: Evaluation
    # ====================================================================
    evaluator = ModelEvaluator()
    base_preds = ensemble.get_base_model_predictions()
    eval_df = evaluator.evaluate_all(results["y_test"], base_preds)
    print("\n" + eval_df.to_string(index=False) + "\n")

    # ====================================================================
    # STEP 8: Anomaly Detection
    # ====================================================================
    detector = AnomalyDetector()
    anomaly_results = detector.run(df=df_featured)
    anomaly_summary = detector.get_anomaly_summary()

    # ====================================================================
    # STEP 9: SHAP Explainability
    # ====================================================================
    shap_explainer = ShapExplainer(target_col=args.target)
    shap_results = shap_explainer.run(
        X_train=results["X_train"],
        X_test=results["X_test"],
        feature_names=ensemble.feature_names,
    )
    shap_importance = shap_results.get("feature_importance")

    # ====================================================================
    # Time-Series Forecasting
    # ====================================================================
    forecaster = TimeSeriesForecaster(target_col=args.target, horizon=args.horizon)
    forecast_df = forecaster.forecast(df=df_featured)

    # ====================================================================
    # STEP 12: Report Generation
    # ====================================================================
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("STEP 12 — GENERATING REPORTS")
    logger.info("=" * 60)
    generate_reports(eval_df, anomaly_summary, shap_importance, forecast_df, args.target, elapsed)

    # ====================================================================
    # Final Summary
    # ====================================================================
    logger.info("")
    logger.info("+" + "=" * 58 + "+")
    logger.info("|              PIPELINE EXECUTION COMPLETE                 |")
    logger.info("+" + "=" * 58 + "+")
    logger.info(f"  Total time: {elapsed:.1f} seconds")
    logger.info(f"  Cleaned data: {df_clean.shape}")
    logger.info(f"  Featured data: {df_featured.shape}")
    logger.info(f"  Best R2: {eval_df['r2_score'].max():.4f}")
    logger.info(f"  Anomalies: {anomaly_summary.get('total_anomalies', 0)}")
    logger.info(f"  Forecast: {len(forecast_df)} days ahead")
    logger.info("")
    logger.info("  -> Launch dashboard: streamlit run app.py")
    logger.info("")


if __name__ == "__main__":
    main()
