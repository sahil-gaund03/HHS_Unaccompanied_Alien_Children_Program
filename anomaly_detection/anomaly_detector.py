"""
anomaly_detection/anomaly_detector.py
=====================================
Step 8 — Anomaly Detection using Isolation Forest.

Detects operational anomalies including:
  • Backlog spikes
  • Transfer collapses
  • Sudden custody surges
  • Reunification slowdowns
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.helpers import (
    FEATURED_DATA_PATH, ANOMALY_DATA_PATH,
    load_dataframe, save_dataframe, logger,
)


# Isolation Forest hyperparameters from specification
ISOLATION_FOREST_PARAMS = {
    "n_estimators": 300,
    "contamination": 0.03,
    "random_state": 42,
}

# Columns to monitor for anomalies
ANOMALY_FEATURES = [
    "apprehended", "cbp_custody", "transferred_out", "hhs_care", "discharged",
    "backlog_accumulation", "transfer_efficiency_ratio",
    "discharge_effectiveness", "pipeline_throughput",
]


class AnomalyDetector:
    """Isolation Forest-based anomaly detection for UAC operational data."""

    def __init__(self, params: Optional[Dict] = None):
        self.params = params or ISOLATION_FOREST_PARAMS
        self.model = IsolationForest(**self.params)
        self.scaler = StandardScaler()
        self.anomaly_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Run anomaly detection and return annotated DataFrame."""
        logger.info("=" * 60)
        logger.info("STEP 8 — ANOMALY DETECTION")
        logger.info("=" * 60)

        if df is None:
            df = load_dataframe(FEATURED_DATA_PATH, parse_dates=["date"])

        df = df.copy()

        # Select available anomaly features
        available = [c for c in ANOMALY_FEATURES if c in df.columns]
        logger.info(f"  Monitoring {len(available)} features for anomalies")

        # Prepare feature matrix
        X = df[available].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)

        # Fit and predict
        self.model.fit(X_scaled)
        labels = self.model.predict(X_scaled)  # 1=normal, -1=anomaly
        scores = self.model.decision_function(X_scaled)  # lower = more anomalous

        # Add results to DataFrame
        df["anomaly_label"] = labels
        df["anomaly_score"] = scores
        df["is_anomaly"] = (labels == -1).astype(int)

        # Classify anomaly types
        df = self._classify_anomalies(df)

        # Stats
        n_anomalies = df["is_anomaly"].sum()
        logger.info(f"  Detected {n_anomalies} anomalies ({n_anomalies/len(df)*100:.1f}%)")

        self.anomaly_df = df

        # Save
        save_dataframe(df, ANOMALY_DATA_PATH, index=False)
        logger.info("✓ Anomaly detection complete")

        return df

    # ------------------------------------------------------------------
    # Anomaly classification
    # ------------------------------------------------------------------
    def _classify_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize detected anomalies into operational types."""
        df["anomaly_type"] = "normal"
        anomaly_mask = df["is_anomaly"] == 1

        if not anomaly_mask.any():
            return df

        # Backlog spike: backlog_accumulation significantly above mean
        if "backlog_accumulation" in df.columns:
            backlog_threshold = df["backlog_accumulation"].mean() + 2 * df["backlog_accumulation"].std()
            backlog_spike = anomaly_mask & (df["backlog_accumulation"] > backlog_threshold)
            df.loc[backlog_spike, "anomaly_type"] = "backlog_spike"

        # Transfer collapse: very low transfer_efficiency_ratio
        if "transfer_efficiency_ratio" in df.columns:
            transfer_threshold = df["transfer_efficiency_ratio"].mean() - 1.5 * df["transfer_efficiency_ratio"].std()
            transfer_collapse = anomaly_mask & (df["transfer_efficiency_ratio"] < max(0, transfer_threshold))
            df.loc[transfer_collapse, "anomaly_type"] = "transfer_collapse"

        # Custody surge: CBP custody significantly above mean
        if "cbp_custody" in df.columns:
            custody_threshold = df["cbp_custody"].mean() + 2 * df["cbp_custody"].std()
            custody_surge = anomaly_mask & (df["cbp_custody"] > custody_threshold)
            df.loc[custody_surge, "anomaly_type"] = "custody_surge"

        # Reunification slowdown: very low discharge_effectiveness
        if "discharge_effectiveness" in df.columns:
            discharge_threshold = df["discharge_effectiveness"].mean() - 1.5 * df["discharge_effectiveness"].std()
            slowdown = anomaly_mask & (df["discharge_effectiveness"] < max(0, discharge_threshold))
            df.loc[slowdown, "anomaly_type"] = "reunification_slowdown"

        # Any remaining anomalies
        unclassified = anomaly_mask & (df["anomaly_type"] == "normal")
        df.loc[unclassified, "anomaly_type"] = "operational_anomaly"

        # Log type breakdown
        for atype, count in df[anomaly_mask]["anomaly_type"].value_counts().items():
            logger.info(f"    {atype}: {count}")

        return df

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_anomaly_summary(self) -> Dict:
        """Return summary statistics of detected anomalies."""
        if self.anomaly_df is None:
            return {}

        adf = self.anomaly_df[self.anomaly_df["is_anomaly"] == 1]
        return {
            "total_anomalies": int(len(adf)),
            "anomaly_rate": float(len(adf) / len(self.anomaly_df)),
            "type_counts": adf["anomaly_type"].value_counts().to_dict(),
            "date_range": {
                "earliest": str(adf["date"].min()) if len(adf) > 0 else None,
                "latest": str(adf["date"].max()) if len(adf) > 0 else None,
            },
            "avg_anomaly_score": float(adf["anomaly_score"].mean()) if len(adf) > 0 else 0,
        }


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    detector = AnomalyDetector()
    result = detector.run()
    print(f"\nAnomaly summary:")
    for k, v in detector.get_anomaly_summary().items():
        print(f"  {k}: {v}")
