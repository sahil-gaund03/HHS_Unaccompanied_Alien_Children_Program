"""
feature_engineering/feature_builder.py
======================================
Step 2 — Advanced Feature Engineering for UAC operational data.

Generates 200+ features including:
  • Date features (day, month, year, quarter, week, day_of_week, is_weekend)
  • Lag features (1, 3, 7, 14, 30 days) for all operational columns
  • Rolling window features (mean, std, max, min) for 3, 7, 14, 30-day windows
  • Trend features (daily growth, pct change, momentum, velocity, rate of change)
  • Derived KPI features (transfer efficiency, discharge effectiveness, throughput, backlog)
"""

import pandas as pd
import numpy as np
from typing import List, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.helpers import (
    CLEANED_DATA_PATH, FEATURED_DATA_PATH,
    OPERATIONAL_COLS, load_dataframe, save_dataframe, logger,
)


class FeatureBuilder:
    """Generates advanced features from cleaned UAC operational data."""

    LAG_PERIODS = [1, 3, 7, 14, 30]
    ROLLING_WINDOWS = [3, 7, 14, 30]
    TARGET_COLS = OPERATIONAL_COLS  # All 5 operational columns

    def __init__(self, df: Optional[pd.DataFrame] = None):
        if df is not None:
            self.df = df.copy()
        else:
            self.df = load_dataframe(CLEANED_DATA_PATH, parse_dates=["date"])

        self.feature_cols: List[str] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """Execute the full feature engineering pipeline."""
        logger.info("=" * 60)
        logger.info("STEP 2 — ADVANCED FEATURE ENGINEERING")
        logger.info("=" * 60)

        # Ensure sorted by date
        self.df = self.df.sort_values("date").reset_index(drop=True)

        self._create_date_features()
        self._create_lag_features()
        self._create_rolling_features()
        self._create_trend_features()
        self._create_derived_kpi_features()
        self._handle_infinities()
        self._drop_warmup_rows()
        self._save()

        logger.info(f"✓ Feature engineering complete: {len(self.feature_cols)} features created")
        logger.info(f"  Final shape: {self.df.shape}")
        return self.df

    # ------------------------------------------------------------------
    # Date Features
    # ------------------------------------------------------------------
    def _create_date_features(self):
        """Extract temporal features from the date column."""
        logger.info("  Creating date features...")
        dt = self.df["date"].dt

        self.df["day"] = dt.day
        self.df["month"] = dt.month
        self.df["year"] = dt.year
        self.df["quarter"] = dt.quarter
        self.df["week_number"] = dt.isocalendar().week.astype(int)
        self.df["day_of_week"] = dt.dayofweek  # 0=Monday, 6=Sunday
        self.df["is_weekend"] = (dt.dayofweek >= 5).astype(int)
        self.df["day_of_year"] = dt.dayofyear

        # Cyclical encoding for month and day_of_week (helps ML models)
        self.df["month_sin"] = np.sin(2 * np.pi * self.df["month"] / 12)
        self.df["month_cos"] = np.cos(2 * np.pi * self.df["month"] / 12)
        self.df["dow_sin"] = np.sin(2 * np.pi * self.df["day_of_week"] / 7)
        self.df["dow_cos"] = np.cos(2 * np.pi * self.df["day_of_week"] / 7)

        date_feats = [
            "day", "month", "year", "quarter", "week_number",
            "day_of_week", "is_weekend", "day_of_year",
            "month_sin", "month_cos", "dow_sin", "dow_cos",
        ]
        self.feature_cols.extend(date_feats)
        logger.info(f"    → {len(date_feats)} date features")

    # ------------------------------------------------------------------
    # Lag Features
    # ------------------------------------------------------------------
    def _create_lag_features(self):
        """Create lag features for all operational columns."""
        logger.info("  Creating lag features...")
        count = 0
        for col in self.TARGET_COLS:
            for lag in self.LAG_PERIODS:
                feat_name = f"{col}_lag_{lag}"
                self.df[feat_name] = self.df[col].shift(lag)
                self.feature_cols.append(feat_name)
                count += 1
        logger.info(f"    → {count} lag features")

    # ------------------------------------------------------------------
    # Rolling Window Features
    # ------------------------------------------------------------------
    def _create_rolling_features(self):
        """Create rolling mean, std, max, min for multiple windows."""
        logger.info("  Creating rolling window features...")
        count = 0
        for col in self.TARGET_COLS:
            for window in self.ROLLING_WINDOWS:
                prefix = f"{col}_roll_{window}"

                self.df[f"{prefix}_mean"] = self.df[col].rolling(window=window, min_periods=1).mean()
                self.df[f"{prefix}_std"] = self.df[col].rolling(window=window, min_periods=1).std()
                self.df[f"{prefix}_max"] = self.df[col].rolling(window=window, min_periods=1).max()
                self.df[f"{prefix}_min"] = self.df[col].rolling(window=window, min_periods=1).min()

                new_feats = [f"{prefix}_mean", f"{prefix}_std", f"{prefix}_max", f"{prefix}_min"]
                self.feature_cols.extend(new_feats)
                count += 4

        logger.info(f"    → {count} rolling window features")

    # ------------------------------------------------------------------
    # Trend Features
    # ------------------------------------------------------------------
    def _create_trend_features(self):
        """Create daily growth, pct change, momentum, velocity, rate of change."""
        logger.info("  Creating trend features...")
        count = 0
        for col in self.TARGET_COLS:
            # Daily growth (first difference)
            self.df[f"{col}_daily_growth"] = self.df[col].diff()

            # Percentage change
            self.df[f"{col}_pct_change"] = self.df[col].pct_change()

            # Momentum (second-order difference: acceleration)
            self.df[f"{col}_momentum"] = self.df[col].diff().diff()

            # Velocity (exponential weighted moving average of changes)
            self.df[f"{col}_velocity"] = self.df[col].diff().ewm(span=7, min_periods=1).mean()

            # Rate of change (7-day)
            self.df[f"{col}_roc_7"] = self.df[col].pct_change(periods=7)

            new_feats = [
                f"{col}_daily_growth", f"{col}_pct_change",
                f"{col}_momentum", f"{col}_velocity", f"{col}_roc_7",
            ]
            self.feature_cols.extend(new_feats)
            count += 5

        logger.info(f"    → {count} trend features")

    # ------------------------------------------------------------------
    # Derived KPI Features
    # ------------------------------------------------------------------
    def _create_derived_kpi_features(self):
        """Create business KPI features from operational columns."""
        logger.info("  Creating derived KPI features...")

        # Transfer Efficiency Ratio: Transferred_Out / CBP_Custody
        self.df["transfer_efficiency_ratio"] = np.where(
            self.df["cbp_custody"] > 0,
            self.df["transferred_out"] / self.df["cbp_custody"],
            0.0,
        )

        # Discharge Effectiveness: Discharged / HHS_Care
        self.df["discharge_effectiveness"] = np.where(
            self.df["hhs_care"] > 0,
            self.df["discharged"] / self.df["hhs_care"],
            0.0,
        )

        # Pipeline Throughput: Discharges / Apprehensions
        self.df["pipeline_throughput"] = np.where(
            self.df["apprehended"] > 0,
            self.df["discharged"] / self.df["apprehended"],
            0.0,
        )

        # Backlog Accumulation: Apprehensions − Discharges
        self.df["backlog_accumulation"] = self.df["apprehended"] - self.df["discharged"]

        # Cumulative backlog
        self.df["cumulative_backlog"] = self.df["backlog_accumulation"].cumsum()

        # Net flow into HHS care
        self.df["net_hhs_flow"] = self.df["transferred_out"] - self.df["discharged"]

        # CBP processing rate
        self.df["cbp_processing_rate"] = np.where(
            self.df["cbp_custody"] > 0,
            self.df["transferred_out"] / self.df["cbp_custody"],
            0.0,
        )

        # Custody utilization change
        self.df["custody_change"] = self.df["cbp_custody"].diff()

        # HHS care change
        self.df["hhs_care_change"] = self.df["hhs_care"].diff()

        kpi_feats = [
            "transfer_efficiency_ratio", "discharge_effectiveness",
            "pipeline_throughput", "backlog_accumulation",
            "cumulative_backlog", "net_hhs_flow",
            "cbp_processing_rate", "custody_change", "hhs_care_change",
        ]
        self.feature_cols.extend(kpi_feats)
        logger.info(f"    → {len(kpi_feats)} derived KPI features")

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    def _handle_infinities(self):
        """Replace infinite values with NaN, then forward-fill."""
        inf_count = np.isinf(self.df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            logger.info(f"  Replacing {inf_count} infinite values")
        self.df = self.df.replace([np.inf, -np.inf], np.nan)

    def _drop_warmup_rows(self):
        """Drop initial rows that have NaN from lag/rolling calculations."""
        max_lag = max(self.LAG_PERIODS)  # 30
        n_before = len(self.df)
        self.df = self.df.iloc[max_lag:].reset_index(drop=True)

        # Forward-fill any remaining NaN in feature columns, then fill with 0
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].ffill().fillna(0)

        logger.info(f"  Dropped {n_before - len(self.df)} warmup rows (lag={max_lag})")

    def _save(self):
        """Persist the featured DataFrame."""
        save_dataframe(self.df, FEATURED_DATA_PATH, index=False)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def get_feature_columns(self) -> List[str]:
        """Return list of engineered feature column names (excludes date and raw operational cols)."""
        exclude = {"date"} | set(OPERATIONAL_COLS)
        return [c for c in self.df.columns if c not in exclude]


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    builder = FeatureBuilder()
    df = builder.run()
    print(f"\nFeatured data shape: {df.shape}")
    print(f"Feature columns ({len(builder.get_feature_columns())}):")
    for c in builder.get_feature_columns():
        print(f"  • {c}")
