"""
preprocessing/data_cleaner.py
=============================
Step 1 — Data Cleaning Pipeline for UAC/HHS operational data.

Handles:
  • Dropping empty trailing rows
  • Parsing dates from string format (e.g., "December 21, 2025")
  • Removing comma formatting from numeric columns ("2,484" → 2484)
  • Converting dtypes to optimal numeric types
  • Renaming columns to clean snake_case
  • Sorting chronologically (ascending)
  • Duplicate detection and removal
  • Outlier detection via IQR (flagging, not removal)
  • Generating a detailed cleaning report
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.helpers import (
    RAW_CSV_PATH, CLEANED_DATA_PATH, CLEANING_REPORT_PATH,
    COLUMN_RENAME_MAP, OPERATIONAL_COLS,
    save_dataframe, save_json, iqr_bounds, logger,
)


class DataCleaner:
    """Production-grade data cleaning pipeline for UAC operational data."""

    def __init__(self, raw_path=None):
        self.raw_path = raw_path or RAW_CSV_PATH
        self.report: Dict[str, Any] = {}
        self.df_raw: pd.DataFrame = None
        self.df_clean: pd.DataFrame = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """Execute the full cleaning pipeline and return cleaned DataFrame."""
        logger.info("=" * 60)
        logger.info("STEP 1 — DATA CLEANING PIPELINE")
        logger.info("=" * 60)

        self._load_raw()
        self._drop_empty_rows()
        self._rename_columns()
        self._parse_dates()
        self._clean_numeric_columns()
        self._remove_duplicates()
        self._sort_chronologically()
        self._detect_outliers()
        self._compute_basic_stats()
        self._generate_report()
        self._save()

        logger.info(f"✓ Cleaning complete: {self.df_clean.shape[0]} rows × {self.df_clean.shape[1]} cols")
        return self.df_clean

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------
    def _load_raw(self):
        """Load the raw CSV file."""
        logger.info(f"Loading raw data from {self.raw_path}")
        self.df_raw = pd.read_csv(self.raw_path)
        self.report["raw_shape"] = list(self.df_raw.shape)
        self.report["raw_columns"] = list(self.df_raw.columns)
        logger.info(f"  Raw shape: {self.df_raw.shape}")

    def _drop_empty_rows(self):
        """Remove rows where all data columns are NaN (trailing empties)."""
        df = self.df_raw.copy()

        # Count fully empty rows
        all_cols = df.columns.tolist()
        data_cols = [c for c in all_cols if c != "Date"] if "Date" in all_cols else all_cols
        all_na = df[data_cols].isna().all(axis=1)
        all_blank = df[data_cols].apply(lambda col: col.astype(str).str.strip().eq("")).all(axis=1)
        empty_mask = all_na | all_blank

        # Also drop rows where Date is NaN/empty
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        date_empty = df[date_col].isna() | (df[date_col].astype(str).str.strip() == "")
        combined_mask = empty_mask | date_empty

        df = df[~combined_mask].reset_index(drop=True)

        self.report["empty_rows_dropped"] = int(combined_mask.sum())
        logger.info(f"  Dropped {combined_mask.sum()} empty/invalid rows")
        self.df_clean = df

    def _rename_columns(self):
        """Rename columns to clean snake_case."""
        self.df_clean = self.df_clean.rename(columns=COLUMN_RENAME_MAP)
        self.report["column_mapping"] = COLUMN_RENAME_MAP
        logger.info(f"  Renamed columns: {list(self.df_clean.columns)}")

    def _parse_dates(self):
        """Parse date strings to datetime objects."""
        self.df_clean["date"] = pd.to_datetime(self.df_clean["date"], format="mixed", dayfirst=False)

        n_failed = self.df_clean["date"].isna().sum()
        if n_failed > 0:
            logger.warning(f"  {n_failed} dates failed to parse — dropping those rows")
            self.df_clean = self.df_clean.dropna(subset=["date"]).reset_index(drop=True)

        self.report["date_parse_failures"] = int(n_failed)
        self.report["date_range"] = {
            "min": str(self.df_clean["date"].min()),
            "max": str(self.df_clean["date"].max()),
        }
        logger.info(f"  Date range: {self.df_clean['date'].min()} → {self.df_clean['date'].max()}")

    def _clean_numeric_columns(self):
        """Remove comma formatting and convert to numeric types."""
        missing_before = {}
        type_changes = {}

        for col in OPERATIONAL_COLS:
            if col not in self.df_clean.columns:
                continue

            # Track missing before cleaning
            missing_before[col] = int(self.df_clean[col].isna().sum())

            # Remove commas and quotes from string representations
            self.df_clean[col] = (
                self.df_clean[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace('"', "", regex=False)
                .str.strip()
            )

            # Convert to numeric
            self.df_clean[col] = pd.to_numeric(self.df_clean[col], errors="coerce")

            # Fill any remaining NaNs with 0 (operational data: missing = no activity)
            n_still_missing = int(self.df_clean[col].isna().sum())
            if n_still_missing > 0:
                logger.warning(f"  {col}: {n_still_missing} values couldn't be parsed → filling with 0")
                self.df_clean[col] = self.df_clean[col].fillna(0)

            # Downcast to int
            self.df_clean[col] = self.df_clean[col].astype(np.int64)
            type_changes[col] = "int64"

        self.report["missing_values_before"] = missing_before
        self.report["type_conversions"] = type_changes
        logger.info(f"  Cleaned {len(OPERATIONAL_COLS)} numeric columns → int64")

    def _remove_duplicates(self):
        """Check for and remove duplicate date entries."""
        n_before = len(self.df_clean)
        self.df_clean = self.df_clean.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
        n_dupes = n_before - len(self.df_clean)
        self.report["duplicates_removed"] = n_dupes
        if n_dupes > 0:
            logger.info(f"  Removed {n_dupes} duplicate date entries")
        else:
            logger.info("  No duplicate dates found")

    def _sort_chronologically(self):
        """Sort data by date ascending."""
        self.df_clean = self.df_clean.sort_values("date").reset_index(drop=True)
        logger.info("  Sorted chronologically (ascending)")

    def _detect_outliers(self):
        """Flag outliers using IQR method (preserves all rows)."""
        outlier_flags = {}
        outlier_counts = {}

        for col in OPERATIONAL_COLS:
            if col not in self.df_clean.columns:
                continue

            lower, upper = iqr_bounds(self.df_clean[col])
            flag_col = f"{col}_outlier"
            self.df_clean[flag_col] = (
                (self.df_clean[col] < lower) | (self.df_clean[col] > upper)
            )
            n_outliers = int(self.df_clean[flag_col].sum())
            outlier_counts[col] = n_outliers
            outlier_flags[col] = {"lower_bound": float(lower), "upper_bound": float(upper)}

            if n_outliers > 0:
                logger.info(f"  {col}: {n_outliers} outliers flagged (IQR bounds: [{lower:.0f}, {upper:.0f}])")

        self.report["outlier_bounds"] = outlier_flags
        self.report["outlier_counts"] = outlier_counts

    def _compute_basic_stats(self):
        """Compute basic descriptive statistics for the report."""
        stats = {}
        for col in OPERATIONAL_COLS:
            if col not in self.df_clean.columns:
                continue
            stats[col] = {
                "mean": float(self.df_clean[col].mean()),
                "median": float(self.df_clean[col].median()),
                "std": float(self.df_clean[col].std()),
                "min": int(self.df_clean[col].min()),
                "max": int(self.df_clean[col].max()),
                "zeros": int((self.df_clean[col] == 0).sum()),
            }
        self.report["basic_statistics"] = stats

    def _generate_report(self):
        """Finalize the cleaning report."""
        self.report["final_shape"] = list(self.df_clean.shape)
        self.report["final_dtypes"] = {col: str(dtype) for col, dtype in self.df_clean.dtypes.items()}
        self.report["final_null_counts"] = {
            col: int(self.df_clean[col].isna().sum()) for col in self.df_clean.columns
        }
        self.report["timestamp"] = str(pd.Timestamp.now())

    def _save(self):
        """Persist cleaned data and cleaning report."""
        # Drop outlier flag columns from saved data (keep in report only)
        save_cols = [c for c in self.df_clean.columns if not c.endswith("_outlier")]
        save_dataframe(self.df_clean[save_cols], CLEANED_DATA_PATH, index=False)
        save_json(self.report, CLEANING_REPORT_PATH)
        logger.info(f"  Cleaning report saved → {CLEANING_REPORT_PATH.name}")


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cleaner = DataCleaner()
    df = cleaner.run()
    print(f"\nCleaned data preview:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nShape: {df.shape}")
