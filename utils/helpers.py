"""
utils/helpers.py
================
Common utility functions used across the UAC Analytics Platform.
Provides path management, data I/O, logging, and formatting helpers.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
REPORTS_DIR = PROJECT_ROOT / "reports"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Raw CSV filename
RAW_CSV_FILENAME = "HHS_Unaccompanied_Alien_Children_Program (1).csv"
RAW_CSV_PATH = PROJECT_ROOT / RAW_CSV_FILENAME

# Cleaned / featured data cache paths
CLEANED_DATA_PATH = DATA_DIR / "cleaned_data.csv"
FEATURED_DATA_PATH = DATA_DIR / "featured_data.csv"
ANOMALY_DATA_PATH = DATA_DIR / "anomaly_results.csv"
SHAP_VALUES_PATH = DATA_DIR / "shap_values.npz"
FORECAST_DATA_PATH = DATA_DIR / "forecast_results.csv"
EVALUATION_METRICS_PATH = DATA_DIR / "evaluation_metrics.json"
TUNED_PARAMS_PATH = DATA_DIR / "tuned_params.json"
CLEANING_REPORT_PATH = REPORTS_DIR / "cleaning_report.json"

# Column name mapping (raw → clean)
COLUMN_RENAME_MAP = {
    "Date": "date",
    "Children apprehended and placed in CBP custody*": "apprehended",
    "Children in CBP custody": "cbp_custody",
    "Children transferred out of CBP custody": "transferred_out",
    "Children in HHS Care": "hhs_care",
    "Children discharged from HHS Care": "discharged",
}

# Clean column names for operational metrics
OPERATIONAL_COLS = ["apprehended", "cbp_custody", "transferred_out", "hhs_care", "discharged"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger(name: str = "uac_analytics", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a project-wide logger."""
    # Reconfigure stdout/stderr to UTF-8 on Windows to prevent encoding errors
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = setup_logger()

# ---------------------------------------------------------------------------
# I/O Helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    """Save a DataFrame to CSV, creating parent dirs as needed."""
    ensure_dir(path.parent)
    df.to_csv(path, index=index)
    logger.info(f"Saved DataFrame ({df.shape[0]} rows × {df.shape[1]} cols) → {path.name}")


def load_dataframe(path: Path, parse_dates: Optional[List[str]] = None, index_col: Optional[str] = None) -> pd.DataFrame:
    """Load a DataFrame from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, parse_dates=parse_dates or False, index_col=index_col)
    logger.info(f"Loaded DataFrame ({df.shape[0]} rows × {df.shape[1]} cols) ← {path.name}")
    return df


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save a dictionary as JSON."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved JSON → {path.name}")


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file as dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------
def format_number(n: float, decimals: int = 0) -> str:
    """Format number with comma separators (e.g., 2484 → '2,484')."""
    if pd.isna(n):
        return "N/A"
    if decimals == 0:
        return f"{int(n):,}"
    return f"{n:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format as percentage string (e.g., 0.85 → '85.0%')."""
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_delta(current: float, previous: float) -> str:
    """Compute and format delta between two values."""
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return "N/A"
    delta = ((current - previous) / previous) * 100
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.1f}%"

# ---------------------------------------------------------------------------
# Date Helpers
# ---------------------------------------------------------------------------
def get_date_range_label(start: datetime, end: datetime) -> str:
    """Human-readable date range string."""
    return f"{start.strftime('%b %d, %Y')} — {end.strftime('%b %d, %Y')}"


def fiscal_year(date: datetime) -> int:
    """US Federal fiscal year (Oct 1 start)."""
    return date.year + 1 if date.month >= 10 else date.year


def fiscal_quarter(date: datetime) -> int:
    """US Federal fiscal quarter."""
    month = date.month
    if month >= 10:
        return 1
    elif month >= 7:
        return 4
    elif month >= 4:
        return 3
    else:
        return 2

# ---------------------------------------------------------------------------
# Statistical Helpers
# ---------------------------------------------------------------------------
def iqr_bounds(series: pd.Series, multiplier: float = 1.5):
    """Return (lower, upper) IQR bounds for outlier detection."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - multiplier * iqr, q3 + multiplier * iqr
