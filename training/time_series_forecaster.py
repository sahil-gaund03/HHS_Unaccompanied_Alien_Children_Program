"""
training/time_series_forecaster.py
==================================
Time-series forecasting using the trained stacking ensemble.

Implements recursive multi-step forecasting with bootstrap confidence
intervals for operational metrics (CBP custody, HHS care, discharges).
"""

import pandas as pd
import numpy as np
import joblib
from typing import Optional, Dict, Tuple

from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.helpers import (
    FEATURED_DATA_PATH, SAVED_MODELS_DIR, FORECAST_DATA_PATH,
    OPERATIONAL_COLS, load_dataframe, save_dataframe, load_json, logger,
)


class TimeSeriesForecaster:
    """
    Multi-step recursive forecaster using the trained stacking model.

    For each forecast step, it:
      1. Uses the latest available features to predict the next value.
      2. Appends the prediction to the history.
      3. Rebuilds lag/rolling features for the next step.
      4. Repeats for the requested horizon.
    """

    def __init__(self, target_col: str = "discharged", horizon: int = 30):
        self.target_col = target_col
        self.horizon = horizon
        self.model = None
        self.scaler = None
        self.feature_names = []

    # ------------------------------------------------------------------
    # Load trained model
    # ------------------------------------------------------------------
    def _load_model(self):
        """Load the trained stacking model and scaler."""
        model_path = SAVED_MODELS_DIR / f"stacking_{self.target_col}.joblib"
        scaler_path = SAVED_MODELS_DIR / f"scaler_{self.target_col}.joblib"
        meta_path = SAVED_MODELS_DIR / f"meta_{self.target_col}.json"

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        meta = load_json(meta_path)
        self.feature_names = meta["feature_names"]
        logger.info(f"  Loaded forecast model for '{self.target_col}'")

    # ------------------------------------------------------------------
    # Recursive forecasting
    # ------------------------------------------------------------------
    def forecast(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate multi-step forecasts with confidence intervals.

        Returns DataFrame with columns:
          date, forecast, lower_bound, upper_bound
        """
        logger.info("=" * 60)
        logger.info("TIME-SERIES FORECASTING")
        logger.info("=" * 60)

        self._load_model()

        if df is None:
            df = load_dataframe(FEATURED_DATA_PATH, parse_dates=["date"])

        df = df.sort_values("date").reset_index(drop=True)

        # We'll use the last portion of the data to build up our forecast buffer
        history = df.copy()
        last_date = history["date"].max()

        forecasts = []
        lower_bounds = []
        upper_bounds = []
        forecast_dates = []

        logger.info(f"  Forecasting {self.horizon} days from {last_date.strftime('%Y-%m-%d')}...")

        for step in range(1, self.horizon + 1):
            # Get the latest row's features
            latest_row = history.iloc[[-1]]
            feature_values = latest_row[self.feature_names].values

            # Handle any NaN in features
            feature_values = np.nan_to_num(feature_values, nan=0.0)

            # Scale and predict
            scaled = self.scaler.transform(feature_values)
            pred = self.model.predict(scaled)[0]

            # Ensure non-negative
            pred = max(0, pred)

            # Bootstrap confidence interval estimation
            # Use residual-based approach: +/- historical std of recent residuals
            recent_actuals = history[self.target_col].tail(30).values
            recent_preds_features = history[self.feature_names].tail(30).values
            recent_preds_features = np.nan_to_num(recent_preds_features, nan=0.0)

            try:
                recent_preds = self.model.predict(self.scaler.transform(recent_preds_features))
                residual_std = np.std(recent_actuals - recent_preds)
            except Exception:
                residual_std = np.std(recent_actuals) * 0.1

            # Widen CI as we forecast further out
            ci_width = residual_std * (1 + 0.05 * step)
            lower = max(0, pred - 1.96 * ci_width)
            upper = pred + 1.96 * ci_width

            # Next date (skip weekends to match data pattern)
            next_date = last_date + pd.Timedelta(days=step)

            forecasts.append(pred)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
            forecast_dates.append(next_date)

            # Update history with the new prediction for recursive features
            new_row = history.iloc[-1:].copy()
            new_row["date"] = next_date
            new_row[self.target_col] = pred

            # Update lag features approximately
            for lag in [1, 3, 7, 14, 30]:
                lag_col = f"{self.target_col}_lag_{lag}"
                if lag_col in new_row.columns:
                    if len(history) >= lag:
                        new_row[lag_col] = history[self.target_col].iloc[-lag]

            # Update rolling features approximately
            for window in [3, 7, 14, 30]:
                prefix = f"{self.target_col}_roll_{window}"
                recent = history[self.target_col].tail(window)
                if f"{prefix}_mean" in new_row.columns:
                    new_row[f"{prefix}_mean"] = recent.mean()
                if f"{prefix}_std" in new_row.columns:
                    new_row[f"{prefix}_std"] = recent.std()
                if f"{prefix}_max" in new_row.columns:
                    new_row[f"{prefix}_max"] = recent.max()
                if f"{prefix}_min" in new_row.columns:
                    new_row[f"{prefix}_min"] = recent.min()

            history = pd.concat([history, new_row], ignore_index=True)

        # Build result DataFrame
        forecast_df = pd.DataFrame({
            "date": forecast_dates,
            "forecast": forecasts,
            "lower_bound": lower_bounds,
            "upper_bound": upper_bounds,
        })

        # Save
        save_dataframe(forecast_df, FORECAST_DATA_PATH, index=False)
        logger.info(f"✓ Forecast complete: {len(forecast_df)} days ahead")

        return forecast_df


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    forecaster = TimeSeriesForecaster(target_col="discharged", horizon=30)
    result = forecaster.forecast()
    print(result)
