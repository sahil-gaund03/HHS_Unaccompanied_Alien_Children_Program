"""
evaluation/evaluator.py
=======================
Step 7 — Model Evaluation and Comparison.

Computes RMSE, MAE, MAPE, R² for each model and the stacked ensemble.
Generates comparison tables, error analysis, residual plots data, and
prediction confidence charts data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.helpers import (
    EVALUATION_METRICS_PATH, save_json, logger,
)


class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics and diagnostics."""

    def __init__(self):
        self.results: Dict[str, Dict[str, float]] = {}
        self.residuals: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute RMSE, MAE, MAPE, R² for a single model."""
        # Handle edge cases
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # MAPE: avoid division by zero
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = float("inf")

        return {
            "rmse": round(float(rmse), 4),
            "mae": round(float(mae), 4),
            "mape": round(float(mape), 4),
            "r2_score": round(float(r2), 6),
        }

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------
    def evaluate_all(
        self,
        y_true: np.ndarray,
        model_predictions: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """
        Evaluate all models and return a comparison DataFrame.

        Parameters:
            y_true: Actual target values (test set)
            model_predictions: Dict of {model_name: predictions_array}

        Returns:
            DataFrame with metrics for each model
        """
        logger.info("=" * 60)
        logger.info("STEP 7 — MODEL EVALUATION")
        logger.info("=" * 60)

        rows = []
        for name, preds in model_predictions.items():
            metrics = self.compute_metrics(y_true, preds)
            self.results[name] = metrics
            self.residuals[name] = y_true - preds

            rows.append({"Model": name, **metrics})
            logger.info(
                f"  {name:25s} | R²={metrics['r2_score']:.4f} | "
                f"RMSE={metrics['rmse']:.2f} | MAE={metrics['mae']:.2f} | "
                f"MAPE={metrics['mape']:.2f}%"
            )

        comparison_df = pd.DataFrame(rows).sort_values("r2_score", ascending=False)

        # Save metrics
        save_json(self.results, EVALUATION_METRICS_PATH)

        logger.info("✓ Evaluation complete")
        return comparison_df

    # ------------------------------------------------------------------
    # Diagnostic data (for dashboard consumption)
    # ------------------------------------------------------------------
    def get_residual_data(self, model_name: str = "stacking_ensemble") -> Dict[str, Any]:
        """Return residual analysis data for plotting."""
        if model_name not in self.residuals:
            return {}

        resid = self.residuals[model_name]
        return {
            "residuals": resid.tolist(),
            "mean_residual": float(np.mean(resid)),
            "std_residual": float(np.std(resid)),
            "max_residual": float(np.max(np.abs(resid))),
            "skewness": float(pd.Series(resid).skew()),
            "kurtosis": float(pd.Series(resid).kurtosis()),
        }

    def get_prediction_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence: float = 0.95,
    ) -> Dict[str, np.ndarray]:
        """Calculate prediction intervals based on residual distribution."""
        residuals = y_true - y_pred
        std_resid = np.std(residuals)
        z = 1.96 if confidence == 0.95 else 1.645  # 95% or 90%

        return {
            "predictions": y_pred,
            "lower_bound": y_pred - z * std_resid,
            "upper_bound": y_pred + z * std_resid,
            "std_residual": std_resid,
        }

    # ------------------------------------------------------------------
    # Cross-validation fold analysis
    # ------------------------------------------------------------------
    def cross_val_fold_analysis(
        self, cv_scores: Dict[str, List[float]]
    ) -> pd.DataFrame:
        """Analyze per-fold CV scores for each model."""
        rows = []
        for model_name, scores in cv_scores.items():
            for fold_idx, score in enumerate(scores):
                rows.append({
                    "Model": model_name,
                    "Fold": fold_idx + 1,
                    "Score": score,
                })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick test with dummy data
    np.random.seed(42)
    y_true = np.random.randint(50, 200, 100).astype(float)
    preds = {
        "model_a": y_true + np.random.randn(100) * 10,
        "model_b": y_true + np.random.randn(100) * 15,
    }
    evaluator = ModelEvaluator()
    df = evaluator.evaluate_all(y_true, preds)
    print(df)
