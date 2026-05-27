"""
explainability/shap_explainer.py
================================
Step 9 — SHAP Model Explainability.

Generates SHAP analysis for the XGBoost base model within the stacking
ensemble. Produces:
  • SHAP summary plots
  • Feature importance rankings
  • Waterfall plots (single prediction)
  • Dependence plots
"""

import pandas as pd
import numpy as np
import joblib
from typing import Optional, Dict, List, Any

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from utils.helpers import (
    SAVED_MODELS_DIR, SHAP_VALUES_PATH, DATA_DIR,
    load_json, save_json, ensure_dir, logger,
)


class ShapExplainer:
    """SHAP-based model explainability for the UAC stacking ensemble."""

    def __init__(self, target_col: str = "discharged"):
        self.target_col = target_col
        self.shap_values = None
        self.expected_value = None
        self.feature_names: List[str] = []
        self.feature_importance: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """
        Compute SHAP values using the XGBoost base model.

        Returns dict with SHAP values, expected value, and feature importance.
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not installed — skipping explainability")
            return {}

        logger.info("=" * 60)
        logger.info("STEP 9 — SHAP EXPLAINABILITY")
        logger.info("=" * 60)

        self.feature_names = feature_names

        # Load the stacking model and extract XGBoost estimator
        model_path = SAVED_MODELS_DIR / f"stacking_{self.target_col}.joblib"
        stacking_model = joblib.load(model_path)

        # Get the XGBoost base model
        xgb_model = None
        for name, estimator in stacking_model.named_estimators_.items():
            if "xgboost" in name.lower() or "xgb" in name.lower():
                xgb_model = estimator
                break

        if xgb_model is None:
            # Fall back to any tree-based model
            for name, estimator in stacking_model.named_estimators_.items():
                if hasattr(estimator, "feature_importances_"):
                    xgb_model = estimator
                    logger.info(f"  Using {name} for SHAP (XGBoost not found)")
                    break

        if xgb_model is None:
            logger.error("  No suitable model found for SHAP analysis")
            return {}

        # Compute SHAP values
        logger.info("  Computing SHAP values (this may take a moment)...")

        try:
            explainer = shap.TreeExplainer(xgb_model)
            self.shap_values = explainer.shap_values(X_test)
            self.expected_value = explainer.expected_value
        except Exception as e:
            logger.warning(f"  TreeExplainer failed ({e}), trying KernelExplainer with sample...")
            # Subsample for KernelExplainer (it's slow)
            n_bg = min(100, X_train.shape[0])
            bg = shap.sample(X_train, n_bg)
            explainer = shap.KernelExplainer(xgb_model.predict, bg)
            n_eval = min(100, X_test.shape[0])
            self.shap_values = explainer.shap_values(X_test[:n_eval])
            self.expected_value = explainer.expected_value

        # Feature importance from SHAP
        self.feature_importance = self._compute_feature_importance()

        # Save SHAP values
        self._save()

        logger.info(f"  Top 10 features by SHAP importance:")
        for _, row in self.feature_importance.head(10).iterrows():
            logger.info(f"    {row['feature']:40s} {row['importance']:.4f}")

        logger.info("✓ SHAP analysis complete")

        return {
            "shap_values": self.shap_values,
            "expected_value": self.expected_value,
            "feature_importance": self.feature_importance,
            "feature_names": self.feature_names,
        }

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    def _compute_feature_importance(self) -> pd.DataFrame:
        """Compute mean absolute SHAP value per feature."""
        mean_shap = np.abs(self.shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature": self.feature_names[:len(mean_shap)],
            "importance": mean_shap,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return importance_df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save(self):
        """Save SHAP values and feature importance."""
        ensure_dir(DATA_DIR)

        # Save SHAP values as compressed numpy
        np.savez_compressed(
            SHAP_VALUES_PATH,
            shap_values=self.shap_values,
            expected_value=np.array([self.expected_value] if isinstance(self.expected_value, (int, float)) else self.expected_value),
        )

        # Save feature importance as CSV
        if self.feature_importance is not None:
            self.feature_importance.to_csv(
                DATA_DIR / "shap_feature_importance.csv", index=False
            )

        logger.info(f"  SHAP data saved → {SHAP_VALUES_PATH.name}")

    def load(self) -> Dict[str, Any]:
        """Load previously computed SHAP values."""
        if not SHAP_VALUES_PATH.exists():
            raise FileNotFoundError("No saved SHAP values found")

        data = np.load(SHAP_VALUES_PATH, allow_pickle=True)
        self.shap_values = data["shap_values"]
        self.expected_value = data["expected_value"]
        if len(self.expected_value) == 1:
            self.expected_value = float(self.expected_value[0])

        imp_path = DATA_DIR / "shap_feature_importance.csv"
        if imp_path.exists():
            self.feature_importance = pd.read_csv(imp_path)

        meta_path = SAVED_MODELS_DIR / f"meta_{self.target_col}.json"
        if meta_path.exists():
            meta = load_json(meta_path)
            self.feature_names = meta.get("feature_names", [])

        return {
            "shap_values": self.shap_values,
            "expected_value": self.expected_value,
            "feature_importance": self.feature_importance,
            "feature_names": self.feature_names,
        }


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("SHAP Explainer module — run via run_pipeline.py")
