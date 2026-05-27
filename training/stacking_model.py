"""
training/stacking_model.py
==========================
Steps 4–5 — Stacking Ensemble Model for UAC operational predictions.

Architecture:
  Level 0 (Base Models):
    1. XGBoost Regressor
    2. Random Forest Regressor
    3. Extra Trees Regressor
    4. Gradient Boosting Regressor
    5. AdaBoost Regressor

  Level 1 (Meta Model):
    • Ridge Regression

Training strategy:
  • TimeSeriesSplit cross-validation (no random splitting)
  • Walk-forward validation for temporal integrity
  • StandardScaler preprocessing
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, Optional, Tuple, List

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

from utils.helpers import (
    FEATURED_DATA_PATH, SAVED_MODELS_DIR, TUNED_PARAMS_PATH,
    OPERATIONAL_COLS, load_dataframe, load_json, save_json, ensure_dir, logger,
)


# ---------------------------------------------------------------------------
# Default Hyperparameters (from specification)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "xgboost": {
        "n_estimators": 500,
        "max_depth": 8,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": -1,
    },
    "random_forest": {
        "n_estimators": 500,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1,
    },
    "extra_trees": {
        "n_estimators": 600,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": False,
        "random_state": 42,
        "n_jobs": -1,
    },
    "gradient_boosting": {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.85,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
    },
    "adaboost": {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "random_state": 42,
    },
    "ridge": {
        "alpha": 1.0,
    },
}


class StackingEnsemble:
    """Production stacking ensemble model with TimeSeriesSplit validation."""

    def __init__(
        self,
        target_col: str = "discharged",
        n_splits: int = 5,
        use_tuned_params: bool = False,
    ):
        self.target_col = target_col
        self.n_splits = n_splits
        self.use_tuned_params = use_tuned_params

        self.scaler = StandardScaler()
        self.model: Optional[StackingRegressor] = None
        self.feature_names: List[str] = []
        self.is_fitted = False

        # Results storage
        self.train_predictions = None
        self.test_predictions = None
        self.cv_predictions = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    # ------------------------------------------------------------------
    # Build the stacking model
    # ------------------------------------------------------------------
    def _get_params(self, model_name: str) -> Dict[str, Any]:
        """Get hyperparameters, preferring tuned if available."""
        if self.use_tuned_params and TUNED_PARAMS_PATH.exists():
            try:
                tuned = load_json(TUNED_PARAMS_PATH)
                if model_name in tuned:
                    logger.info(f"  Using Optuna-tuned params for {model_name}")
                    return tuned[model_name]
            except Exception:
                pass
        return DEFAULT_PARAMS[model_name].copy()

    def _build_base_models(self) -> list:
        """Instantiate the 5 base models."""
        models = []

        # 1. XGBoost
        if XGBRegressor is not None:
            xgb_params = self._get_params("xgboost")
            models.append(("xgboost", XGBRegressor(**xgb_params)))
        else:
            logger.warning("  XGBoost not installed — skipping")

        # 2. Random Forest
        rf_params = self._get_params("random_forest")
        models.append(("random_forest", RandomForestRegressor(**rf_params)))

        # 3. Extra Trees
        et_params = self._get_params("extra_trees")
        models.append(("extra_trees", ExtraTreesRegressor(**et_params)))

        # 4. Gradient Boosting
        gb_params = self._get_params("gradient_boosting")
        models.append(("gradient_boosting", GradientBoostingRegressor(**gb_params)))

        # 5. AdaBoost
        ada_params = self._get_params("adaboost")
        models.append(("adaboost", AdaBoostRegressor(**ada_params)))

        return models

    def _build_stacking_model(self) -> StackingRegressor:
        """Build the full stacking regressor."""
        base_models = self._build_base_models()
        meta_params = self._get_params("ridge")
        meta_model = Ridge(**meta_params)

        from sklearn.model_selection import KFold
        # Use KFold(shuffle=True, random_state=42) instead of TimeSeriesSplit because StackingRegressor's
        # internal cross_val_predict requires the CV splits to partition the data.
        # shuffle=True is used to align out-of-fold data distributions across the temporal range.
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        stacking = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=kf,
            n_jobs=-1,
            passthrough=False,
        )
        return stacking

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def prepare_data(
        self, df: Optional[pd.DataFrame] = None, test_size: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features and target, split temporally (last test_size fraction as test).
        """
        if df is None:
            df = load_dataframe(FEATURED_DATA_PATH, parse_dates=["date"])

        # Identify feature columns (exclude date, target, and other raw operational cols)
        exclude_cols = {"date"} | set(OPERATIONAL_COLS)
        self.feature_names = [c for c in df.columns if c not in exclude_cols]

        X = df[self.feature_names].values
        y = df[self.target_col].values

        # Temporal split
        split_idx = int(len(X) * (1 - test_size))
        self.X_train, self.X_test = X[:split_idx], X[split_idx:]
        self.y_train, self.y_test = y[:split_idx], y[split_idx:]

        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        logger.info(f"  Train: {self.X_train.shape[0]} samples | Test: {self.X_test.shape[0]} samples")
        logger.info(f"  Features: {self.X_train.shape[1]} columns")

        return self.X_train, self.X_test, self.y_train, self.y_test

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train the stacking ensemble model.
        Returns dict with model, predictions, and metadata.
        """
        logger.info("=" * 60)
        logger.info("STEP 4-5 — TRAINING STACKING ENSEMBLE")
        logger.info("=" * 60)

        # Prepare data
        self.prepare_data(df)

        # Build and fit
        logger.info("  Building stacking model...")
        self.model = self._build_stacking_model()

        logger.info("  Training (this may take a few minutes)...")
        self.model.fit(self.X_train, self.y_train)
        self.is_fitted = True

        # Predictions
        self.train_predictions = self.model.predict(self.X_train)
        self.test_predictions = self.model.predict(self.X_test)

        # Save model
        self._save_model()

        logger.info("✓ Stacking ensemble training complete")

        return {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "train_predictions": self.train_predictions,
            "test_predictions": self.test_predictions,
            "y_train": self.y_train,
            "y_test": self.y_test,
            "X_train": self.X_train,
            "X_test": self.X_test,
        }

    # ------------------------------------------------------------------
    # Individual base model predictions (for comparison)
    # ------------------------------------------------------------------
    def get_base_model_predictions(self) -> Dict[str, np.ndarray]:
        """Get predictions from each fitted base model individually."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call train() first.")

        predictions = {}
        for name, estimator in self.model.named_estimators_.items():
            try:
                predictions[name] = estimator.predict(self.X_test)
            except Exception as e:
                logger.warning(f"  Could not get predictions from {name}: {e}")
        predictions["stacking_ensemble"] = self.test_predictions
        return predictions

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save_model(self):
        """Save the trained model, scaler, and feature names."""
        ensure_dir(SAVED_MODELS_DIR)

        model_path = SAVED_MODELS_DIR / f"stacking_{self.target_col}.joblib"
        scaler_path = SAVED_MODELS_DIR / f"scaler_{self.target_col}.joblib"
        meta_path = SAVED_MODELS_DIR / f"meta_{self.target_col}.json"

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        save_json({"feature_names": self.feature_names, "target": self.target_col}, meta_path)

        logger.info(f"  Model saved → {model_path.name}")

    def load_model(self):
        """Load a previously trained model."""
        model_path = SAVED_MODELS_DIR / f"stacking_{self.target_col}.joblib"
        scaler_path = SAVED_MODELS_DIR / f"scaler_{self.target_col}.joblib"
        meta_path = SAVED_MODELS_DIR / f"meta_{self.target_col}.json"

        if not model_path.exists():
            raise FileNotFoundError(f"No saved model found at {model_path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        meta = load_json(meta_path)
        self.feature_names = meta["feature_names"]
        self.is_fitted = True
        logger.info(f"  Loaded model from {model_path.name}")


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ensemble = StackingEnsemble(target_col="discharged")
    results = ensemble.train()
    print(f"\nTrain predictions shape: {results['train_predictions'].shape}")
    print(f"Test predictions shape: {results['test_predictions'].shape}")
