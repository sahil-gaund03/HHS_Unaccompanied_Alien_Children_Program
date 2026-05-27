"""
training/hyperparameter_tuning.py
=================================
Step 6 — Optuna Hyperparameter Optimization for base models.

Uses Optuna to optimize each base model's hyperparameters with
TimeSeriesSplit cross-validation, minimizing RMSE.
"""

import numpy as np
import json
from typing import Dict, Any, Optional

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.metrics import make_scorer, mean_squared_error

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

from utils.helpers import (
    FEATURED_DATA_PATH, TUNED_PARAMS_PATH, OPERATIONAL_COLS,
    load_dataframe, save_json, logger,
)


def _rmse_scorer():
    """Return a negative RMSE scorer for sklearn cross_val_score."""
    return make_scorer(mean_squared_error, squared=False, greater_is_better=False)


class OptunaOptimizer:
    """Optuna-based hyperparameter tuner for UAC ensemble models."""

    def __init__(
        self,
        target_col: str = "discharged",
        n_trials: int = 50,
        n_splits: int = 5,
    ):
        self.target_col = target_col
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.best_params: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Run Optuna optimization for all base models."""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not installed — skipping tuning")
            return {}

        logger.info("=" * 60)
        logger.info("STEP 6 — OPTUNA HYPERPARAMETER TUNING")
        logger.info("=" * 60)

        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        # Tune each model
        if XGBRegressor is not None:
            self.best_params["xgboost"] = self._tune_xgboost(X, y, tscv)
        self.best_params["random_forest"] = self._tune_rf(X, y, tscv)
        self.best_params["extra_trees"] = self._tune_et(X, y, tscv)
        self.best_params["gradient_boosting"] = self._tune_gb(X, y, tscv)

        # Save
        save_json(self.best_params, TUNED_PARAMS_PATH)
        logger.info("✓ Tuning complete — best params saved")

        return self.best_params

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------
    def _tune_xgboost(self, X, y, tscv) -> Dict[str, Any]:
        """Tune XGBoost hyperparameters."""
        logger.info("  Tuning XGBoost...")

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 0.5),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
                "random_state": 42,
                "verbosity": 0,
                "n_jobs": -1,
            }
            model = XGBRegressor(**params)
            scores = cross_val_score(model, X, y, cv=tscv, scoring=_rmse_scorer(), n_jobs=1)
            return -scores.mean()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        best = study.best_params
        best["random_state"] = 42
        best["verbosity"] = 0
        best["n_jobs"] = -1
        logger.info(f"    Best RMSE: {study.best_value:.4f}")
        return best

    # ------------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------------
    def _tune_rf(self, X, y, tscv) -> Dict[str, Any]:
        """Tune Random Forest hyperparameters."""
        logger.info("  Tuning Random Forest...")

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 5, 25),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "bootstrap": True,
                "random_state": 42,
                "n_jobs": -1,
            }
            model = RandomForestRegressor(**params)
            scores = cross_val_score(model, X, y, cv=tscv, scoring=_rmse_scorer(), n_jobs=1)
            return -scores.mean()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        best = study.best_params
        best["bootstrap"] = True
        best["random_state"] = 42
        best["n_jobs"] = -1
        logger.info(f"    Best RMSE: {study.best_value:.4f}")
        return best

    # ------------------------------------------------------------------
    # Extra Trees
    # ------------------------------------------------------------------
    def _tune_et(self, X, y, tscv) -> Dict[str, Any]:
        """Tune Extra Trees hyperparameters."""
        logger.info("  Tuning Extra Trees...")

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 900),
                "max_depth": trial.suggest_int("max_depth", 10, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "bootstrap": False,
                "random_state": 42,
                "n_jobs": -1,
            }
            model = ExtraTreesRegressor(**params)
            scores = cross_val_score(model, X, y, cv=tscv, scoring=_rmse_scorer(), n_jobs=1)
            return -scores.mean()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        best = study.best_params
        best["bootstrap"] = False
        best["random_state"] = 42
        best["n_jobs"] = -1
        logger.info(f"    Best RMSE: {study.best_value:.4f}")
        return best

    # ------------------------------------------------------------------
    # Gradient Boosting
    # ------------------------------------------------------------------
    def _tune_gb(self, X, y, tscv) -> Dict[str, Any]:
        """Tune Gradient Boosting hyperparameters."""
        logger.info("  Tuning Gradient Boosting...")

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "random_state": 42,
            }
            model = GradientBoostingRegressor(**params)
            scores = cross_val_score(model, X, y, cv=tscv, scoring=_rmse_scorer(), n_jobs=1)
            return -scores.mean()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        best = study.best_params
        best["random_state"] = 42
        logger.info(f"    Best RMSE: {study.best_value:.4f}")
        return best


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from utils.helpers import load_dataframe
    df = load_dataframe(FEATURED_DATA_PATH, parse_dates=["date"])
    exclude = {"date"} | set(OPERATIONAL_COLS)
    feat_cols = [c for c in df.columns if c not in exclude]
    X = df[feat_cols].values
    y = df["discharged"].values

    optimizer = OptunaOptimizer(n_trials=20)
    best = optimizer.run(X, y)
    print(json.dumps(best, indent=2))
