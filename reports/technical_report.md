# Technical Report — UAC Analytics Platform
*Generated: 2026-05-27 07:38:06*

## Architecture
- **Base Models**: XGBoost, Random Forest, Extra Trees, Gradient Boosting, AdaBoost
- **Meta Model**: Ridge Regression (alpha=1.0)
- **Validation**: TimeSeriesSplit (5 folds)
- **Anomaly Detection**: Isolation Forest (n_estimators=300, contamination=0.03)
- **Explainability**: SHAP TreeExplainer

## Model Comparison
| Model             |   rmse |    mae |    mape |   r2_score |
|:------------------|-------:|-------:|--------:|-----------:|
| stacking_ensemble | 2.0247 | 1.6114 | 16.6684 |   0.812665 |
| gradient_boosting | 2.1378 | 1.8193 | 21.0796 |   0.791151 |
| xgboost           | 2.1679 | 1.5243 | 13.8248 |   0.785228 |
| random_forest     | 2.8084 | 2.2833 | 34.6247 |   0.639588 |
| extra_trees       | 3.273  | 2.705  | 40.8676 |   0.510474 |
| adaboost          | 6.9588 | 5.8109 | 91.6167 |  -1.21287  |

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
