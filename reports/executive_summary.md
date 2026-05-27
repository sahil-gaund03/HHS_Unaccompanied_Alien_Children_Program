# Executive Summary — UAC Operational Intelligence Platform
*Generated: 2026-05-27 07:38:06*

## Overview
This report presents AI-driven analytics for the HHS Unaccompanied Alien Children (UAC) Program
operational pipeline, covering apprehension through discharge.

## Key Findings

### Model Performance
| Model             |   rmse |    mae |    mape |   r2_score |
|:------------------|-------:|-------:|--------:|-----------:|
| stacking_ensemble | 2.0247 | 1.6114 | 16.6684 |   0.812665 |
| gradient_boosting | 2.1378 | 1.8193 | 21.0796 |   0.791151 |
| xgboost           | 2.1679 | 1.5243 | 13.8248 |   0.785228 |
| random_forest     | 2.8084 | 2.2833 | 34.6247 |   0.639588 |
| extra_trees       | 3.273  | 2.705  | 40.8676 |   0.510474 |
| adaboost          | 6.9588 | 5.8109 | 91.6167 |  -1.21287  |

### Anomalies Detected
- **Total Anomalies**: 21
- **Anomaly Rate**: 3.0%
- **Types**: {'operational_anomaly': 14, 'transfer_collapse': 4, 'custody_surge': 3}

### Forecast
- **Horizon**: 30 days ahead
- **Target**: discharged

### Top Predictive Features (SHAP)
| feature                     |   importance |
|:----------------------------|-------------:|
| discharged_roll_3_mean      |     50.3896  |
| discharge_effectiveness     |     37.9782  |
| discharged_roll_7_mean      |     31.6324  |
| discharged_roll_3_min       |     17.0303  |
| apprehended_roll_3_mean     |      9.3615  |
| discharged_roll_3_max       |      6.29065 |
| apprehended_lag_1           |      5.686   |
| backlog_accumulation        |      5.5169  |
| hhs_care_lag_1              |      3.79184 |
| transferred_out_roll_3_mean |      2.57289 |

## Pipeline Execution
- **Total Runtime**: 185.7 seconds
