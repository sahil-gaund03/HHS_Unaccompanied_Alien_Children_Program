# UAC Operational Intelligence Platform

> AI-Powered Government Healthcare Analytics for the HHS Unaccompanied Alien Children Program

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  STREAMLIT DASHBOARD                │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐     │
│  │Exec  │ │KPIs  │ │Fore- │ │Back- │ │Anom- │     │
│  │Over. │ │Mon.  │ │cast  │ │log   │ │aly   │     │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘     │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐     │
│  │Pipe  │ │SHAP  │ │Trend │ │Trans │ │Disch │     │
│  │Flow  │ │Expl. │ │Anal. │ │Eff.  │ │Out.  │     │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘     │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│              ML & ANALYTICS ENGINE                  │
│  ┌────────────┐ ┌─────────────┐ ┌──────────────┐   │
│  │ Stacking   │ │  Isolation  │ │    SHAP      │   │
│  │ Ensemble   │ │   Forest    │ │ Explainer    │   │
│  │ (5+Ridge)  │ │  Anomaly    │ │              │   │
│  └────────────┘ └─────────────┘ └──────────────┘   │
│  ┌────────────┐ ┌─────────────┐                     │
│  │  Optuna    │ │  TimeSeries │                     │
│  │  Tuning    │ │ Forecaster  │                     │
│  └────────────┘ └─────────────┘                     │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│             DATA PIPELINE                           │
│  ┌────────────┐ ┌─────────────┐ ┌──────────────┐   │
│  │   Data     │ │   Feature   │ │  Evaluation  │   │
│  │  Cleaner   │ │  Builder    │ │   Engine     │   │
│  └────────────┘ └─────────────┘ └──────────────┘   │
└─────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the ML Pipeline

```bash
# Full pipeline (with Optuna tuning — ~15-30 min)
python run_pipeline.py

# Quick run (skip tuning — ~2-5 min)
python run_pipeline.py --skip-tuning
```

### 3. Launch the Dashboard

```bash
streamlit run app.py
```

## 📁 Project Structure

```
├── data/                     # Processed data artifacts
├── preprocessing/            # Step 1: Data cleaning
├── feature_engineering/      # Step 2: 200+ feature creation
├── training/                 # Steps 4-6: ML models + Optuna
├── evaluation/               # Step 7: Metrics & comparison
├── anomaly_detection/        # Step 8: Isolation Forest
├── explainability/           # Step 9: SHAP analysis
├── dashboard/                # Step 10: Streamlit UI
│   ├── components/           # Reusable UI components
│   └── pages/                # 10 analytics modules
├── utils/                    # Shared utilities
├── saved_models/             # Trained model artifacts
├── reports/                  # Generated markdown reports
├── run_pipeline.py           # Master orchestrator
├── app.py                    # Streamlit entry point
└── requirements.txt          # Dependencies
```

## 🧠 ML Architecture

- **Base Models**: XGBoost, Random Forest, Extra Trees, Gradient Boosting, AdaBoost
- **Meta Model**: Ridge Regression
- **Validation**: TimeSeriesSplit (5 folds)
- **Tuning**: Optuna (50 trials/model)
- **Anomaly Detection**: Isolation Forest
- **Explainability**: SHAP TreeExplainer
- **Target**: R² 0.93–0.96+

## 📊 Dashboard Modules

1. **Executive Overview** — KPIs, pipeline summary, trends
2. **KPI Monitoring** — Gauges, thresholds, alerts
3. **Forecast Dashboard** — 30-day predictions + confidence intervals
4. **Backlog Prediction** — Accumulation forecasting
5. **Anomaly Detection** — Operational anomaly identification
6. **Pipeline Flow** — Sankey diagram + funnel
7. **SHAP Explainability** — Feature importance + dependence
8. **Trend Analysis** — Rolling averages, seasonality, YoY
9. **Transfer Efficiency** — CBP→HHS pipeline metrics
10. **Discharge Outcomes** — Reunification performance

## 📈 Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| ML | scikit-learn, XGBoost |
| Tuning | Optuna |
| Explainability | SHAP |
| Visualization | Plotly, Matplotlib, Seaborn |
| Data | Pandas, NumPy |

## 📝 License

Government analytics research project.
