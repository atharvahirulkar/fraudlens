# Fraud Detection

> Real-time ML inference API - XGBoost · LightGBM · MLflow · FastAPI · Docker

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What this is

A production-style machine learning system that trains, tracks, and serves a fraud detection model via a REST API. XGBoost and LightGBM are compared across 20+ MLflow experiment runs on the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection). The best model by AUC-PR is registered in the MLflow model registry and loaded at startup by a FastAPI inference service - containerized with Docker and deployed to the cloud.

The focus is not just model accuracy. It is the full MLOps loop: reproducible training, experiment tracking, model versioning, containerized serving, and measurable inference latency.

---

## Architecture

```
IEEE-CIS Dataset (Kaggle)
        │
        ▼
  Feature Engineering         ← Pandas, Scikit-learn
  · Transaction aggregates
  · Temporal features
  · Encoding + imputation
        │
        ▼
  MLflow Experiment Tracking
  · XGBoost runs  (N hyperparameter configs)
  · LightGBM runs (N hyperparameter configs)
  · Logged: AUC-ROC, AUC-PR, F1, precision, recall
        │
        ▼
  MLflow Model Registry       ← best run promoted to "production"
        │
        ▼
  FastAPI Inference Service
  · POST /predict             ← returns fraud score + label
  · GET  /health
  · GET  /model/info          ← active model version + metrics
        │
        ▼
  Docker → GCP Cloud Run      ← auto-scales to zero when idle
```

---

## Stack

| Layer | Technology |
|---|---|
| Dataset | IEEE-CIS Fraud Detection (Kaggle) |
| Models | XGBoost, LightGBM |
| Experiment tracking | MLflow |
| API framework | FastAPI |
| Data validation | Pydantic v2 |
| Containerization | Docker |
| Deployment | GCP Cloud Run |
| CI/CD | GitHub Actions |
| Language | Python 3.11 |

---

## Project structure

```
fraud-detection-api/
├── data/
│   ├── raw/                 # Kaggle CSVs (gitignored)
│   └── processed/           # Feature-engineered parquet files
├── notebooks/
│   ├── eda.ipynb            # Exploratory data analysis
│   └── feature_eng.ipynb    # Feature development scratch space
├── src/
│   ├── train/
│   │   ├── train.py         # MLflow experiment runs - XGB + LGBM
│   │   ├── features.py      # Feature engineering pipeline
│   │   └── evaluate.py      # AUC-PR, threshold tuning, SHAP
│   ├── api/
│   │   ├── main.py          # FastAPI application
│   │   ├── schemas.py       # Pydantic request/response models
│   │   └── predictor.py     # Loads best MLflow model at startup
│   └── tests/
│       ├── test_api.py      # Endpoint integration tests
│       └── test_predict.py  # Prediction unit tests
├── mlruns/                  # MLflow local tracking store
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .github/
│   └── workflows/
│       └── deploy.yml       # Build → push → deploy to Cloud Run
└── README.md
```

---

## API reference

### `POST /predict`

```json
// Request
{
  "transaction_amt": 142.50,
  "product_cd": "W",
  "card_type": "credit",
  "addr1": 315,
  "dist1": 19.0,
  "p_emaildomain": "gmail.com",
  "r_emaildomain": "gmail.com"
}

// Response
{
  "fraud_score": 0.83,
  "is_fraud": true,
  "model_version": "xgboost-v3",
  "latency_ms": 12.4
}
```

### `GET /model/info`

```json
{
  "model_name": "fraud-detector",
  "version": "3",
  "algorithm": "XGBoost",
  "auc_pr": 0.91,
  "auc_roc": 0.93,
  "registered_at": "2026-03-28T10:00:00Z"
}
```

---

## Experiment tracking

Both XGBoost and LightGBM are evaluated across multiple hyperparameter configurations. All runs are tracked in MLflow:

| Logged artifact | Description |
|---|---|
| `auc_roc` | Primary ranking metric |
| `auc_pr` | Primary metric for imbalanced data |
| `f1_score` | At optimal threshold |
| `feature_importance` | SHAP values plot |
| `confusion_matrix` | At 0.5 and optimal threshold |
| `model` | Serialized model artifact |

The best run by AUC-PR is promoted to the `production` stage in the MLflow model registry. `predictor.py` loads this model at API startup - no hardcoded paths.

---

## Quickstart

> **Prerequisites:** Docker Desktop, Python 3.11+, Kaggle account (free dataset download)

```bash
# 1. Clone
git clone https://github.com/atharvahirulkar/fraud-detection-api.git
cd fraud-detection-api

# 2. Download dataset
# Place train_transaction.csv and train_identity.csv in data/raw/

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run training experiments
python src/train/train.py
# Opens MLflow UI: mlflow ui --port 5000

# 5. Start the API locally
uvicorn src.api.main:app --reload --port 8000

# 6. Or run fully containerized
docker compose up
```

---

## Performance targets

| Metric | Target |
|---|---|
| AUC-PR (imbalanced test set) | > 0.88 |
| AUC-ROC | > 0.92 |
| p50 inference latency | < 20ms |
| p99 inference latency | < 50ms |
| Cold start (Cloud Run) | < 3s |

*Benchmarks will be updated as the project is completed.*

---

## Status

- [x] Project structure + Docker Compose
- [x] README + API contract defined
- [ ] EDA notebook
- [ ] Feature engineering pipeline
- [ ] MLflow experiment runs - XGBoost + LightGBM
- [ ] FastAPI service + Pydantic schemas
- [ ] Model registry integration
- [ ] Docker image + GCP Cloud Run deploy
- [ ] GitHub Actions CI/CD
- [ ] Latency benchmarks

---

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) - 590K transactions, ~3.5% fraud rate. A heavily imbalanced binary classification problem, which makes AUC-PR a more meaningful metric than AUC-ROC for this task.

---

## Author

**Atharva Hirulkar** - MS Data Science, UC San Diego  
[GitHub](https://github.com/atharvahirulkar) · [LinkedIn](https://linkedin.com/in/atharva-hirulkar)
