# FraudLens

> Production ML system for real-time fraud detection with explainable AI.  
> IEEE-CIS · XGBoost · LightGBM · MLflow · FastAPI · AWS ECS Fargate · Qdrant · Airflow

![Status](https://img.shields.io/badge/status-live-brightgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)


Deployment status: API paused to manage AWS costs. Full infrastructure implemented and documented in infra/ and .github/workflows/. To run locally see Quickstart below.

---

## What this is

FraudLens is a production-grade ML system that trains, tracks, and serves a fraud detection model - and explains every prediction using a RAG-powered explanation engine. XGBoost and LightGBM are compared across 20+ MLflow experiment runs on the IEEE-CIS Fraud Detection dataset. The best model by AUC-PR is registered in the MLflow model registry, containerized, and deployed to AWS ECS Fargate behind an Application Load Balancer. A Qdrant-backed RAG layer retrieves fraud pattern context and generates natural language explanations for each flagged transaction using GPT-4o-mini. A daily Airflow DAG scores new batches and writes results to S3 + Athena.

The focus is the full production loop: reproducible training, experiment tracking, cloud deployment, explainability, and orchestrated batch inference.

---

## Architecture

```
IEEE-CIS Dataset (Kaggle)
        │
        ▼
Feature Engineering               ← Pandas, Scikit-learn
· Transaction aggregates
· Temporal + behavioral features
· Target encoding + imputation
        │
        ▼
MLflow Experiment Tracking
· XGBoost runs  (N hyperparameter configs)
· LightGBM runs (N hyperparameter configs)
· Logged: AUC-ROC, AUC-PR, F1, SHAP values
        │
        ▼
MLflow Model Registry             ← best AUC-PR run → "production"
        │
        ▼
FastAPI Inference Service
· POST /predict                   ← fraud score + label + explanation
· POST /explain                   ← SHAP + RAG explanation only
· GET  /health
· GET  /model/info
        │
        ├── AWS ECS Fargate + ALB ← containerized, auto-scales
        │
        ▼
Qdrant Vector Store               ← fraud pattern knowledge base
· 384-dim embeddings (all-MiniLM-L6-v2)
· Retrieves top-K relevant fraud patterns
· LLM (Claude Haiku / Ollama Mistral) generates explanation
        │
        ▼
Airflow DAG (daily batch)
· Pulls new transactions from S3
· Scores via inference API
· Writes results → S3 + Athena
        │
        ▼
Athena SQL queries over S3        ← fraud trend analysis, reporting
```

---

## Stack

| Layer | Technology |
|---|---|
| Dataset | IEEE-CIS Fraud Detection (Kaggle) |
| Models | XGBoost, LightGBM |
| Experiment tracking | MLflow |
| API framework | FastAPI + Uvicorn |
| Data validation | Pydantic v2 |
| Containerization | Docker |
| Cloud deployment | AWS ECS Fargate + ALB |
| Batch storage | AWS S3 + Athena |
| Vector store | Qdrant |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM (explanation) | OpenAI GPT-4o-mini (prod) / Ollama Mistral (local dev) |
| Orchestration | Apache Airflow |
| CI/CD | GitHub Actions |
| Language | Python 3.11 |

---

## Project Structure

```
fraudlens/
├── data/
│   ├── raw/                     # Kaggle CSVs (gitignored)
│   ├── processed/               # Feature-engineered parquet files
│   └── fraud_patterns/          # RAG knowledge base docs (JSON)
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis (see GitHub)
│   └── 02_feature_eng.ipynb     # Feature development
├── src/
│   ├── train/
│   │   ├── train.py             # MLflow experiment runs - XGB + LGBM
│   │   ├── features.py          # Feature engineering pipeline
│   │   └── evaluate.py          # AUC-PR, threshold tuning, SHAP
│   ├── api/
│   │   ├── main.py              # FastAPI application
│   │   ├── schemas.py           # Pydantic request/response models
│   │   ├── predictor.py         # Loads best MLflow model at startup
│   │   └── explainer.py         # SHAP + RAG explanation engine
│   ├── rag/
│   │   ├── ingest.py            # Embed + index fraud patterns → Qdrant
│   │   ├── retriever.py         # Top-K pattern retrieval
│   │   └── generator.py         # LLM explanation generation
│   └── tests/
│       ├── test_api.py          # Endpoint integration tests
│       └── test_predict.py      # Prediction unit tests
├── dags/
│   └── daily_scoring.py         # Airflow DAG: S3 → score → S3/Athena
├── mlruns/                      # MLflow local tracking store
├── infra/
│   ├── setup-ecs.sh             # One-time AWS ECS Fargate bootstrap
│   └── apprunner.yaml           # (legacy - replaced by ECS)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .github/
│   └── workflows/
│       └── deploy.yml           # Build → ECR → ECS Fargate deploy
└── README.md
```

---

## API Reference

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
  "r_emaildomain": "anonymous.com"
}

// Response
{
  "fraud_score": 0.83,
  "is_fraud": true,
  "model_version": "lightgbm-v2",
  "latency_ms": 14.2,
  "explanation": "This transaction was flagged primarily due to a mismatched recipient email domain (anonymous.com) combined with an above-average transaction amount. These features match patterns associated with card-not-present fraud where the billing and shipping email domains differ. SHAP top drivers: r_emaildomain (+0.31), transaction_amt (+0.18), dist1 (+0.09)."
}
```

### `POST /explain`

```json
// Request - same schema as /predict
// Response
{
  "shap_values": {
    "r_emaildomain": 0.31,
    "transaction_amt": 0.18,
    "dist1": 0.09
  },
  "retrieved_patterns": [
    "card-not-present email mismatch",
    "high-value CNP transaction"
  ],
  "explanation": "..."
}
```

### `GET /model/info`

```json
{
  "model_name": "fraudlens-detector",
  "version": "2",
  "algorithm": "LightGBM",
  "auc_pr": 0.91,
  "auc_roc": 0.93,
  "registered_at": "2026-05-15T10:00:00Z"
}
```

---

## RAG Explanation System

The explanation layer retrieves relevant fraud pattern context before generating a natural language explanation:

```
Incoming transaction features + SHAP values
        │
        ▼
Embed top SHAP features → query vector (384-dim)
        │
        ▼
Qdrant similarity search → top-3 fraud pattern docs
(e.g. "card-not-present email mismatch", "velocity abuse")
        │
        ▼
LLM prompt:
  · Transaction summary
  · SHAP top drivers
  · Retrieved pattern context
        │
        ▼
Natural language explanation (2–3 sentences)
```

**Fraud pattern knowledge base** (`data/fraud_patterns/`): ~50 curated JSON documents describing common IEEE-CIS fraud patterns, each with feature signatures, typical SHAP driver profiles, and plain-English descriptions. Embedded at startup and indexed in Qdrant.

**LLM config:**
- Dev: Ollama (Mistral 7B, fully local, zero cost)
- Prod: OpenAI GPT-4o-mini (low latency, low cost per call)

---

## Airflow Batch Scoring DAG

`dags/daily_scoring.py` runs nightly at 2 AM UTC:

```
S3: s3://fraudlens/incoming/YYYY-MM-DD/
        │ download new transactions
        ▼
Score via inference API (POST /predict)
        │
        ▼
S3: s3://fraudlens/scored/YYYY-MM-DD/
        │ Athena table: fraudlens_scored
        ▼
Athena: daily fraud rate · top flagged merchants · score distribution
```

Athena queries run directly on Parquet files in S3 - no separate database to manage.

---

## Experiment Tracking

Both models are evaluated across multiple hyperparameter configurations in MLflow:

| Logged artifact | Description |
|---|---|
| `auc_roc` | Standard ranking metric |
| `auc_pr` | Primary metric (imbalanced data) |
| `f1_score` | At optimal threshold |
| `feature_importance` | SHAP summary plot |
| `confusion_matrix` | At 0.5 + optimal threshold |
| `model` | Serialized artifact |

Best run by AUC-PR is promoted to `production` in the MLflow model registry. `predictor.py` loads this at API startup - no hardcoded paths.

---

## Cloud Deployment (AWS)

**Live API:** `http://fraudlens-alb-1532793415.us-east-1.elb.amazonaws.com`

### ECS Fargate + ALB (inference API)

Infrastructure is bootstrapped once with `infra/setup-ecs.sh`. After that, every push to `main` triggers GitHub Actions which:
1. Syncs model artifacts from S3 (`mlruns/`)
2. Builds a `linux/amd64` Docker image and pushes to ECR
3. Registers a new ECS task definition revision with the new image digest
4. Rolls out to the ECS Fargate service (zero-downtime rolling deploy)
5. Waits for service stability and smoke-tests `/health`

```bash
# One-time bootstrap (creates ALB, ECS cluster, IAM roles, task def, service)
GITHUB_ORG=atharvahirulkar GITHUB_REPO=fraudlens bash infra/setup-ecs.sh

# Redeploy: just push to main
git push origin main

# Get live URL
aws elbv2 describe-load-balancers --names fraudlens-alb \
  --query "LoadBalancers[0].DNSName" --output text --region us-east-1
```

### AWS S3 + Athena (batch results)

```bash
# Create bucket
aws s3 mb s3://fraudlens

# Create Athena table over scored parquet files
CREATE EXTERNAL TABLE fraudlens_scored (
  transaction_id STRING,
  fraud_score    DOUBLE,
  is_fraud       BOOLEAN,
  scored_at      TIMESTAMP
)
STORED AS PARQUET
LOCATION 's3://fraudlens/scored/';
```

---

## Quickstart

> **Prerequisites:** Docker Desktop · Python 3.11+ · Kaggle account (free) · AWS account (AWS Educate - no credit card)

```bash
# 1. Clone and install
git clone https://github.com/atharvahirulkar/fraudlens.git
cd fraudlens
pip install -r requirements.txt

# 2. Download dataset (place in data/raw/)
#    train_transaction.csv + train_identity.csv
#    https://www.kaggle.com/c/ieee-fraud-detection/data

# 3. Run training experiments
python src/train/train.py
mlflow ui --port 5000
# Open http://localhost:5000 - compare XGBoost vs LightGBM runs

# 4. Start Qdrant and ingest fraud patterns
docker run -d -p 6333:6333 qdrant/qdrant
python src/rag/ingest.py

# 5. Start inference API locally
uvicorn src.api.main:app --reload --port 8000

# 6. Run full stack via Docker Compose
docker compose up
# API:     http://localhost:8000
# MLflow:  http://localhost:5000
# Qdrant:  http://localhost:6333/dashboard

# 7. Test prediction + explanation
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amt": 142.50,
    "product_cd": "W",
    "card_type": "credit",
    "addr1": 315,
    "dist1": 19.0,
    "p_emaildomain": "gmail.com",
    "r_emaildomain": "anonymous.com"
  }'

# 8. Health check
curl http://localhost:8000/health
```

---

## Performance Targets

| Metric | Target |
|---|---|
| AUC-PR (imbalanced test set) | > 0.88 |
| AUC-ROC | > 0.92 |
| p50 inference latency (local) | < 20ms |
| p99 inference latency (ECS Fargate) | < 80ms |
| Explanation generation latency | < 2s |
| Airflow DAG daily run time | < 5 min |

*Benchmarks measured on ECS Fargate (us-east-1, 1 vCPU / 2 GB).*

---

## Status

- [x] EDA notebook (`01_eda.ipynb`)
- [x] Feature engineering pipeline (`src/train/features.py`)
- [x] MLflow experiment runs - XGBoost + LightGBM (20+ configs)
- [x] Threshold tuning + SHAP (`src/train/evaluate.py`)
- [x] FastAPI service + Pydantic schemas
- [x] MLflow model registry integration (`src/api/predictor.py`)
- [x] Full feature pipeline applied at inference time (not just 7 raw fields)
- [x] Real SHAP values via `shap.TreeExplainer` on native model
- [x] Fraud pattern knowledge base (`data/fraud_patterns/`)
- [x] Qdrant ingestion pipeline (`src/rag/ingest.py`)
- [x] RAG retrieval + in-memory fallback (`src/rag/retriever.py`)
- [x] LLM explanation generator - GPT-4o-mini / Ollama (`src/rag/generator.py`)
- [x] SHAP + RAG explainer endpoint (`src/api/explainer.py`)
- [x] Docker image + docker-compose (Qdrant + MLflow + API)
- [x] AWS ECR + ECS Fargate + ALB deploy (`infra/setup-ecs.sh`)
- [x] S3 model artifact store + versioning
- [x] Airflow DAG - daily batch scoring → S3 + Athena (`dags/daily_scoring.py`)
- [x] GitHub Actions CI/CD - push to main auto-deploys to ECS (`deploy.yml`)

---

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) - 590K transactions, ~3.5% fraud rate. A heavily imbalanced binary classification problem, making AUC-PR the primary evaluation metric over AUC-ROC.

---

## Author

**Atharva Hirulkar** - MS Data Science, UC San Diego  
[GitHub](https://github.com/atharvahirulkar) · [LinkedIn](https://linkedin.com/in/atharva-hirulkar)
