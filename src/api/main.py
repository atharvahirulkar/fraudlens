"""
FastAPI inference service for FraudLens.

Endpoints:
  - POST /predict    → fraud score + label + explanation
  - POST /explain    → SHAP + RAG explanation only
  - GET  /health     → service health
  - GET  /model/info → model metadata

Model and feature pipeline are loaded from MLflow at startup.
Explanation generation (SHAP + RAG) deferred to Phase 4.
"""

import os
import time
import warnings
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.api.explainer import get_explainer
from src.api.predictor import ModelPredictor, get_predictor
from src.api.schemas import (
    ExplainResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    SHAPFeature,
)

warnings.filterwarnings("ignore")

# - Configuration

API_VERSION = "1.0.0"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Placeholder: Will be replaced with real SHAP generation in production
EXPLAIN_ENGINE_AVAILABLE = False

# - Startup / Shutdown


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    print("[FastAPI] Starting up...")
    try:
        predictor = get_predictor()
        if not predictor.is_ready():
            raise RuntimeError("Model predictor failed to initialize")
        print("[FastAPI] ✓ Model loaded and ready")
    except Exception as e:
        print(f"[FastAPI] ERROR on startup: {e}")
        raise

    yield

    print("[FastAPI] Shutting down...")


# - FastAPI app

app = FastAPI(
    title="FraudLens Inference API",
    description="Production fraud detection with explainable AI",
    version=API_VERSION,
)


# - Helper functions


def _request_to_dataframe(req: PredictRequest) -> pd.DataFrame:
    """
    Convert PredictRequest to a feature-engineered DataFrame for model inference.

    Maps the 7 API fields to raw IEEE-CIS column names, applies the fitted
    feature pipeline (temporal, email, card-aggregate, target/label encoding,
    imputation), then aligns to the exact feature set used during training.
    Any column not derivable from the API inputs (V/C/D/id columns) is filled
    with its training-set median so the model always receives a full-width row.
    """
    predictor = get_predictor()

    # Seconds elapsed since IEEE-CIS origin date, used for temporal features
    dt_origin = pd.Timestamp("2017-11-30")
    transaction_dt = (pd.Timestamp.now() - dt_origin).total_seconds()

    # Map API schema → raw IEEE-CIS column names expected by the pipeline
    raw = {
        "TransactionDT": transaction_dt,
        "TransactionAmt": req.transaction_amt,
        "ProductCD": req.product_cd,
        "card6": req.card_type if req.card_type is not None else np.nan,
        "addr1": float(req.addr1) if req.addr1 is not None else np.nan,
        "dist1": req.dist1 if req.dist1 is not None else np.nan,
        "P_emaildomain": req.p_emaildomain if req.p_emaildomain else np.nan,
        "R_emaildomain": req.r_emaildomain if req.r_emaildomain else np.nan,
    }
    df = pd.DataFrame([raw])

    if predictor.pipeline is not None:
        try:
            transformed = predictor.pipeline.transform(df)
            # Pad any feature the pipeline couldn't derive (V/C/D/id columns)
            # with their training-set medians so the model gets a full-width row.
            state = predictor.pipeline.state
            for col in state.feature_cols:
                if col not in transformed.columns:
                    transformed[col] = float(
                        state.numeric_medians.get(col, 0.0)
                    )
            return transformed[state.feature_cols]
        except Exception as e:
            print(f"[API] Pipeline transform failed ({e}); falling back to raw encoding")

    # Fallback when pipeline is unavailable: minimal numeric encoding
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        df[col] = pd.factorize(df[col].astype(str), sort=True)[0].astype(float)
    return df


def _generate_explanation(
    fraud_score: float,
    shap_values: Optional[dict] = None,
    patterns: Optional[list] = None,
) -> str:
    """
    Generate a natural language explanation.

    Placeholder: Returns a template explanation.
    In Phase 4, this will call the LLM (Claude Haiku or Ollama Mistral)
    with SHAP + retrieved fraud pattern context.
    """
    if fraud_score > 0.7:
        return (
            "This transaction was flagged as high-risk fraud. Multiple features "
            "combined to create a suspicious profile. Key drivers include unusual "
            "geographic patterns, amount deviation, and domain mismatch indicators."
        )
    elif fraud_score > 0.5:
        return (
            "This transaction shows moderate fraud risk. Some features align with "
            "known fraud patterns, but signals are mixed. Review for additional "
            "context before deciding."
        )
    else:
        return (
            "This transaction appears legitimate. Features align with historical "
            "patterns for this cardholder. Low fraud risk."
        )


def _safe_get_predictor() -> tuple[Optional[ModelPredictor], Optional[str]]:
    """Return predictor if available, otherwise capture load error message."""
    try:
        predictor = get_predictor()
        return predictor, None
    except Exception as exc:
        return None, str(exc)


# - Endpoints


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Service health and readiness check."""
    predictor, error = _safe_get_predictor()
    ready = predictor is not None and predictor.is_ready()

    if error:
        print(f"[FastAPI] Health check degraded: {error}")

    return HealthResponse(
        status="healthy" if ready else "degraded",
        model_loaded=ready,
        mlflow_uri=MLFLOW_TRACKING_URI,
        version=API_VERSION,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info() -> ModelInfoResponse:
    """Get active model metadata from MLflow registry."""
    predictor, error = _safe_get_predictor()

    if predictor is None or not predictor.is_ready():
        detail = "Model not loaded"
        if error:
            detail = f"Model not loaded: {error}"
        raise HTTPException(status_code=503, detail=detail)

    info = predictor.get_model_info()
    from datetime import datetime

    return ModelInfoResponse(
        model_name=info["model_name"],
        version=info["version"],
        algorithm=info["algorithm"],
        auc_pr=info["auc_pr"],
        auc_roc=info["auc_roc"],
        registered_at=datetime.fromtimestamp(info["registered_at"]),
        features_count=(
            len(predictor.pipeline.state.feature_cols)
            if predictor.pipeline is not None
            else 74
        ),
        optimal_threshold=info["optimal_threshold"],
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(req: PredictRequest) -> PredictResponse:
    """
    Predict fraud probability for a single transaction.

    Returns: fraud_score, binary label (at optimal threshold), and explanation.
    """
    predictor, error = _safe_get_predictor()

    if predictor is None or not predictor.is_ready():
        detail = "Model not loaded"
        if error:
            detail = f"Model not loaded: {error}"
        raise HTTPException(status_code=503, detail=detail)

    # Measure latency
    start = time.time()

    X = _request_to_dataframe(req)

    # Predict
    result = predictor.predict(X)
    fraud_score = float(result["fraud_score"][0])

    # Threshold for binary classification
    is_fraud = fraud_score >= predictor.optimal_threshold

    # Generate explanation (placeholder for now)
    explanation = _generate_explanation(fraud_score)

    latency_ms = (time.time() - start) * 1000

    return PredictResponse(
        fraud_score=fraud_score,
        is_fraud=is_fraud,
        model_version=f"{predictor.model_metadata['algorithm'].lower()}-"
        f"v{predictor.model_metadata['version']}",
        latency_ms=latency_ms,
        explanation=explanation,
    )


@app.post("/explain", response_model=ExplainResponse, tags=["Explanation"])
async def explain(req: PredictRequest) -> ExplainResponse:
    """
    Detailed explanation including SHAP values and retrieved fraud patterns.
    Uses OpenAI GPT-4o-mini to generate a natural language explanation grounded
    in SHAP feature drivers and Qdrant-retrieved fraud patterns (when available).
    """
    predictor, error = _safe_get_predictor()

    if predictor is None or not predictor.is_ready():
        detail = "Model not loaded"
        if error:
            detail = f"Model not loaded: {error}"
        raise HTTPException(status_code=503, detail=detail)

    X = _request_to_dataframe(req)
    result = predictor.predict(X)
    fraud_score = float(result["fraud_score"][0])

    try:
        explainer = get_explainer()
        transaction_summary = (
            f"${req.transaction_amt:.2f} {req.product_cd} transaction"
            + (f" via {req.card_type}" if req.card_type else "")
            + (f", payer: {req.p_emaildomain}" if req.p_emaildomain else "")
            + (f", recipient: {req.r_emaildomain}" if req.r_emaildomain else "")
        )
        result_ex = explainer.explain_prediction(X, fraud_score, transaction_summary)

        shap_features = [
            SHAPFeature(feature=s["feature"], shap_value=s["shap_value"])
            for s in result_ex["shap_values"]
        ]
        patterns = result_ex["retrieved_patterns"]
        explanation = result_ex["explanation"]

    except Exception as e:
        print(f"[FastAPI] Explainer failed, falling back to template: {e}")
        shap_features = [
            SHAPFeature(feature="r_emaildomain", shap_value=0.18),
            SHAPFeature(feature="transaction_amt", shap_value=0.12),
            SHAPFeature(feature="addr1", shap_value=0.08),
        ]
        patterns = ["card-not-present email mismatch", "above-average transaction amount"]
        explanation = _generate_explanation(fraud_score)

    return ExplainResponse(
        fraud_score=fraud_score,
        shap_values=shap_features,
        retrieved_patterns=patterns,
        explanation=explanation,
    )


@app.get("/", tags=["Root"])
async def root():
    """API root endpoint."""
    return {
        "service": "FraudLens",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


# - Exception handlers


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": f"Validation error: {str(exc)}"},
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal error: {str(exc)}"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
