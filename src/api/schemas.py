"""
Pydantic v2 request/response schemas for FastAPI.

Handles validation, JSON serialization, and documentation for all endpoints.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# - Request schemas

class PredictRequest(BaseModel):
    """Input schema for POST /predict and POST /explain."""

    # Transaction amount (in USD)
    transaction_amt: float = Field(..., gt=0, description="Transaction amount in USD")

    # Product code (W = card withdrawal, R = refund, etc.)
    product_cd: str = Field(..., min_length=1, max_length=1, description="Product code (e.g., W, R)")

    # Card type (credit, debit, etc.) — optional for flexibility
    card_type: Optional[str] = Field(
        None, max_length=20, description="Card type (e.g., credit, debit)"
    )

    # Address state (coded as numeric)
    addr1: Optional[int] = Field(None, ge=0, description="Address state code")

    # Distance between billing and shipping address (km)
    dist1: Optional[float] = Field(
        None, ge=0, description="Distance between addresses (km)"
    )

    # Payer email domain
    p_emaildomain: Optional[str] = Field(
        None, max_length=100, description="Payer email domain (e.g., gmail.com)"
    )

    # Recipient email domain
    r_emaildomain: Optional[str] = Field(
        None, max_length=100, description="Recipient email domain (e.g., yahoo.com)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_amt": 142.50,
                "product_cd": "W",
                "card_type": "credit",
                "addr1": 315,
                "dist1": 19.0,
                "p_emaildomain": "gmail.com",
                "r_emaildomain": "anonymous.com",
            }
        }
    }


# - Response schemas

class PredictResponse(BaseModel):
    """Output schema for POST /predict."""

    fraud_score: float = Field(..., ge=0, le=1, description="Fraud probability (0–1)")
    is_fraud: bool = Field(..., description="Binary fraud flag at optimal threshold")
    model_version: str = Field(..., description="Model identifier (e.g., lightgbm-v2)")
    latency_ms: float = Field(..., ge=0, description="Inference time in milliseconds")
    explanation: str = Field(..., description="Plain-English explanation grounded in SHAP + RAG")

    model_config = {
        "json_schema_extra": {
            "example": {
                "fraud_score": 0.83,
                "is_fraud": True,
                "model_version": "lightgbm-v2",
                "latency_ms": 14.2,
                "explanation": "This transaction was flagged primarily due to a mismatched recipient email domain (anonymous.com) combined with an above-average transaction amount. These features match patterns associated with card-not-present fraud where the billing and shipping email domains differ. SHAP top drivers: r_emaildomain (+0.31), transaction_amt (+0.18), dist1 (+0.09).",
            }
        }
    }


class SHAPFeature(BaseModel):
    """Single feature and its SHAP value."""

    feature: str = Field(..., description="Feature name")
    shap_value: float = Field(..., description="SHAP value (positive/negative impact)")


class ExplainResponse(BaseModel):
    """Output schema for POST /explain."""

    fraud_score: float = Field(..., ge=0, le=1, description="Fraud probability (0–1)")
    shap_values: list[SHAPFeature] = Field(
        ..., description="Top SHAP features ranked by absolute impact"
    )
    retrieved_patterns: list[str] = Field(
        ..., description="Fraud pattern context retrieved from RAG"
    )
    explanation: str = Field(
        ..., description="Natural language explanation from LLM"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "fraud_score": 0.83,
                "shap_values": [
                    {"feature": "r_emaildomain", "shap_value": 0.31},
                    {"feature": "transaction_amt", "shap_value": 0.18},
                    {"feature": "dist1", "shap_value": 0.09},
                ],
                "retrieved_patterns": [
                    "card-not-present email mismatch",
                    "high-value CNP transaction",
                ],
                "explanation": "...",
            }
        }
    }


class ModelInfoResponse(BaseModel):
    """Output schema for GET /model/info."""

    model_name: str = Field(..., description="Model registry name (e.g., fraudlens-detector)")
    version: str = Field(..., description="Model version in registry")
    algorithm: str = Field(..., description="Algorithm used (XGBoost or LightGBM)")
    auc_pr: float = Field(..., ge=0, le=1, description="Area Under Precision-Recall curve")
    auc_roc: float = Field(..., ge=0, le=1, description="Area Under ROC curve")
    registered_at: datetime = Field(..., description="Timestamp when registered in MLflow")
    features_count: int = Field(..., ge=1, description="Number of input features")
    optimal_threshold: float = Field(
        ..., ge=0, le=1, description="Optimal classification threshold"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_name": "fraudlens-detector",
                "version": "2",
                "algorithm": "LightGBM",
                "auc_pr": 0.91,
                "auc_roc": 0.93,
                "registered_at": "2026-05-15T10:00:00Z",
                "features_count": 74,
                "optimal_threshold": 0.42,
            }
        }
    }


class HealthResponse(BaseModel):
    """Output schema for GET /health."""

    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether the model has been loaded")
    mlflow_uri: str = Field(..., description="MLflow tracking server URI")
    version: str = Field(..., description="API version")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "mlflow_uri": "http://localhost:5000",
                "version": "1.0.0",
            }
        }
    }
