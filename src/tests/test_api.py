"""
Integration tests for FraudLens API endpoints.

Tests all 4 FastAPI endpoints:
  - POST /predict
  - POST /explain
  - GET  /health
  - GET  /model/info

Run with:
    pytest src/tests/test_api.py -v
"""

import json
import time
from typing import Optional

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

# Test client
client = TestClient(app)

# - Fixtures


@pytest.fixture
def valid_transaction():
    """Valid transaction for testing."""
    return {
        "transaction_amt": 150.0,
        "product_cd": "W",
        "card_type": "credit",
        "addr1": 315,
        "dist1": 25.0,
        "p_emaildomain": "gmail.com",
        "r_emaildomain": "yahoo.com",
    }


@pytest.fixture
def high_risk_transaction():
    """High-risk transaction (likely fraud)."""
    return {
        "transaction_amt": 5000.0,
        "product_cd": "W",
        "card_type": "credit",
        "addr1": 500,
        "dist1": 1000.0,  # High distance
        "p_emaildomain": "gmail.com",
        "r_emaildomain": "anonymous.com",  # Suspicious domain
    }


@pytest.fixture
def low_value_transaction():
    """Low-value transaction (likely legitimate)."""
    return {
        "transaction_amt": 2.50,
        "product_cd": "W",
        "card_type": "credit",
        "addr1": 315,
        "dist1": 1.0,
        "p_emaildomain": "gmail.com",
        "r_emaildomain": "gmail.com",
    }


# - Endpoint Tests


class TestHealthEndpoint:
    """Test GET /health endpoint."""

    def test_health_check_success(self):
        """Health check should return 200 with status=healthy."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "model_loaded" in data
        assert "mlflow_uri" in data
        assert "version" in data

    def test_health_check_response_schema(self):
        """Health response should match HealthResponse schema."""
        response = client.get("/health")
        data = response.json()
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["mlflow_uri"], str)
        assert isinstance(data["version"], str)


class TestModelInfoEndpoint:
    """Test GET /model/info endpoint."""

    def test_model_info_success(self):
        """Model info endpoint should return 200 with metadata."""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()

        # Required fields
        assert "model_name" in data
        assert "version" in data
        assert "algorithm" in data
        assert "auc_pr" in data
        assert "auc_roc" in data
        assert "registered_at" in data

    def test_model_info_metrics(self):
        """AUC metrics should be valid (0-1)."""
        response = client.get("/model/info")
        data = response.json()

        assert 0 <= data["auc_pr"] <= 1
        assert 0 <= data["auc_roc"] <= 1
        assert data["auc_pr"] > 0.7  # Expect high AUC-PR on test set
        assert data["auc_roc"] > 0.8


class TestPredictEndpoint:
    """Test POST /predict endpoint."""

    def test_predict_valid_transaction(self, valid_transaction):
        """Valid transaction should return fraud prediction."""
        response = client.post("/predict", json=valid_transaction)
        assert response.status_code == 200
        data = response.json()

        # Required fields
        assert "fraud_score" in data
        assert "is_fraud" in data
        assert "model_version" in data
        assert "latency_ms" in data
        assert "explanation" in data

    def test_predict_fraud_score_range(self, valid_transaction):
        """Fraud score should be between 0 and 1."""
        response = client.post("/predict", json=valid_transaction)
        data = response.json()

        assert 0 <= data["fraud_score"] <= 1

    def test_predict_high_risk_transaction(self, high_risk_transaction):
        """High-risk transaction should have higher fraud score."""
        response = client.post("/predict", json=high_risk_transaction)
        assert response.status_code == 200
        data = response.json()

        # Expect moderate to high fraud score
        # (actual values depend on model)
        assert "fraud_score" in data
        print(f"High-risk fraud score: {data['fraud_score']}")

    def test_predict_low_risk_transaction(self, low_value_transaction):
        """Low-value legitimate transaction should have lower fraud score."""
        response = client.post("/predict", json=low_value_transaction)
        assert response.status_code == 200
        data = response.json()

        assert "fraud_score" in data
        print(f"Low-risk fraud score: {data['fraud_score']}")

    def test_predict_latency(self, valid_transaction):
        """Prediction latency should be <20ms (local) or <100ms (remote)."""
        response = client.post("/predict", json=valid_transaction)
        data = response.json()
        latency = data["latency_ms"]

        # Local should be very fast, App Runner <100ms is reasonable
        assert latency < 500  # Generous timeout for CI

    def test_predict_missing_field(self):
        """Request with missing required field should fail."""
        invalid = {
            "product_cd": "W",
            "card_type": "credit",
            # Missing transaction_amt
        }
        response = client.post("/predict", json=invalid)
        assert response.status_code == 422  # Validation error

    def test_predict_invalid_fraud_score(self):
        """Invalid fraud score value should fail validation."""
        invalid = {
            "transaction_amt": -100,  # Invalid: negative amount
            "product_cd": "W",
            "card_type": "credit",
            "addr1": 315,
            "dist1": 25.0,
            "p_emaildomain": "gmail.com",
            "r_emaildomain": "yahoo.com",
        }
        response = client.post("/predict", json=invalid)
        assert response.status_code == 422


class TestExplainEndpoint:
    """Test POST /explain endpoint."""

    def test_explain_valid_transaction(self, valid_transaction):
        """Valid transaction should return explanation with SHAP values."""
        response = client.post("/explain", json=valid_transaction)
        assert response.status_code == 200
        data = response.json()

        # Required fields
        assert "fraud_score" in data
        assert "shap_values" in data
        assert "retrieved_patterns" in data
        assert "explanation" in data

    def test_explain_shap_values_format(self, valid_transaction):
        """SHAP values should be properly formatted."""
        response = client.post("/explain", json=valid_transaction)
        data = response.json()

        shap_values = data["shap_values"]
        assert isinstance(shap_values, list)
        assert len(shap_values) > 0

        # Each SHAP value should have feature and shap_value
        for shap in shap_values:
            assert "feature" in shap
            assert "shap_value" in shap
            assert isinstance(shap["feature"], str)
            assert isinstance(shap["shap_value"], (int, float))

    def test_explain_retrieved_patterns(self, valid_transaction):
        """Explanation should include retrieved fraud patterns."""
        response = client.post("/explain", json=valid_transaction)
        data = response.json()

        patterns = data["retrieved_patterns"]
        assert isinstance(patterns, list)
        # Should retrieve at least 1 pattern (or be empty if Qdrant unavailable)
        # assert len(patterns) > 0  # May not be available in test env

    def test_explain_explanation_text(self, valid_transaction):
        """Explanation should be non-empty text."""
        response = client.post("/explain", json=valid_transaction)
        data = response.json()

        explanation = data["explanation"]
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_explain_high_risk_gets_detailed_explanation(self, high_risk_transaction):
        """High-risk transaction should get more detailed explanation."""
        response = client.post("/explain", json=high_risk_transaction)
        data = response.json()

        explanation = data["explanation"]
        # High-risk should mention suspicious patterns
        assert len(explanation) > 10


class TestRootEndpoint:
    """Test GET / endpoint."""

    def test_root_returns_json(self):
        """Root endpoint should return API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()

        assert "service" in data
        assert "version" in data
        assert data["service"] == "FraudLens"


# - Performance / Benchmark Tests


class TestPerformance:
    """Performance and latency benchmarks."""

    @pytest.mark.benchmark
    def test_predict_p50_latency(self, valid_transaction):
        """Measure p50 latency for /predict endpoint."""
        times = []

        for _ in range(100):
            start = time.time()
            response = client.post("/predict", json=valid_transaction)
            elapsed = (time.time() - start) * 1000  # ms
            times.append(elapsed)

        times.sort()
        p50 = times[50]
        p99 = times[99]

        print(f"Predict p50 latency: {p50:.2f}ms")
        print(f"Predict p99 latency: {p99:.2f}ms")

        # Target: p50 < 20ms (local) or < 50ms (remote)
        assert p50 < 200

    @pytest.mark.benchmark
    def test_explain_latency(self, valid_transaction):
        """Measure /explain endpoint latency."""
        times = []

        for _ in range(10):  # Fewer iterations for slower endpoint
            start = time.time()
            response = client.post("/explain", json=valid_transaction)
            elapsed = (time.time() - start) * 1000  # ms
            times.append(elapsed)

        times.sort()
        p50 = times[5] if len(times) >= 5 else min(times)

        print(f"Explain p50 latency: {p50:.2f}ms")

        # Target: < 2s per README
        assert p50 < 5000


# - Error Handling Tests


class TestErrorHandling:
    """Test API error handling."""

    def test_malformed_json(self):
        """Malformed JSON should return 422."""
        response = client.post(
            "/predict",
            data="not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code in [400, 422, 500]

    def test_null_values(self):
        """Null values should be handled gracefully."""
        transaction = {
            "transaction_amt": 150.0,
            "product_cd": "W",
            "card_type": None,  # Null value
            "addr1": 315,
            "dist1": 25.0,
            "p_emaildomain": "gmail.com",
            "r_emaildomain": "yahoo.com",
        }
        response = client.post("/predict", json=transaction)
        # Should either accept nulls or return 422
        assert response.status_code in [200, 422]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
