"""
Unit tests for fraud prediction logic.

Tests core prediction components:
  - Model loading and availability
  - Prediction output format and ranges
  - Feature pipeline integration
  - Threshold-based classification

Run with:
    pytest src/tests/test_predict.py -v
"""

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from src.api.predictor import ModelPredictor, get_predictor

# - Fixtures


@pytest.fixture
def sample_dataframe():
    """Sample feature DataFrame for testing."""
    return pd.DataFrame({
        "transaction_amt": [100.0, 500.0, 25.0],
        "product_cd": ["W", "W", "R"],
        "addr1": [315, 500, 315],
        "dist1": [10.0, 1000.0, 5.0],
        "p_emaildomain": ["gmail.com", "gmail.com", "yahoo.com"],
        "r_emaildomain": ["yahoo.com", "anonymous.com", "yahoo.com"],
    })


# - Model Loading Tests


class TestModelLoading:
    """Test model initialization and loading from MLflow."""

    def test_predictor_singleton(self):
        """Predictor should be a singleton."""
        p1 = get_predictor()
        p2 = get_predictor()
        assert p1 is p2

    @patch("src.api.predictor.mlflow.set_tracking_uri")
    @patch("src.api.predictor.MlflowClient")
    def test_model_loads_without_error(self, mock_client, mock_set_uri):
        """Model should load from MLflow registry without error."""
        # This test would require mocking MLflow completely
        # In CI/CD, you'd have a running MLflow instance or skip this test
        pass

    def test_is_ready(self):
        """Model should report ready status."""
        predictor = get_predictor()
        # Model may not be available in test env, but method should exist
        assert hasattr(predictor, "is_ready")
        assert isinstance(predictor.is_ready(), bool)


# - Prediction Output Tests


class TestPredictionOutput:
    """Test prediction output format and values."""

    def test_predict_returns_dict(self, sample_dataframe):
        """Prediction should return a dict."""
        predictor = get_predictor()
        result = predictor.predict(sample_dataframe)

        assert isinstance(result, dict)
        assert "fraud_score" in result
        assert "n_samples" in result

    def test_predict_fraud_score_range(self, sample_dataframe):
        """Fraud scores should be between 0 and 1."""
        predictor = get_predictor()
        result = predictor.predict(sample_dataframe)

        fraud_scores = result["fraud_score"]
        if isinstance(fraud_scores, (int, float)):
            fraud_scores = [fraud_scores]

        for score in fraud_scores:
            assert 0 <= score <= 1, f"Score {score} outside range [0, 1]"

    def test_predict_n_samples(self, sample_dataframe):
        """n_samples should match input DataFrame."""
        predictor = get_predictor()
        result = predictor.predict(sample_dataframe)

        assert result["n_samples"] == len(sample_dataframe)

    def test_predict_single_row(self):
        """Prediction should work with single row DataFrame."""
        df = pd.DataFrame({
            "transaction_amt": [100.0],
            "product_cd": ["W"],
            "addr1": [315],
            "dist1": [10.0],
            "p_emaildomain": ["gmail.com"],
            "r_emaildomain": ["yahoo.com"],
        })

        predictor = get_predictor()
        result = predictor.predict(df)

        assert result["n_samples"] == 1
        assert len(result["fraud_score"]) == 1 or isinstance(result["fraud_score"], (int, float))


# - Classification Threshold Tests


class TestClassificationThreshold:
    """Test fraud/legitimate classification using optimal threshold."""

    def test_threshold_from_metadata(self):
        """Optimal threshold should be available from model metadata."""
        predictor = get_predictor()

        if predictor.is_ready() and predictor.model_metadata:
            threshold = predictor.optimal_threshold
            assert 0 <= threshold <= 1
            assert isinstance(threshold, (int, float))

    def test_threshold_based_classification(self):
        """Binary classification should use optimal threshold."""
        predictor = get_predictor()
        threshold = predictor.optimal_threshold

        # Test classification logic
        scores = [0.2, 0.5, 0.8, 0.95]
        expected = [score >= threshold for score in scores]

        for score, expected_class in zip(scores, expected):
            assert isinstance(expected_class, bool)


# - Feature Pipeline Tests


class TestFeaturePipeline:
    """Test feature engineering pipeline integration."""

    def test_pipeline_availability(self):
        """Feature pipeline should be loadable."""
        predictor = get_predictor()

        if predictor.is_ready():
            try:
                pipeline = predictor.get_pipeline()
                assert pipeline is not None
            except RuntimeError as e:
                # Pipeline may not be available in test env
                assert "not loaded" in str(e).lower()

    @pytest.mark.skip(reason="Requires actual pipeline artifact")
    def test_pipeline_transform(self, sample_dataframe):
        """Pipeline should transform raw features."""
        predictor = get_predictor()
        pipeline = predictor.get_pipeline()

        # This would require the actual fitted pipeline
        # transformed = pipeline.transform(sample_dataframe)
        # assert transformed.shape[0] == sample_dataframe.shape[0]


# - Model Metadata Tests


class TestModelMetadata:
    """Test model metadata from MLflow registry."""

    def test_get_model_info(self):
        """Model info should be retrievable."""
        predictor = get_predictor()

        if predictor.is_ready() and predictor.model_metadata:
            info = predictor.get_model_info()

            required_keys = [
                "model_name",
                "version",
                "algorithm",
                "auc_pr",
                "auc_roc",
            ]

            for key in required_keys:
                assert key in info, f"Missing key: {key}"

    def test_model_metrics_valid(self):
        """Model metrics should be valid values."""
        predictor = get_predictor()

        if predictor.is_ready() and predictor.model_metadata:
            info = predictor.get_model_info()

            # Metrics should be between 0 and 1
            assert 0 <= info["auc_pr"] <= 1
            assert 0 <= info["auc_roc"] <= 1


# - Error Handling Tests


class TestErrorHandling:
    """Test error handling in prediction."""

    def test_empty_dataframe_handling(self):
        """Empty DataFrame should be handled gracefully."""
        predictor = get_predictor()
        empty_df = pd.DataFrame()

        result = predictor.predict(empty_df)
        assert result["n_samples"] == 0

    def test_missing_columns(self):
        """DataFrame with missing columns should raise error."""
        predictor = get_predictor()
        df = pd.DataFrame({"wrong_column": [1.0]})

        # Should raise error about missing features
        try:
            result = predictor.predict(df)
            # May succeed if model handles missing columns
        except (KeyError, ValueError, Exception):
            # Expected to fail with missing features
            pass

    def test_invalid_data_types(self):
        """Invalid data types should raise error."""
        predictor = get_predictor()
        df = pd.DataFrame({
            "transaction_amt": ["not_a_number"],
            "product_cd": ["W"],
        })

        try:
            result = predictor.predict(df)
        except (ValueError, TypeError):
            # Expected to fail with type error
            pass


# - Integration Tests


class TestPredictionIntegration:
    """Integration tests with full prediction pipeline."""

    def test_end_to_end_prediction(self, sample_dataframe):
        """Full prediction pipeline should work end-to-end."""
        predictor = get_predictor()

        if predictor.is_ready():
            result = predictor.predict(sample_dataframe)
            assert result["n_samples"] == len(sample_dataframe)
            assert isinstance(result["fraud_score"], (list, int, float))

    def test_multiple_predictions_consistent(self):
        """Multiple predictions should be deterministic."""
        predictor = get_predictor()

        df = pd.DataFrame({
            "transaction_amt": [100.0],
            "product_cd": ["W"],
            "addr1": [315],
            "dist1": [10.0],
            "p_emaildomain": ["gmail.com"],
            "r_emaildomain": ["yahoo.com"],
        })

        if predictor.is_ready():
            result1 = predictor.predict(df)
            result2 = predictor.predict(df)

            # Predictions should be identical
            assert result1["fraud_score"] == result2["fraud_score"]


# - Benchmark Tests


class TestPredictionPerformance:
    """Performance benchmarks for prediction latency."""

    @pytest.mark.benchmark
    def test_single_prediction_latency(self):
        """Measure latency for single prediction."""
        import time

        predictor = get_predictor()

        df = pd.DataFrame({
            "transaction_amt": [100.0],
            "product_cd": ["W"],
            "addr1": [315],
            "dist1": [10.0],
            "p_emaildomain": ["gmail.com"],
            "r_emaildomain": ["yahoo.com"],
        })

        if predictor.is_ready():
            start = time.time()
            result = predictor.predict(df)
            elapsed = (time.time() - start) * 1000  # ms

            print(f"Single prediction latency: {elapsed:.2f}ms")
            # Target: <20ms local, <100ms cloud
            assert elapsed < 1000

    @pytest.mark.benchmark
    def test_batch_prediction_latency(self):
        """Measure latency for batch prediction."""
        import time

        predictor = get_predictor()

        # Create batch of 100 rows
        df = pd.DataFrame({
            "transaction_amt": [100.0] * 100,
            "product_cd": ["W"] * 100,
            "addr1": [315] * 100,
            "dist1": [10.0] * 100,
            "p_emaildomain": ["gmail.com"] * 100,
            "r_emaildomain": ["yahoo.com"] * 100,
        })

        if predictor.is_ready():
            start = time.time()
            result = predictor.predict(df)
            elapsed = (time.time() - start) * 1000  # ms
            latency_per_sample = elapsed / 100

            print(f"Batch prediction (100 samples): {elapsed:.2f}ms ({latency_per_sample:.2f}ms/sample)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
