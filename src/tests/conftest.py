"""
Pytest configuration and fixtures for FraudLens test suite.

Patches ModelPredictor with a deterministic mock so tests run without
a live MLflow server. The mock satisfies all contracts expected by
the FastAPI endpoints and predictor unit tests.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_mock_predictor() -> MagicMock:
    mock = MagicMock()

    mock.is_ready.return_value = True
    mock.optimal_threshold = 0.5
    mock.model_metadata = {
        "run_id": "test-run-id-0000",
        "version": "1",
        "algorithm": "xgboost",
        "auc_pr": 0.91,
        "auc_roc": 0.93,
        "f1": 0.85,
        "best_threshold": 0.5,
        "creation_timestamp": 1716000000000,
    }

    def _predict(X):
        n = len(X)
        return {
            "fraud_score": [0.125] * n if n > 0 else [],
            "n_samples": n,
        }

    mock.predict.side_effect = _predict

    mock.get_model_info.return_value = {
        "model_name": "fraudlens-detector",
        "version": "1",
        "algorithm": "xgboost",
        "auc_pr": 0.91,
        "auc_roc": 0.93,
        "f1": 0.85,
        "optimal_threshold": 0.5,
        "registered_at": 1716000000.0,
    }

    mock.get_pipeline.side_effect = RuntimeError("Feature pipeline not loaded")

    return mock


@pytest.fixture(scope="session", autouse=True)
def mock_mlflow_predictor():
    """
    Replace the module-level predictor singleton before any test runs.

    get_predictor() checks `if predictor is None` — by injecting a mock
    here, every endpoint and unit test gets the same deterministic stub
    without touching MLflow or the filesystem.
    """
    mock = _make_mock_predictor()

    with patch("src.api.predictor.predictor", mock):
        yield mock
