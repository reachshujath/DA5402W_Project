import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from api.main import app, model_service
from fraud_mlops.config import FEATURE_COLUMNS
from fraud_mlops.preprocessing import build_model_pipeline


@pytest.fixture(autouse=True)
def loaded_model():
    frame = pd.DataFrame(np.zeros((4, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    frame["Amount"] = 1.0
    model = build_model_pipeline(DummyClassifier(strategy="prior"))
    model.fit(frame, [0, 0, 1, 1])
    previous = model_service.artifact
    model_service.artifact = {
        "pipeline": model,
        "threshold": 0.5,
        "feature_columns": FEATURE_COLUMNS,
        "model_name": "test-model",
        "model_type": "dummy",
        "model_version": "test-1",
    }
    yield
    model_service.artifact = previous


def test_health_endpoint_returns_ok():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_returns_traceable_model_metadata():
    client = TestClient(app)
    payload = {column: 0.0 for column in FEATURE_COLUMNS}
    payload["Amount"] = 10.0
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "test-model"
    assert body["model_type"] == "dummy"
    assert body["request_id"]


def test_prediction_rejects_missing_features():
    response = TestClient(app).post("/predict", json={"Amount": 10.0})
    assert response.status_code == 422


def test_drift_endpoint_returns_monitor_state():
    response = TestClient(app).get("/drift")
    assert response.status_code == 200
    assert "window_size" in response.json()
