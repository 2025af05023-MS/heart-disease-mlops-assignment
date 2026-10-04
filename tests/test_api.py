import numpy as np
from fastapi.testclient import TestClient

from heartlab import api


class FakeModel:
    def predict_proba(self, frame):
        assert list(frame.columns) == api.FEATURES
        return np.array([[0.25, 0.75]])


SAMPLE = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
}


def test_predict_returns_probability_and_confidence(monkeypatch):
    monkeypatch.setattr(api, "get_model", lambda: FakeModel())
    result = TestClient(api.app).post("/predict", json=SAMPLE)
    assert result.status_code == 200
    assert result.json()["prediction"] == 1
    assert result.json()["probability_positive"] == 0.75
    assert result.json()["confidence"] == 0.75


def test_predict_rejects_bad_category():
    result = TestClient(api.app).post("/predict", json={**SAMPLE, "cp": 9})
    assert result.status_code == 422


def test_metrics_endpoint():
    result = TestClient(api.app).get("/metrics")
    assert result.status_code == 200
    assert "heartlab_prediction_seconds" in result.text


def test_health_rejects_unloadable_model(monkeypatch):
    def broken_model():
        raise ValueError("invalid model file")

    monkeypatch.setattr(api, "get_model", broken_model)
    result = TestClient(api.app).get("/health")
    assert result.status_code == 503
