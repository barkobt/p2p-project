import json

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

SAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 1020.0,
}


class DummyPreprocessor:
    def transform(self, df):
        cols = [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "ServiceCount",
        ]
        for col in cols:
            if col not in df.columns:
                raise KeyError(f"{col} sütunu bekleniyor.")
        return df[cols].to_numpy(dtype=float)


class DummyModel:
    def predict_proba(self, X):
        score = (X[:, 1] * 0.02) + (X[:, 2] * 0.01) - (X[:, 0] * 0.2)
        probs = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - probs, probs])


def _patch_model_paths(monkeypatch: pytest.MonkeyPatch, model_dir, log_dir) -> None:
    from api.routes import health, predict

    monkeypatch.setattr(predict, "MODEL_DIR", model_dir)
    monkeypatch.setattr(health, "MODEL_DIR", model_dir)
    monkeypatch.setattr(
        predict, "MODEL_METADATA_PATH", model_dir / "model_metadata.json"
    )
    monkeypatch.setattr(
        health, "MODEL_METADATA_PATH", model_dir / "model_metadata.json"
    )
    monkeypatch.setattr(predict, "PREDICTION_LOG_DIR", log_dir)
    predict._model = None
    predict._preprocessor = None
    predict._metadata = None
    predict._threshold = None


@pytest.fixture
def artifact_dir(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    log_dir = tmp_path / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(DummyModel(), model_dir / "best_model.pkl")
    joblib.dump(DummyPreprocessor(), model_dir / "preprocessor.pkl")
    with (model_dir / "model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"selected_threshold": 0.42, "run_id": "test-run"}, f)

    _patch_model_paths(monkeypatch, model_dir, log_dir)
    return model_dir


@pytest.fixture
def client(artifact_dir):
    from api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_uses_metadata_threshold(client):
    resp = client.post("/api/v1/predict", json=SAMPLE_CUSTOMER)
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["threshold_used"] == 0.42
    assert isinstance(data["churn_prediction"], bool)


def test_predict_batch(client):
    payload = {"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]}
    resp = client.post("/api/v1/predict/batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["summary"]["total"] == 2
    assert len(data["predictions"]) == 2
    assert data["summary"]["predicted_churn"] in (0, 1, 2)
    assert data["predictions"][0]["row_index"] == 0


def test_predict_batch_empty_list(client):
    resp = client.post("/api/v1/predict/batch", json={"customers": []})
    assert resp.status_code == 422


def test_predict_batch_limit(client):
    payload = {"customers": [SAMPLE_CUSTOMER] * 501}
    resp = client.post("/api/v1/predict/batch", json=payload)
    assert resp.status_code == 422


def test_predict_invalid_input(client):
    resp = client.post("/api/v1/predict", json={"gender": "Unknown"})
    assert resp.status_code == 422


def test_startup_fails_without_metadata(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    log_dir = tmp_path / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(DummyModel(), model_dir / "best_model.pkl")
    joblib.dump(DummyPreprocessor(), model_dir / "preprocessor.pkl")
    _patch_model_paths(monkeypatch, model_dir, log_dir)

    from api.main import app

    with pytest.raises(FileNotFoundError):
        with TestClient(app):
            pass
