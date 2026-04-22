import pytest
from fastapi.testclient import TestClient

SAMPLE_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5, "TotalCharges": 1020.0,
}


@pytest.fixture(scope="module")
def client():
    # Modeller eğitildikten sonra çalıştır: pytest tests/test_api.py
    from api.main import app
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_predict_returns_probability(client):
    resp = client.post("/api/v1/predict", json=SAMPLE_CUSTOMER)
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["churn_prediction"], bool)


def test_predict_invalid_input(client):
    resp = client.post("/api/v1/predict", json={"gender": "Unknown"})
    assert resp.status_code == 422
