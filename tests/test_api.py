import pytest
from fastapi.testclient import TestClient
from main import app
from train import build_pipeline

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Customer Churn Prediction API"
    assert data["status"] == "online"
    assert "version" in data


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_churn_endpoint():
    payload = {
        "gender": "Female",
        "age": 39.0,
        "tenure_months": 59.7,
        "contract_type": "One year",
        "internet_service": "Fiber optic",
        "payment_method": "Credit card",
        "technical_support": "No",
        "paperless_billing": "Yes",
        "streaming_service": "Yes",
        "has_dependents": "No",
        "monthly_charges": 74.92,
        "total_charges": 4516.74,
        "senior_citizen": 0.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "will_churn" in data
    assert isinstance(data["will_churn"], bool)
    assert "churn_probability_percentage" in data
    assert 0.0 <= data["churn_probability_percentage"] <= 100.0
    assert "message" in data


def test_predict_validation_error():
    # Negative tenure should fail validation (ge=0)
    payload = {
        "gender": "Male",
        "contract_type": "Month-to-month",
        "tenure_months": -5.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_pipeline_architecture():
    pipeline = build_pipeline(random_state=42)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
