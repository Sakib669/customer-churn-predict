"""Automated Integration & Unit Test Suite for Customer Churn Prediction.

This test suite ensures:
1. API Availability: Verifies the root (`/`) and health check (`/health`) endpoints.
2. Inference Validity: Confirms valid payloads return proper schemas and bounded probabilities.
3. Input Validation: Ensures Pydantic schema catches boundary violations (e.g. negative tenure,
   out-of-range age) with HTTP 422 Unprocessable Entity.
4. Pipeline Structural Integrity: Validates that the ML pipeline architecture matches expected
   preprocessor and estimator stages.
"""

import pytest
from fastapi.testclient import TestClient
from main import app
from train import build_pipeline

# Initialize TestClient bound to the FastAPI application
client = TestClient(app)


def test_root_endpoint() -> None:
    """Tests that the service index returns HTTP 200 OK and accurate metadata."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["service"] == "Customer Churn Prediction API"
    assert data["status"] == "online"
    assert data["version"] == "1.0.0"
    assert "docs_url" in data


def test_health_check_endpoint() -> None:
    """Verifies that the health check endpoint confirms model readiness."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_churn_endpoint_valid_payload() -> None:
    """Verifies end-to-end inference for a complete, valid customer payload.

    Assertions:
    - Status code is 200 OK.
    - `will_churn` is a boolean.
    - `churn_probability_percentage` is bounded between 0.0% and 100.0%.
    - `message` string is populated.
    """
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


def test_predict_validation_error_negative_tenure() -> None:
    """Ensures Pydantic rejects negative tenure with HTTP 422 Unprocessable Entity."""
    payload = {
        "gender": "Male",
        "contract_type": "Month-to-month",
        "tenure_months": -5.0,  # Fails ge=0 validation
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_validation_error_invalid_age() -> None:
    """Ensures Pydantic rejects age values under 18 with HTTP 422."""
    payload = {
        "gender": "Female",
        "contract_type": "One year",
        "age": 12.0,  # Fails ge=18 validation
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_pipeline_architecture_verification() -> None:
    """Verifies that build_pipeline() constructs the expected composite steps."""
    pipeline = build_pipeline(random_state=42)

    # Check top-level stages
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

    # Verify column transformers
    preprocessor = pipeline.named_steps["preprocessor"]
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names
