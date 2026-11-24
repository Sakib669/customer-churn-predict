"""Customer Churn Prediction API Service.

This module provides a production-grade REST API for serving real-time
customer churn inferences using a serialized Scikit-Learn pipeline.

Key Capabilities:
- Pydantic validation with domain-specific constraints (age bounds, non-negative charges).
- Real-time probability estimation and binary risk classification.
- Health checking and API documentation endpoints.
"""

from typing import Any, Literal
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Application Initialization & Model Loading
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production-grade REST service for telecommunication customer churn prediction. "
        "Provides binary churn forecasts and risk probabilities."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Load the pre-trained Scikit-learn Pipeline (ColumnTransformer + RandomForestClassifier)
MODEL_PATH = "churn_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------

class CustomerData(BaseModel):
    """Input customer profile schema with validation constraints."""

    gender: Literal["Female", "Male", "Other"] | str = Field(
        ...,
        description="Gender of the customer ('Female', 'Male', or 'Other').",
        examples=["Female"],
    )
    age: float | None = Field(
        default=None,
        ge=18,
        le=120,
        description="Customer age in years (must be between 18 and 120).",
        examples=[39.0],
    )
    tenure_months: float | None = Field(
        default=None,
        ge=0,
        description="Total duration with the provider in months.",
        examples=[59.7],
    )
    contract_type: Literal["Month-to-month", "One year", "Two year"] | str = Field(
        ...,
        description="Subscription contract duration type.",
        examples=["One year"],
    )
    internet_service: Literal["Fiber optic", "DSL", "Cable", "None"] | str | None = Field(
        default=None,
        description="Subscribed internet connectivity type.",
        examples=["Fiber optic"],
    )
    payment_method: (
        Literal[
            "Mailed check",
            "Electronic check",
            "Credit card",
            "Bank transfer",
        ]
        | str
        | None
    ) = Field(
        default=None,
        description="Primary billing and payment method.",
        examples=["Credit card"],
    )
    technical_support: Literal["Yes", "No"] | str | None = Field(
        default=None,
        description="Indicates whether customer subscribes to dedicated technical support.",
        examples=["No"],
    )
    paperless_billing: Literal["Yes", "No"] | str | None = Field(
        default=None,
        description="Indicates whether customer opted for electronic/paperless statements.",
        examples=["Yes"],
    )
    streaming_service: Literal["Yes", "No"] | str | None = Field(
        default=None,
        description="Indicates whether customer uses streaming entertainment add-ons.",
        examples=["Yes"],
    )
    has_dependents: Literal["Yes", "No"] | str | None = Field(
        default=None,
        description="Indicates whether customer has dependent family members.",
        examples=["No"],
    )
    monthly_charges: float | None = Field(
        default=None,
        ge=0,
        description="Current monthly subscription charge in USD.",
        examples=[74.92],
    )
    total_charges: float | None = Field(
        default=None,
        ge=0,
        description="Cumulative charges billed throughout tenure in USD.",
        examples=[4516.74],
    )
    senior_citizen: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Binary indicator: 1.0 for senior citizen, 0.0 otherwise.",
        examples=[0.0],
    )


class PredictionResponse(BaseModel):
    """Structured response payload returned by the inference endpoint."""

    will_churn: bool = Field(
        ...,
        description="Binary prediction: True indicates high risk of churn, False indicates likely retention.",
    )
    churn_probability_percentage: float = Field(
        ...,
        description="Predicted probability of churn as a percentage (0.0 to 100.0).",
        examples=[26.0],
    )
    message: str = Field(
        ...,
        description="Human-readable assessment summarizing customer risk status.",
        examples=["Customer is likely to stay."],
    )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["General"])
def root() -> dict[str, str]:
    """Root endpoint returning basic service metadata and documentation links."""
    return {
        "service": "Customer Churn Prediction API",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, Any]:
    """Liveness & readiness probe verifying application status and model availability."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact could not be loaded.",
        )
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
)
def predict_churn(customer: CustomerData) -> PredictionResponse:
    """Computes churn probability and classification for a given customer profile.

    Processing Steps:
    1. Converts Pydantic customer payload into a 1-row Pandas DataFrame.
    2. Runs inference through the Scikit-learn Pipeline (preprocessing + Random Forest).
    3. Extracts class probability for the positive churn class (index 1).
    4. Formats and returns a strongly-typed PredictionResponse.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Churn model is not initialized.",
        )

    # Convert customer data to DataFrame aligned with training schema
    input_data = pd.DataFrame([customer.model_dump()])

    # Generate binary prediction (0: Stay, 1: Churn)
    prediction = model.predict(input_data)
    prediction_val = int(prediction[0])
    will_churn = bool(prediction_val == 1)

    # Extract prediction probability for churn class (index 1)
    probability = float(model.predict_proba(input_data)[0][1])

    return PredictionResponse(
        will_churn=will_churn,
        churn_probability_percentage=round(probability * 100, 2),
        message=(
            "High risk of churning!"
            if will_churn
            else "Customer is likely to stay."
        ),
    )
