from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production REST API for predicting telecommunication customer churn.",
    version="1.0.0",
)

model = joblib.load("churn_model.pkl")


class CustomerData(BaseModel):
    gender: Literal["Female", "Male", "Other"] | str
    age: float | None = Field(default=None, ge=18, le=120)
    tenure_months: float | None = Field(default=None, ge=0)
    contract_type: Literal["Month-to-month", "One year", "Two year"] | str
    internet_service: Literal["Fiber optic", "DSL", "Cable", "None"] | str | None = None
    payment_method: (
        Literal[
            "Mailed check",
            "Electronic check",
            "Credit card",
            "Bank transfer",
        ]
        | str
        | None
    ) = None
    technical_support: Literal["Yes", "No"] | str | None = None
    paperless_billing: Literal["Yes", "No"] | str | None = None
    streaming_service: Literal["Yes", "No"] | str | None = None
    has_dependents: Literal["Yes", "No"] | str | None = None
    monthly_charges: float | None = Field(default=None, ge=0)
    total_charges: float | None = Field(default=None, ge=0)
    senior_citizen: float | None = Field(default=None, ge=0, le=1)


class PredictionResponse(BaseModel):
    will_churn: bool
    churn_probability_percentage: float
    message: str


@app.get("/", tags=["General"])
def root():
    return {
        "service": "Customer Churn Prediction API",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(customer: CustomerData):
    input_data = pd.DataFrame([customer.model_dump()])

    prediction = model.predict(input_data)
    prediction_val = int(prediction[0])
    probability = float(model.predict_proba(input_data)[0][1])
    will_churn = bool(prediction_val == 1)

    return PredictionResponse(
        will_churn=will_churn,
        churn_probability_percentage=round(probability * 100, 2),
        message=(
            "High risk of churning!"
            if will_churn
            else "Customer is likely to stay."
        ),
    )
