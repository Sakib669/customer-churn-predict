from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("churn_model.pkl")


class CustomerData(BaseModel):
    gender: str
    age: float | None = None
    tenure_months: float | None = None
    contract_type: str
    internet_service: str | None = None
    payment_method: str | None = None
    technical_support: str | None = None
    paperless_billing: str | None = None
    streaming_service: str | None = None
    has_dependents: str | None = None
    monthly_charges: float | None = None
    total_charges: float | None = None
    senior_citizen: float | None = None


@app.post("/predict")
def predict_churn(customer: CustomerData):

    input_data = pd.DataFrame([customer.model_dump()])

    prediction = model.predict(input_data)
    prediction_val = int(prediction[0])
    probability = float(model.predict_proba(input_data)[0][1])
    will_churn = bool(prediction_val == 1)

    return {
        "will_churn": will_churn,
        "churn_probability_percentage": round(probability * 100, 2),
        "message": (
            "High risk of churning!"
            if will_churn
            else "Customer is likely to stay."
        ),
    }
