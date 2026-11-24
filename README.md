# Customer Churn Prediction System

A production-ready Machine Learning system that predicts customer churn in telecommunications using **Scikit-Learn**, **FastAPI**, **Pydantic**, and **UV**.

---

## 📌 Architecture Overview

```
                      +------------------------------------------+
                      | customer_churn_prediction_dataset.csv   |
                      +--------------------+---------------------+
                                           |
                                           v
                             +-----------------------------+
                             | train.py / index.ipynb      |
                             | (ColumnTransformer Pipeline)|
                             +-------------+---------------+
                                           |
                                           v
                               +-----------------------+
                               |    churn_model.pkl    |
                               +-----------+-----------+
                                           |
                                           v
+-----------------------+      +-----------------------+      +-------------------------+
| Client / HTTP Request | ---> | FastAPI Serving Layer | ---> |   PredictionResponse    |
| (JSON Payload)        |      | (main.py on Port 8000)|      | (will_churn, % prob)    |
+-----------------------+      +-----------------------+      +-------------------------+
```

---

## 📊 Dataset & Pipeline

* **Source File**: `customer_churn_prediction_dataset.csv` (8,080 records)
* **Target Feature**: `churn` (`Yes`: 40.78%, `No`: 59.22%)
* **Feature Processing**:
  * **Numeric Features** (`age`, `tenure_months`, `monthly_charges`, `total_charges`, `senior_citizen`): Imputed with median and scaled using `StandardScaler`.
  * **Categorical Features** (`gender`, `contract_type`, `internet_service`, `payment_method`, `technical_support`, `paperless_billing`, `streaming_service`, `has_dependents`): Imputed with most frequent values and encoded using `OneHotEncoder(handle_unknown='ignore')`.
* **Model Classifier**: `RandomForestClassifier(random_state=42)`

### Model Performance

| Metric | Retained (0) | Churned (1) | Overall |
| :--- | :---: | :---: | :---: |
| **Precision** | 0.68 | 0.59 | — |
| **Recall** | 0.76 | 0.50 | — |
| **F1-Score** | 0.72 | 0.54 | — |
| **Accuracy** | — | — | **65.10%** |

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
* Python `>=3.14`
* [uv](https://github.com/astral-sh/uv) fast package manager

### 2. Install Dependencies
```bash
uv sync
```

### 3. Train Model
To re-train the pipeline headlessly and generate an updated `churn_model.pkl`:
```bash
uv run python train.py
```

### 4. Run Development API Server
```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Interactive Swagger documentation is available at: [http://localhost:8000/docs](http://localhost:8000/docs).

### 5. Run Automated Tests
```bash
uv run pytest
```

---

## 🔌 API Reference

### Health Check
`GET /health`
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Predict Churn
`POST /predict`

**Request Body:**
```json
{
  "gender": "Female",
  "age": 38.0,
  "tenure_months": 14.9,
  "contract_type": "Month-to-month",
  "internet_service": "Fiber optic",
  "payment_method": "Credit card",
  "technical_support": "No",
  "paperless_billing": "No",
  "streaming_service": "No",
  "has_dependents": "Yes",
  "monthly_charges": 71.41,
  "total_charges": 1055.68,
  "senior_citizen": 0.0
}
```

**Response:**
```json
{
  "will_churn": true,
  "churn_probability_percentage": 74.0,
  "message": "High risk of churning!"
}
```

---

## 🐳 Docker Deployment

Build and run using Docker:
```bash
# Build Docker image
docker build -t customer-churn-api:latest .

# Run container on port 8000
docker run -d -p 8000:8000 --name churn-service customer-churn-api:latest
```

Verify service:
```bash
curl http://localhost:8000/health
```
