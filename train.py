"""Customer Churn Model Training Pipeline.

Extracts data preprocessing and model training from the experimental notebook
into a production-grade, headless Python script.
"""

import argparse
import sys
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "senior_citizen",
]

CATEGORICAL_FEATURES = [
    "gender",
    "contract_type",
    "internet_service",
    "payment_method",
    "technical_support",
    "paperless_billing",
    "streaming_service",
    "has_dependents",
]


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Builds the preprocessing and classifier pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=random_state)),
        ]
    )


def train(
    data_path: str = "customer_churn_prediction_dataset.csv",
    model_output_path: str = "churn_model.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Loads dataset, trains churn model pipeline, evaluates, and exports artifact."""
    print(f"Loading dataset from: {data_path}...")
    df = pd.read_csv(data_path)

    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    if "churn" not in df.columns:
        raise ValueError("Target column 'churn' not found in dataset.")

    x = df.drop(columns=["churn"])
    y = df["churn"].map({"Yes": 1, "No": 0})

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    print(f"Dataset split: {len(x_train)} train samples, {len(x_test)} test samples.")

    pipeline = build_pipeline(random_state=random_state)

    print("Training Random Forest Churn model...")
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    acc = accuracy_score(y_test, predictions)

    print("\n--- Model Evaluation ---")
    print(f"Test Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, predictions, target_names=["Retained", "Churned"]))

    print(f"Saving model artifact to: {model_output_path}...")
    joblib.dump(pipeline, model_output_path)
    print("Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Customer Churn Prediction Model")
    parser.add_argument(
        "--data-path",
        default="customer_churn_prediction_dataset.csv",
        help="Path to training CSV file",
    )
    parser.add_argument(
        "--model-output",
        default="churn_model.pkl",
        help="Path where trained model artifact will be saved",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()
    train(
        data_path=args.data_path,
        model_output_path=args.model_output,
        test_size=args.test_size,
        random_state=args.random_state,
    )
