"""Customer Churn Model Training & Preprocessing Pipeline.

This module automates the end-to-end Machine Learning training workflow:
1. Data Ingestion: Loads raw tabular customer records from CSV.
2. Data Sanitization: Removes non-predictive identifiers (e.g. `customer_id`)
   to prevent target leakage.
3. Feature Engineering & Preprocessing:
   - Numerical: Imputes missing values via median and normalizes with StandardScaler.
   - Categorical: Imputes missing values with modal frequency and applies OneHotEncoding.
4. Model Training: Fits a balanced Random Forest ensemble classifier.
5. Evaluation: Computes classification metrics (Precision, Recall, F1, Accuracy).
6. Artifact Serialization: Exports fitted Pipeline with joblib for production serving.
"""

import argparse
import sys
from typing import Tuple
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Feature Column Definitions
# ---------------------------------------------------------------------------

# Continuous and discrete numerical features requiring scaling and numeric imputation
NUMERIC_FEATURES = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "senior_citizen",
]

# Categorical features requiring string imputation and one-hot encoding
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


# ---------------------------------------------------------------------------
# Pipeline Builder
# ---------------------------------------------------------------------------

def build_pipeline(random_state: int = 42) -> Pipeline:
    """Constructs a unified scikit-learn Pipeline with preprocessing and classifier.

    Design Rationale:
    - Numeric Sub-pipeline:
      * `SimpleImputer(strategy='median')`: Resilient against extreme billing/tenure outliers.
      * `StandardScaler`: Normalizes variances to unit scale, preventing scale dominance.
    - Categorical Sub-pipeline:
      * `SimpleImputer(strategy='most_frequent')`: Fills null categories with modal class.
      * `OneHotEncoder(handle_unknown='ignore')`: Prevents runtime crashes when novel or
        unseen category levels appear during production serving.
    - Estimator:
      * `RandomForestClassifier`: Robust ensemble method capable of capturing non-linear
        interactions between billing duration and service tiers without heavy tuning.

    Args:
        random_state: Random seed ensuring deterministic model training and reproducibility.

    Returns:
        Configured, un-fitted scikit-learn Pipeline instance.
    """
    # Define transformations for numeric features
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Define transformations for categorical features
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine transformers across targeted feature columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # Discard unlisted or unexpected columns safely
    )

    # Assemble complete preprocessing + model execution graph
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=random_state)),
        ]
    )


# ---------------------------------------------------------------------------
# Training & Evaluation Engine
# ---------------------------------------------------------------------------

def train(
    data_path: str = "customer_churn_prediction_dataset.csv",
    model_output_path: str = "churn_model.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, float]:
    """Executes the complete training workflow from data load to model serialization.

    Args:
        data_path: File system path to the customer dataset CSV.
        model_output_path: Destination path for saving the trained .pkl model artifact.
        test_size: Proportion of the dataset reserved for test evaluation (default 0.2).
        random_state: Seed used for train-test splitting and model initialization.

    Returns:
        Tuple containing the fitted Pipeline object and the test accuracy score.

    Raises:
        FileNotFoundError: If the CSV file does not exist at data_path.
        ValueError: If the required 'churn' target column is missing.
    """
    # Step 1: Load raw customer dataset
    print(f"[1/6] Loading customer records from: {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError as err:
        print(f"Error: Dataset not found at {data_path}", file=sys.stderr)
        raise err

    # Step 2: Prevent data leakage by eliminating arbitrary customer identifiers
    if "customer_id" in df.columns:
        print("  - Dropping non-predictive identifier column: 'customer_id'")
        df = df.drop(columns=["customer_id"])

    # Step 3: Validate and extract target variable
    if "churn" not in df.columns:
        raise ValueError("Target column 'churn' not found in dataset.")

    # Separate feature matrix (X) and map binary target (y: Yes -> 1, No -> 0)
    x = df.drop(columns=["churn"])
    y = df["churn"].map({"Yes": 1, "No": 0})

    # Step 4: Perform stratified/controlled train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    print(f"[2/6] Split dataset into: {len(x_train)} train samples, {len(x_test)} test samples.")

    # Step 5: Build pipeline and fit model
    print("[3/6] Initializing preprocessing and Random Forest classifier...")
    pipeline = build_pipeline(random_state=random_state)

    print("[4/6] Fitting pipeline on training partition...")
    pipeline.fit(x_train, y_train)

    # Step 6: Generate inferences and evaluate generalization metrics
    print("[5/6] Evaluating model on unseen holdout test set...")
    predictions = pipeline.predict(x_test)
    acc = accuracy_score(y_test, predictions)

    print("\n" + "=" * 50)
    print("                EVALUATION REPORT")
    print("=" * 50)
    print(f"Holdout Test Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, predictions, target_names=["Retained (0)", "Churned (1)"]))
    print("=" * 50 + "\n")

    # Step 7: Serialize production model artifact
    print(f"[6/6] Serializing fitted model artifact to: {model_output_path}...")
    joblib.dump(pipeline, model_output_path)
    print("Training pipeline finished successfully!\n")

    return pipeline, acc


# ---------------------------------------------------------------------------
# CLI Command Line Interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Headless CLI for training Customer Churn Prediction model."
    )
    parser.add_argument(
        "--data-path",
        default="customer_churn_prediction_dataset.csv",
        help="Path to training CSV file (default: customer_churn_prediction_dataset.csv)",
    )
    parser.add_argument(
        "--model-output",
        default="churn_model.pkl",
        help="Output path for serialized model pickle (default: churn_model.pkl)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of dataset to hold out for testing (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for deterministic runs (default: 42)",
    )

    cli_args = parser.parse_args()
    train(
        data_path=cli_args.data_path,
        model_output_path=cli_args.model_output,
        test_size=cli_args.test_size,
        random_state=cli_args.random_state,
    )
