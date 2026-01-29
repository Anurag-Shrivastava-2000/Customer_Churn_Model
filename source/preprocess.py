import pandas as pd
import json
from pathlib import Path

FEATURE_PATH = Path(__file__).parent / "feature_columns.json"


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Production preprocessing function for inference.
    - No target column
    - Feature-locked
    - Safe defaults
    """

    df = df.copy()
    df.columns = df.columns.str.strip()

    # ===============================
    # Drop ID columns (if present)
    # ===============================
    for col in ["customerID", "CustomerID", "customer_id", "Churn"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # ===============================
    # Binary columns
    # ===============================
    binary_cols = [
        "gender", "Partner", "Dependents",
        "PhoneService", "PaperlessBilling"
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].replace({
                "Yes": 1, "No": 0,
                "Male": 1, "Female": 0
            })

    # ===============================
    # SeniorCitizen (CRITICAL)
    # ===============================
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    else:
        df["SeniorCitizen"] = 0  # safe default

    # ===============================
    # Multi-category binary
    # ===============================
    multi_cat_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]
    for col in multi_cat_cols:
        if col in df.columns:
            df[col] = df[col].replace({
                "Yes": 1,
                "No": 0,
                "No internet service": 0,
                "No phone service": 0
            })

    # ===============================
    # Ordinal encoding
    # ===============================
    ordinal_maps = {
        "InternetService": {"No": 0, "DSL": 1, "Fiber optic": 2},
        "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
    }
    for col, mapping in ordinal_maps.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)

    # ===============================
    # One-hot encode PaymentMethod
    # ===============================
    if "PaymentMethod" in df.columns:
        df = pd.get_dummies(df, columns=["PaymentMethod"], drop_first=True)

    # ===============================
    # Numeric coercion
    # ===============================
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    df[obj_cols] = df[obj_cols].apply(pd.to_numeric, errors="coerce")

    df.fillna(0, inplace=True)

    # ===============================
    # 🔒 Feature locking (MOST IMPORTANT)
    # ===============================
    if FEATURE_PATH.exists():
        with open(FEATURE_PATH) as f:
            feature_cols = json.load(f)

        df = df.reindex(columns=feature_cols, fill_value=0)

    # ===============================
    # Final safety: no bools
    # ===============================
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df
