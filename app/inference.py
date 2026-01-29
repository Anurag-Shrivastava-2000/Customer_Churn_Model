import pandas as pd
import joblib
from pathlib import Path
from source.preprocess import preprocess_data

import mlflow.sklearn

MODEL_URI = "models:/CustomerChurnXGB@production"

model = mlflow.sklearn.load_model(MODEL_URI)

def predict(input_data: dict):
    """
    Shared inference pipeline for Flask & Gradio
    """
    df = pd.DataFrame([input_data])
    df_processed = preprocess_data(df)

    prob = model.predict_proba(df_processed)[:, 1][0]
    pred = "Likely to churn" if prob >= 0.3 else "Not likely to churn"

    return {
        "prediction": pred,
        "churn_probability": round(float(prob), 4)
    }
