"""
FASTAPI + GRADIO SERVING APPLICATION - Production-Ready ML Model Serving

This application provides a complete serving solution for the Telco Customer Churn model
with both programmatic API access and a user-friendly web interface.

Architecture:
- FastAPI: High-performance REST API with automatic OpenAPI documentation
- Gradio: User-friendly web UI for manual testing and demonstrations
- Pydantic: Data validation and automatic API documentation
"""

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr

# Core ML inference logic
from app.inference import predict


# FastAPI app initialization

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for predicting customer churn in the telecom industry",
    version="1.0.0"
)


# HEALTH CHECK ENDPOINT 

@app.get("/")
def health_check():
    """
    Health check endpoint for AWS Application Load Balancer and monitoring.
    """
    return {"status": "ok"}


# REQUEST DATA SCHEMA (PYDANTIC)

class CustomerData(BaseModel):
    """
    Customer data schema for churn prediction.

    Defines all 18 features required by the trained ML model.
    """

    # Demographics
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str

    # Phone services
    PhoneService: str
    MultipleLines: str

    # Internet services
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str

    # Account details
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    # Numeric features
    tenure: int
    MonthlyCharges: float
    TotalCharges: float


# PREDICTION API ENDPOINT

@app.post("/predict")
def predict_churn(data: CustomerData):
    """
    REST API endpoint for churn prediction.

    Flow:
    1. Validate request using Pydantic
    2. Call ML inference pipeline
    3. Return prediction result
    """
    try:
        result = predict(data.dict())
        return {
            "prediction": result
        }
    except Exception as e:
        return {
            "error": str(e)
        }


# GRADIO INTERFACE FUNCTION

def gradio_interface(
    gender, SeniorCitizen, Partner, Dependents,
    PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies,
    Contract, PaperlessBilling, PaymentMethod,
    tenure, MonthlyCharges, TotalCharges
):
    data = {
        "gender": gender,
        "SeniorCitizen": int(SeniorCitizen),
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": int(tenure),
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
        "PaymentMethod": PaymentMethod,
    }

    result = predict(data)
    return f"{result['prediction']} (prob={result['churn_probability']})"



# GRADIO UI CONFIGURATION

demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender", value="Male"),
        gr.Dropdown([0, 1],label="Senior Citizen (1 = Yes, 0 = No)",value=0),
        gr.Dropdown(["Yes", "No"], label="Partner", value="No"),
        gr.Dropdown(["Yes", "No"], label="Dependents", value="No"),

        gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes"),
        gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No"),

        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="Yes"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="Yes"),

        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes"),
        gr.Dropdown(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            label="Payment Method",
            value="Electronic check"
        ),

        gr.Number(label="Tenure (months)", value=1, minimum=0),
        gr.Number(label="Monthly Charges ($)", value=85.0, minimum=0),
        gr.Number(label="Total Charges ($)", value=85.0, minimum=0),
    ],
    outputs=gr.Textbox(label="Churn Prediction", lines=2),
    title="Telco Customer Churn Predictor",
    description="""
    **Predict customer churn using a machine learning model**

    This application uses an XGBoost-based model trained on historical
    telecom customer data to identify customers at risk of churn.
    """,
    examples=[
        ["Female", "No", "No", "Yes", "No", "Fiber optic", "No", "No", "No",
         "No", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check",
         1, 85.0, 85.0],
        ["Male", "Yes", "Yes", "Yes", "Yes", "DSL", "Yes", "Yes", "Yes",
         "Yes", "No", "No", "Two year", "No", "Credit card (automatic)",
         60, 45.0, 2700.0],
    ],
    theme=gr.themes.Soft()
)


# MOUNT GRADIO INTO FASTAPI

app = gr.mount_gradio_app(
    app,
    demo,
    path="/ui"
)
