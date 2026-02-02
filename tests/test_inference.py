from app.inference import predict

def test_predict_returns_valid_output():
    sample_input = {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "MonthlyCharges": 85.0,
        "TotalCharges": 85.0,
        "PaymentMethod": "Electronic check",
    }

    result = predict(sample_input)

    assert "prediction" in result
    assert "churn_probability" in result
    assert 0.0 <= result["churn_probability"] <= 1.0
