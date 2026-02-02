import pandas as pd
from source.preprocess import preprocess_data

def test_preprocess_output_shape_and_columns():
    raw_input = pd.DataFrame([{
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
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
        "MonthlyCharges": 80.0,
        "TotalCharges": 800.0,
        "PaymentMethod": "Electronic check",
    }])

    processed = preprocess_data(raw_input)

    # Basic assertions
    assert processed.shape[0] == 1
    assert processed.isnull().sum().sum() == 0
    assert processed.dtypes.apply(lambda x: x.kind in "iuf").all()
