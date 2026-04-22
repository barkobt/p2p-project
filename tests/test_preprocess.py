import pandas as pd
import pytest

from src.preprocess import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_preprocessor


def make_sample_df() -> pd.DataFrame:
    return pd.DataFrame([{
        "tenure": 12, "MonthlyCharges": 65.0, "TotalCharges": 780.0,
        "gender": "Female", "Partner": "Yes", "Dependents": "No",
        "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No",
        "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }])


def test_build_preprocessor_transforms():
    df = make_sample_df()
    prep = build_preprocessor()
    prep.fit(df)
    out = prep.transform(df)
    assert out.shape[0] == 1
    assert out.shape[1] > len(NUMERIC_FEATURES)  # OHE genişletir


def test_numeric_and_categorical_coverage():
    all_cols = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    assert "tenure" in all_cols
    assert "Contract" in all_cols
