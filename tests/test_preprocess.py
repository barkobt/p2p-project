import pandas as pd

from src.features import add_engineered_features
from src.preprocess import (
    CATEGORICAL_FEATURES,
    EXTENDED_CATEGORICAL,
    EXTENDED_NUMERIC,
    NUMERIC_FEATURES,
    build_preprocessor,
)


def make_sample_df() -> pd.DataFrame:
    raw = pd.DataFrame(
        [
            {
                "SeniorCitizen": 0,
                "tenure": 12,
                "MonthlyCharges": 65.0,
                "TotalCharges": 780.0,
                "gender": "Female",
                "Partner": "Yes",
                "Dependents": "No",
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
                "PaymentMethod": "Electronic check",
            }
        ]
    )
    return add_engineered_features(raw)


def test_build_preprocessor_transforms():
    df = make_sample_df()
    prep = build_preprocessor(include_extended_features=True)
    prep.fit(df)
    out = prep.transform(df)
    assert out.shape[0] == 1
    assert out.shape[1] > len(NUMERIC_FEATURES)  # OHE genişletir


def test_numeric_and_categorical_coverage():
    all_cols = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    assert "SeniorCitizen" in all_cols
    assert "tenure" in all_cols
    assert "ChargePerTenure" in all_cols
    assert "ServiceCount" in all_cols
    assert "HasFamily" in all_cols
    assert "Contract" in all_cols


def test_engineered_features_are_generated():
    df = make_sample_df()
    assert "ContractMonths" in df.columns
    assert "ChargePerTenure" in df.columns
    assert "ServiceCount" in df.columns
    assert "HasFamily" in df.columns
    assert "IsFiberMonthToMonth" in df.columns
    assert "NoSupportFiber" in df.columns
    assert "HighChargeShortTenure" in df.columns


def test_extended_feature_lists_are_exposed():
    assert "HighChargeShortTenure" in EXTENDED_NUMERIC
    assert "IsFiberMonthToMonth" in EXTENDED_CATEGORICAL
