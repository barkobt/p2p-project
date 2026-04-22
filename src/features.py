import numpy as np
import pandas as pd

CONTRACT_MONTHS_MAP = {
    "Month-to-month": 1.0,
    "One year": 12.0,
    "Two year": 24.0,
}

SERVICE_COUNT_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

BASE_NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "ContractMonths",
    "ChargePerTenure",
    "ServiceCount",
]

BASE_CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "HasFamily",
]

EXTENDED_NUMERIC_FEATURES = ["HighChargeShortTenure"]
EXTENDED_CATEGORICAL_FEATURES = ["IsFiberMonthToMonth", "NoSupportFiber"]


def get_feature_lists(
    include_extended_features: bool = False,
) -> tuple[list[str], list[str]]:
    numeric = list(BASE_NUMERIC_FEATURES)
    categorical = list(BASE_CATEGORICAL_FEATURES)
    if include_extended_features:
        numeric.extend(EXTENDED_NUMERIC_FEATURES)
        categorical.extend(EXTENDED_CATEGORICAL_FEATURES)
    return numeric, categorical


def add_engineered_features(
    df: pd.DataFrame, include_extended_features: bool = True
) -> pd.DataFrame:
    out = df.copy()
    out["ContractMonths"] = out["Contract"].map(CONTRACT_MONTHS_MAP).fillna(1.0)

    tenure_safe = out["tenure"].replace(0, np.nan)
    charge_per_tenure = (out["TotalCharges"] / tenure_safe).replace(
        [np.inf, -np.inf], np.nan
    )
    out["ChargePerTenure"] = charge_per_tenure.fillna(out["MonthlyCharges"])

    service_count = np.zeros(out.shape[0], dtype=int)
    for col in SERVICE_COUNT_COLUMNS:
        service_count += (out[col] == "Yes").astype(int)
    out["ServiceCount"] = service_count

    out["HasFamily"] = np.where(
        (out["Partner"] == "Yes") | (out["Dependents"] == "Yes"),
        "Yes",
        "No",
    )

    if include_extended_features:
        out["IsFiberMonthToMonth"] = np.where(
            (out["InternetService"] == "Fiber optic")
            & (out["Contract"] == "Month-to-month"),
            "Yes",
            "No",
        )
        out["NoSupportFiber"] = np.where(
            (out["InternetService"] == "Fiber optic") & (out["TechSupport"] == "No"),
            "Yes",
            "No",
        )
        out["HighChargeShortTenure"] = np.where(
            (out["MonthlyCharges"] >= 80.0) & (out["tenure"] <= 6),
            1.0,
            0.0,
        )
    return out
