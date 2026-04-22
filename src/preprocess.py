import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import TRAINING_CONFIG
from src.features import (
    BASE_CATEGORICAL_FEATURES,
    BASE_NUMERIC_FEATURES,
    EXTENDED_CATEGORICAL_FEATURES,
    EXTENDED_NUMERIC_FEATURES,
    add_engineered_features,
    get_feature_lists,
)
from src.utils import DATA_DIR, MODEL_DIR, TARGET_COL, get_logger

log = get_logger(__name__)

NUMERIC_FEATURES = list(BASE_NUMERIC_FEATURES)
CATEGORICAL_FEATURES = list(BASE_CATEGORICAL_FEATURES)
EXTENDED_NUMERIC = list(EXTENDED_NUMERIC_FEATURES)
EXTENDED_CATEGORICAL = list(EXTENDED_CATEGORICAL_FEATURES)


def build_preprocessor(include_extended_features: bool = False) -> ColumnTransformer:
    numeric_features, categorical_features = get_feature_lists(
        include_extended_features=include_extended_features
    )
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def load_raw_data() -> pd.DataFrame:
    path = DATA_DIR / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(path)
    # Boşluk içeren TotalCharges değerlerini NaN'a çevir
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df[TARGET_COL] = (df[TARGET_COL].str.strip() == "Yes").astype(int)
    return df


def run_preprocessing(
    test_size: float | None = None, include_extended_features: bool = False
) -> None:
    log.info("Ham veri yükleniyor...")
    df = add_engineered_features(
        load_raw_data(), include_extended_features=include_extended_features
    )
    if test_size is None:
        test_size = TRAINING_CONFIG.test_size

    X = df.drop(columns=["customerID", TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=TRAINING_CONFIG.random_seed
    )

    preprocessor = build_preprocessor(
        include_extended_features=include_extended_features
    )
    preprocessor.fit(X_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")
    log.info("Preprocessor kaydedildi → models/preprocessor.pkl")

    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values
    train_df.to_csv(DATA_DIR / "processed" / "train.csv", index=False)
    test_df.to_csv(DATA_DIR / "processed" / "test.csv", index=False)
    log.info("İşlenmiş veriler kaydedildi → data/processed/")


if __name__ == "__main__":
    run_preprocessing()
