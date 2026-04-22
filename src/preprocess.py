import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import DATA_DIR, MODEL_DIR, RANDOM_SEED, TARGET_COL, get_logger

log = get_logger(__name__)

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline,      NUMERIC_FEATURES),
        ("cat", categorical_pipeline,  CATEGORICAL_FEATURES),
    ])


def load_raw_data() -> pd.DataFrame:
    path = DATA_DIR / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(path)
    # Boşluk içeren TotalCharges değerlerini NaN'a çevir
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df[TARGET_COL] = (df[TARGET_COL].str.strip() == "Yes").astype(int)
    return df


def run_preprocessing(test_size: float = 0.2) -> None:
    log.info("Ham veri yükleniyor...")
    df = load_raw_data()

    X = df.drop(columns=["customerID", TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED
    )

    preprocessor = build_preprocessor()
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
    test_df.to_csv(DATA_DIR  / "processed" / "test.csv",  index=False)
    log.info("İşlenmiş veriler kaydedildi → data/processed/")


if __name__ == "__main__":
    run_preprocessing()
