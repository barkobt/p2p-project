import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.evaluate import compute_metrics, plot_confusion_matrix, plot_roc
from src.utils import DATA_DIR, MODEL_DIR, RANDOM_SEED, TARGET_COL, get_logger

log = get_logger(__name__)

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    "RandomForest":       RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED),
    "XGBoost":            XGBClassifier(eval_metric="logloss", random_state=RANDOM_SEED, verbosity=0),
    "LightGBM":           LGBMClassifier(random_state=RANDOM_SEED, verbose=-1),
}


def run_training() -> None:
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")

    train_df = pd.read_csv(DATA_DIR / "processed" / "train.csv")
    test_df  = pd.read_csv(DATA_DIR / "processed" / "test.csv")

    X_train = preprocessor.transform(train_df.drop(columns=[TARGET_COL]))
    y_train = train_df[TARGET_COL]
    X_test  = preprocessor.transform(test_df.drop(columns=[TARGET_COL]))
    y_test  = test_df[TARGET_COL]

    results = []
    trained = {}

    for name, model in MODELS.items():
        log.info(f"{name} eğitiliyor...")
        model.fit(X_train, y_train)
        metrics = compute_metrics(model, X_test, y_test)
        metrics["model"] = name
        results.append(metrics)
        trained[name] = model

        plot_confusion_matrix(model, X_test, y_test, name)
        plot_roc(model, X_test, y_test, name)

    results_df = (
        pd.DataFrame(results)
        .set_index("model")
        .sort_values("roc_auc", ascending=False)
    )
    print("\n=== Model Karşılaştırması ===")
    print(results_df.to_string())

    best_name = results_df.index[0]
    joblib.dump(trained[best_name], MODEL_DIR / "best_model.pkl")
    log.info(f"En iyi model: {best_name} → models/best_model.pkl")


if __name__ == "__main__":
    run_training()
