import json
from datetime import datetime, timezone
from uuid import uuid4

import joblib
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src.config import TRAINING_CONFIG
from src.evaluate import compute_metrics, plot_confusion_matrix, plot_roc
from src.utils import (
    DATA_DIR,
    MODEL_COMPARISON_PATH,
    MODEL_DIR,
    MODEL_METADATA_PATH,
    TARGET_COL,
    get_logger,
)

log = get_logger(__name__)

STRATEGIES = ("baseline", "class_weight", "smote")
MODEL_NAMES = (
    "logreg_liblinear_l2",
    "logreg_saga_elasticnet",
    "calibrated_sgd_logloss",
    "calibrated_linear_svc",
    "xgboost",
    "lightgbm",
)
COMPARISON_COLUMNS = [
    "model",
    "strategy",
    "stage",
    "cv_f1",
    "cv_f1_std",
    "cv_roc_auc",
    "oof_f1_optimized",
    "selected_threshold",
    "params",
]


def optimize_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, points: int | None = None
) -> tuple[float, float]:
    points = points or TRAINING_CONFIG.threshold_points
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, points):
        score = f1_score(y_true, (y_proba >= threshold).astype(int))
        if score > best_f1 or (
            np.isclose(score, best_f1)
            and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_f1 = score
            best_threshold = float(threshold)
    return round(best_threshold, 4), round(float(best_f1), 4)


def _base_estimator(
    model_name: str, strategy: str, pos_weight: float, params: dict | None = None
):
    params = dict(params or {})
    seed = TRAINING_CONFIG.random_seed
    class_weight = "balanced" if strategy == "class_weight" else None
    if strategy == "class_weight" and model_name == "logreg_liblinear_l2":
        params.setdefault("class_weight", class_weight)
    if strategy == "class_weight" and model_name == "logreg_saga_elasticnet":
        params.setdefault("class_weight", class_weight)

    if model_name == "logreg_liblinear_l2":
        defaults = {
            "solver": "liblinear",
            "max_iter": 4000,
            "random_state": seed,
        }
        defaults.update(params)
        return LogisticRegression(**defaults)

    if model_name == "logreg_saga_elasticnet":
        defaults = {
            "solver": "saga",
            "l1_ratio": 0.5,
            "max_iter": 6000,
            "random_state": seed,
        }
        defaults.update(params)
        return LogisticRegression(**defaults)

    if model_name == "calibrated_sgd_logloss":
        base = SGDClassifier(
            loss="log_loss",
            alpha=float(params.get("alpha", 1e-4)),
            penalty=params.get("penalty", "l2"),
            l1_ratio=float(params.get("l1_ratio", 0.15)),
            class_weight=class_weight,
            random_state=seed,
            max_iter=5000,
            tol=1e-4,
        )
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)

    if model_name == "calibrated_linear_svc":
        base = LinearSVC(
            C=float(params.get("C", 1.0)),
            class_weight=class_weight,
            random_state=seed,
            max_iter=6000,
        )
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)

    if model_name == "xgboost":
        scale_pos_weight = pos_weight if strategy == "class_weight" else 1.0
        return XGBClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=seed,
            verbosity=0,
        )

    if model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=int(params.get("max_depth", -1)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            num_leaves=int(params.get("num_leaves", 31)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            class_weight="balanced" if strategy == "class_weight" else None,
            random_state=seed,
            verbose=-1,
        )

    raise ValueError(f"Desteklenmeyen model: {model_name}")


def _build_estimator(
    model_name: str,
    strategy: str,
    pos_weight: float,
    params: dict | None = None,
):
    estimator = _base_estimator(
        model_name=model_name, strategy=strategy, pos_weight=pos_weight, params=params
    )
    if strategy == "smote":
        return ImbPipeline(
            steps=[
                ("smote", SMOTE(random_state=TRAINING_CONFIG.random_seed)),
                ("model", estimator),
            ]
        )
    return estimator


def _cross_validate_candidate(
    model_name: str,
    strategy: str,
    X: np.ndarray,
    y: np.ndarray,
    pos_weight: float,
    params: dict | None = None,
) -> dict:
    cv = StratifiedKFold(
        n_splits=TRAINING_CONFIG.cv_folds,
        shuffle=True,
        random_state=TRAINING_CONFIG.random_seed,
    )
    oof_proba = np.zeros(y.shape[0], dtype=float)
    fold_f1 = []
    fold_roc_auc = []
    for train_idx, valid_idx in cv.split(X, y):
        estimator = _build_estimator(model_name, strategy, pos_weight, params)
        estimator = clone(estimator)
        estimator.fit(X[train_idx], y[train_idx])
        fold_proba = estimator.predict_proba(X[valid_idx])[:, 1]
        oof_proba[valid_idx] = fold_proba
        fold_pred = (fold_proba >= 0.5).astype(int)
        fold_f1.append(f1_score(y[valid_idx], fold_pred))
        fold_roc_auc.append(roc_auc_score(y[valid_idx], fold_proba))

    threshold, best_f1 = optimize_threshold(y, oof_proba)
    return {
        "cv_f1": round(float(np.mean(fold_f1)), 4),
        "cv_f1_std": round(float(np.std(fold_f1)), 4),
        "cv_roc_auc": round(float(np.mean(fold_roc_auc)), 4),
        "oof_f1_optimized": best_f1,
        "selected_threshold": threshold,
    }


def _suggest_params(trial: optuna.Trial, model_name: str) -> dict:
    if model_name == "logreg_liblinear_l2":
        return {
            "C": trial.suggest_float("C", 1e-4, 20.0, log=True),
        }
    if model_name == "logreg_saga_elasticnet":
        return {
            "C": trial.suggest_float("C", 1e-4, 20.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }
    if model_name == "calibrated_sgd_logloss":
        return {
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }
    if model_name == "calibrated_linear_svc":
        return {
            "C": trial.suggest_float("C", 1e-3, 20.0, log=True),
        }
    if model_name == "xgboost":
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
    if model_name == "lightgbm":
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "num_leaves":       trial.suggest_int("num_leaves", 20, 150),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
    raise ValueError(f"Desteklenmeyen model: {model_name}")


def _tune_candidate(
    model_name: str,
    strategy: str,
    X: np.ndarray,
    y: np.ndarray,
    pos_weight: float,
) -> tuple[dict, dict]:
    sampler = optuna.samplers.TPESampler(seed=TRAINING_CONFIG.random_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, model_name)
        stats = _cross_validate_candidate(
            model_name=model_name,
            strategy=strategy,
            X=X,
            y=y,
            pos_weight=pos_weight,
            params=params,
        )
        return stats["cv_f1"]

    study.optimize(
        objective, n_trials=TRAINING_CONFIG.optuna_trials, show_progress_bar=False
    )
    best_params = study.best_trial.params
    tuned_stats = _cross_validate_candidate(
        model_name=model_name,
        strategy=strategy,
        X=X,
        y=y,
        pos_weight=pos_weight,
        params=best_params,
    )
    return best_params, tuned_stats


def compute_acceptance_metrics(winner: dict, baseline: dict) -> dict:
    improvement = round(float(winner["cv_f1"]) - float(baseline["cv_f1"]), 4)
    roc_auc_drop = round(float(baseline["cv_roc_auc"]) - float(winner["cv_roc_auc"]), 4)
    acceptance_checks = {
        "min_f1_improvement_met": improvement >= TRAINING_CONFIG.min_f1_improvement,
        "target_f1_improvement_met": improvement
        >= TRAINING_CONFIG.target_f1_improvement,
        "roc_auc_drop_within_limit": roc_auc_drop <= TRAINING_CONFIG.max_roc_auc_drop,
    }
    return {
        "improvement_vs_baseline_f1": improvement,
        "roc_auc_drop": roc_auc_drop,
        "acceptance_checks": acceptance_checks,
    }


def _load_pass_inputs() -> (
    tuple[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
    train_df = pd.read_csv(DATA_DIR / "processed" / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "processed" / "test.csv")
    X_train = preprocessor.transform(train_df.drop(columns=[TARGET_COL]))
    y_train = train_df[TARGET_COL].to_numpy()
    X_test = preprocessor.transform(test_df.drop(columns=[TARGET_COL]))
    y_test = test_df[TARGET_COL].to_numpy()
    return preprocessor, X_train, y_train, X_test, y_test


def _sort_comparison_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["cv_f1", "cv_roc_auc"], ascending=False).reset_index(
        drop=True
    )


def _run_training_pass(
    run_id: str,
    pass_name: str,
) -> dict:
    preprocessor, X_train, y_train, X_test, y_test = _load_pass_inputs()
    pos_weight = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)

    screening_rows = []
    for model_name in MODEL_NAMES:
        for strategy in STRATEGIES:
            log.info(
                "Aday değerlendiriliyor: %s + %s (%s)", model_name, strategy, pass_name
            )
            stats = _cross_validate_candidate(
                model_name=model_name,
                strategy=strategy,
                X=X_train,
                y=y_train,
                pos_weight=pos_weight,
            )
            screening_rows.append(
                {
                    "model": model_name,
                    "strategy": strategy,
                    "stage": "screening",
                    "params": "{}",
                    **stats,
                }
            )

    screening_df = _sort_comparison_df(pd.DataFrame(screening_rows))
    top_candidates = screening_df.head(TRAINING_CONFIG.top_k_for_tuning)

    tuned_rows = []
    for _, candidate in top_candidates.iterrows():
        model_name = candidate["model"]
        strategy = candidate["strategy"]
        log.info(
            "Optuna tuning başlıyor: %s + %s (%s)", model_name, strategy, pass_name
        )
        best_params, tuned_stats = _tune_candidate(
            model_name=model_name,
            strategy=strategy,
            X=X_train,
            y=y_train,
            pos_weight=pos_weight,
        )
        tuned_rows.append(
            {
                "model": model_name,
                "strategy": strategy,
                "stage": "tuned",
                "params": json.dumps(best_params, sort_keys=True),
                **tuned_stats,
            }
        )

    tuned_df = (
        pd.DataFrame(tuned_rows)
        if tuned_rows
        else pd.DataFrame(columns=screening_df.columns)
    )
    comparison_df = _sort_comparison_df(
        pd.concat([screening_df, tuned_df], ignore_index=True)
    )
    comparison_df = comparison_df[COMPARISON_COLUMNS]

    winner = comparison_df.iloc[0].to_dict()
    best_model_name = winner["model"]
    best_strategy = winner["strategy"]
    best_params = json.loads(winner["params"])
    best_threshold = float(winner["selected_threshold"])

    estimator = _build_estimator(
        model_name=best_model_name,
        strategy=best_strategy,
        pos_weight=pos_weight,
        params=best_params,
    )
    estimator.fit(X_train, y_train)
    test_metrics = compute_metrics(estimator, X_test, y_test, threshold=best_threshold)
    plot_confusion_matrix(
        estimator,
        X_test,
        y_test,
        f"{best_model_name}_{best_strategy}_{pass_name}",
        best_threshold,
    )
    plot_roc(
        estimator, X_test, y_test, f"{best_model_name}_{best_strategy}_{pass_name}"
    )

    baseline_rows = screening_df.loc[screening_df["strategy"] == "baseline"]
    baseline_row = _sort_comparison_df(baseline_rows).iloc[0].to_dict()
    acceptance = compute_acceptance_metrics(winner=winner, baseline=baseline_row)

    metadata = {
        "run_id": run_id,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_name": best_model_name,
        "strategy": best_strategy,
        "cv_f1": float(winner["cv_f1"]),
        "cv_f1_std": float(winner["cv_f1_std"]),
        "cv_roc_auc": float(winner["cv_roc_auc"]),
        "oof_f1_optimized": float(winner["oof_f1_optimized"]),
        "selected_threshold": best_threshold,
        "primary_metric": TRAINING_CONFIG.primary_metric,
        "primary_metric_name": TRAINING_CONFIG.primary_metric_name,
        "primary_metric_value": float(winner["cv_f1"]),
        "baseline_primary_metric_value": float(baseline_row["cv_f1"]),
        "baseline_best_f1": float(baseline_row["cv_f1"]),
        "baseline_reference_roc_auc": float(baseline_row["cv_roc_auc"]),
        "improvement_vs_baseline_f1": acceptance["improvement_vs_baseline_f1"],
        "roc_auc_drop": acceptance["roc_auc_drop"],
        "acceptance_checks": acceptance["acceptance_checks"],
        "params": best_params,
        "test_metrics": test_metrics,
        "search_space_version": TRAINING_CONFIG.search_space_version,
        "trials_used": TRAINING_CONFIG.optuna_trials * TRAINING_CONFIG.top_k_for_tuning,
        "pass_name": pass_name,
    }
    return {
        "comparison_df": comparison_df,
        "winner": winner,
        "model": estimator,
        "preprocessor": preprocessor,
        "metadata": metadata,
    }


def _acceptance_status(acceptance_checks: dict) -> str:
    if acceptance_checks.get("min_f1_improvement_met", False):
        return "target_met"
    return "target_not_met"


def run_training() -> None:
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    result = _run_training_pass(run_id=run_id, pass_name="run1")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df = result["comparison_df"][COMPARISON_COLUMNS]
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    joblib.dump(result["model"], MODEL_DIR / "best_model.pkl")
    joblib.dump(result["preprocessor"], MODEL_DIR / "preprocessor.pkl")

    metadata = dict(result["metadata"])
    metadata["acceptance_status"] = _acceptance_status(metadata["acceptance_checks"])
    with MODEL_METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== Model Karşılaştırması ===")
    print(comparison_df.to_string(index=False))
    print("\n=== Seçilen Model ===")
    print(
        f"{metadata['model_name']} + {metadata['strategy']} | "
        f"CV F1={metadata['cv_f1']:.4f} | "
        f"CV ROC-AUC={metadata['cv_roc_auc']:.4f} | "
        f"threshold={metadata['selected_threshold']:.4f} | "
        f"acceptance={metadata['acceptance_status']}"
    )


if __name__ == "__main__":
    run_training()
