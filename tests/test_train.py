import numpy as np
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.evaluate import compute_metrics
from src.train import (
    MODEL_NAMES,
    _build_estimator,
    _cross_validate_candidate,
    compute_acceptance_metrics,
    optimize_threshold,
)


def test_compute_metrics_keys():
    X, y = make_classification(n_samples=200, random_state=42)
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    metrics = compute_metrics(model, X, y)
    assert set(metrics.keys()) == {"accuracy", "f1", "precision", "recall", "roc_auc"}


def test_metrics_range():
    X, y = make_classification(n_samples=200, random_state=42)
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    metrics = compute_metrics(model, X, y)
    for v in metrics.values():
        assert 0.0 <= v <= 1.0


def test_optimize_threshold_is_bounded_and_deterministic():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0], dtype=int)
    y_proba = np.array([0.1, 0.2, 0.55, 0.8, 0.6, 0.3, 0.7, 0.4], dtype=float)
    t1, f1_1 = optimize_threshold(y_true, y_proba)
    t2, f1_2 = optimize_threshold(y_true, y_proba)
    assert 0.0 <= t1 <= 1.0
    assert 0.0 <= f1_1 <= 1.0
    assert t1 == t2
    assert f1_1 == f1_2


def test_build_estimator_smote_pipeline():
    est = _build_estimator(
        "logreg_liblinear_l2",
        "smote",
        pos_weight=2.0,
        params={"C": 1.0},
    )
    assert isinstance(est, ImbPipeline)
    assert "smote" in est.named_steps


def test_linear_only_model_set():
    assert "RandomForest" not in MODEL_NAMES
    assert "XGBoost" not in MODEL_NAMES
    assert "LightGBM" not in MODEL_NAMES
    assert "logreg_liblinear_l2" in MODEL_NAMES


def test_acceptance_metrics_use_cv_f1_and_roc_auc_from_baseline():
    winner = {"cv_f1": 0.63, "cv_roc_auc": 0.84}
    baseline = {"cv_f1": 0.61, "cv_roc_auc": 0.845}
    out = compute_acceptance_metrics(winner=winner, baseline=baseline)
    assert out["improvement_vs_baseline_f1"] == 0.02
    assert out["roc_auc_drop"] == 0.005
    assert out["acceptance_checks"]["min_f1_improvement_met"] is True


def test_smote_applies_only_on_fold_train(monkeypatch: pytest.MonkeyPatch):
    captured_sizes = []

    class SpySMOTE:
        def __init__(self, random_state):
            self.random_state = random_state

        def fit_resample(self, X, y):
            captured_sizes.append(len(y))
            return X, y

    monkeypatch.setattr("src.train.SMOTE", SpySMOTE)
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=4,
        n_redundant=0,
        weights=[0.75, 0.25],
        random_state=42,
    )
    _cross_validate_candidate(
        model_name="logreg_liblinear_l2",
        strategy="smote",
        X=X,
        y=y,
        pos_weight=3.0,
        params={"C": 1.0},
    )
    assert len(captured_sizes) == 5
    assert all(size < len(y) for size in captured_sizes)
