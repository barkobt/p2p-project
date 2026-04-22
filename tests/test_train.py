import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluate import compute_metrics


def test_compute_metrics_keys():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, random_state=42)
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    metrics = compute_metrics(model, X, y)
    assert set(metrics.keys()) == {"accuracy", "f1", "precision", "recall", "roc_auc"}


def test_metrics_range():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, random_state=42)
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    metrics = compute_metrics(model, X, y)
    for v in metrics.values():
        assert 0.0 <= v <= 1.0
