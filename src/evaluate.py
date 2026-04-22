import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import REPORT_DIR


def compute_metrics(model, X_test, y_test) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "f1":        round(f1_score(y_test, y_pred),        4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred),    4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba),  4),
    }


def plot_confusion_matrix(model, X_test, y_test, model_name: str) -> None:
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT_DIR / f"cm_{model_name}.png", bbox_inches="tight")
    plt.close(fig)


def plot_roc(model, X_test, y_test, model_name: str) -> None:
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT_DIR / f"roc_{model_name}.png", bbox_inches="tight")
    plt.close(fig)
