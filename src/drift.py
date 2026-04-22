import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.features import add_engineered_features
from src.utils import (
    DATA_DIR,
    DRIFT_REPORT_DIR,
    PREDICTION_LOG_DIR,
    TARGET_COL,
    get_logger,
)

log = get_logger(__name__)

NUMERIC_FEATURES_FOR_DRIFT = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "ContractMonths",
    "ChargePerTenure",
    "ServiceCount",
]


def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if ref.size == 0 or cur.size == 0:
        return 0.0

    edges = np.quantile(ref, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if edges.size <= 2:
        return 0.0

    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_ratio = np.clip(ref_hist / max(ref_hist.sum(), 1), 1e-6, None)
    cur_ratio = np.clip(cur_hist / max(cur_hist.sum(), 1), 1e-6, None)
    psi = np.sum((cur_ratio - ref_ratio) * np.log(cur_ratio / ref_ratio))
    return round(float(psi), 4)


def _psi_level(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.2:
        return "moderate_shift"
    return "high_shift"


def _load_logged_features() -> pd.DataFrame:
    rows = []
    for path in sorted(PREDICTION_LOG_DIR.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                features = rec.get("features")
                if isinstance(features, dict):
                    rows.append(features)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def run_drift_check() -> None:
    reference_path = DATA_DIR / "processed" / "train.csv"
    if not reference_path.exists():
        raise FileNotFoundError(
            "Önce preprocess çalıştırılmalı: data/processed/train.csv bulunamadı."
        )

    reference_df = pd.read_csv(reference_path)
    reference_df = add_engineered_features(
        reference_df.drop(columns=[TARGET_COL], errors="ignore")
    )

    current_df = _load_logged_features()
    if current_df.empty:
        log.info("Prediction log bulunamadı. Drift raporu üretilmedi.")
        return
    current_df = add_engineered_features(current_df)

    report_rows = []
    for feature in NUMERIC_FEATURES_FOR_DRIFT:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        psi = compute_psi(reference_df[feature], current_df[feature])
        report_rows.append(
            {
                "feature": feature,
                "psi": psi,
                "level": _psi_level(psi),
                "reference_count": int(reference_df[feature].notna().sum()),
                "current_count": int(current_df[feature].notna().sum()),
            }
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_rows": int(reference_df.shape[0]),
        "current_rows": int(current_df.shape[0]),
        "features": report_rows,
    }

    DRIFT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        DRIFT_REPORT_DIR
        / f"drift_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("Drift raporu yazıldı: %s", out_path)


if __name__ == "__main__":
    run_drift_check()
