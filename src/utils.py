import logging
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
REPORT_DIR = REPORTS_DIR / "figures"
DRIFT_REPORT_DIR = REPORTS_DIR / "drift"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
MODEL_COMPARISON_PATH = MODEL_DIR / "model_comparison.csv"
MLRUNS_DIR = ROOT_DIR / "mlruns"
PREDICTION_LOG_DIR = ROOT_DIR / "logs" / "predictions"

TARGET_COL = "Churn"
RANDOM_SEED = 42


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True))
        f.write("\n")
