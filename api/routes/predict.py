import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import (
    BatchPredictionItem,
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchPredictionSummary,
    CustomerFeatures,
    PredictionResponse,
)
from src.features import add_engineered_features
from src.utils import MODEL_DIR, MODEL_METADATA_PATH, PREDICTION_LOG_DIR, append_jsonl

router = APIRouter()

_model = None
_preprocessor = None
_metadata = None
_threshold = None


def _build_single_response(probability: float) -> PredictionResponse:
    return PredictionResponse(
        churn_probability=round(probability, 4),
        churn_prediction=probability >= _threshold,
        threshold_used=_threshold,
    )


def _predict_proba(rows: list[dict]) -> list[float]:
    raw_df = pd.DataFrame(rows)
    features_df = add_engineered_features(raw_df)
    X = _preprocessor.transform(features_df)
    return _model.predict_proba(X)[:, 1].tolist()


def _log_prediction(
    features: dict, prediction: dict, mode: str, row_index: int | None = None
) -> None:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "features": features,
        "prediction": prediction,
    }
    if row_index is not None:
        payload["row_index"] = row_index
    append_jsonl(
        PREDICTION_LOG_DIR / f"{datetime.now(timezone.utc).date().isoformat()}.jsonl",
        payload,
    )


def load_artifacts() -> None:
    global _model, _preprocessor, _metadata, _threshold
    _model = joblib.load(MODEL_DIR / "best_model.pkl")
    _preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
    with MODEL_METADATA_PATH.open("r", encoding="utf-8") as f:
        _metadata = json.load(f)
    _threshold = float(_metadata["selected_threshold"])
    if not 0.0 <= _threshold <= 1.0:
        raise ValueError(
            "Metadata içindeki selected_threshold [0,1] aralığında olmalı."
        )


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(customer: CustomerFeatures) -> PredictionResponse:
    if (
        _model is None
        or _preprocessor is None
        or _metadata is None
        or _threshold is None
    ):
        raise HTTPException(
            status_code=503, detail="Model artefaktları henüz yüklenmedi."
        )
    try:
        row = customer.model_dump()
        proba = float(_predict_proba([row])[0])
        response = _build_single_response(proba)
        _log_prediction(row, response.model_dump(), mode="single")
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"]
)
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    if (
        _model is None
        or _preprocessor is None
        or _metadata is None
        or _threshold is None
    ):
        raise HTTPException(
            status_code=503, detail="Model artefaktları henüz yüklenmedi."
        )
    try:
        rows = [customer.model_dump() for customer in payload.customers]
        probabilities = _predict_proba(rows)
        items = []
        churn_count = 0
        for idx, (row, proba) in enumerate(zip(rows, probabilities)):
            prediction = _build_single_response(float(proba))
            if prediction.churn_prediction:
                churn_count += 1
            item = BatchPredictionItem(row_index=idx, **prediction.model_dump())
            items.append(item)
            _log_prediction(row, item.model_dump(), mode="batch", row_index=idx)

        return BatchPredictionResponse(
            predictions=items,
            summary=BatchPredictionSummary(
                total=len(items), predicted_churn=churn_count
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
