import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import CustomerFeatures, PredictionResponse
from src.utils import MODEL_DIR

router = APIRouter()

# Model ve preprocessor uygulama başlangıcında bir kez yüklenir
_model        = None
_preprocessor = None

THRESHOLD = 0.5


def load_artifacts() -> None:
    global _model, _preprocessor
    _model        = joblib.load(MODEL_DIR / "best_model.pkl")
    _preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(customer: CustomerFeatures) -> PredictionResponse:
    if _model is None or _preprocessor is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi.")
    try:
        df    = pd.DataFrame([customer.model_dump()])
        X     = _preprocessor.transform(df)
        proba = float(_model.predict_proba(X)[0, 1])
        return PredictionResponse(
            churn_probability=round(proba, 4),
            churn_prediction=proba >= THRESHOLD,
            threshold_used=THRESHOLD,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
