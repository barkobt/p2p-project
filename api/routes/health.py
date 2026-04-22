from fastapi import APIRouter

from src.utils import MODEL_DIR

router = APIRouter()


@router.get("/health", tags=["Ops"])
def health_check() -> dict:
    model_exists = (MODEL_DIR / "best_model.pkl").exists()
    return {"status": "ok", "model_loaded": model_exists}
