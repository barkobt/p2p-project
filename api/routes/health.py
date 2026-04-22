from fastapi import APIRouter

from src.utils import MODEL_DIR, MODEL_METADATA_PATH

router = APIRouter()


@router.get("/health", tags=["Ops"])
def health_check() -> dict:
    model_exists = (MODEL_DIR / "best_model.pkl").exists()
    preprocessor_exists = (MODEL_DIR / "preprocessor.pkl").exists()
    metadata_exists = MODEL_METADATA_PATH.exists()
    return {
        "status": "ok",
        "model_loaded": model_exists and preprocessor_exists and metadata_exists,
    }
