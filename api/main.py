from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import health, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    predict.load_artifacts()
    yield


app = FastAPI(
    title="Telco Customer Churn API",
    description="Müşteri churn olasılığını tahmin eden REST API.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(predict.router, prefix="/api/v1")

# Çalıştırmak için: uvicorn api.main:app --reload --port 8000
