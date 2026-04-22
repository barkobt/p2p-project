from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.routes import health, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    predict.load_artifacts()
    yield


app = FastAPI(
    title="Telco Customer Churn API",
    description="Müşteri churn olasılığını tahmin eden REST API.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="frontend/templates")

app.include_router(health.router)
app.include_router(predict.router, prefix="/api/v1")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

# Çalıştırmak için: uvicorn api.main:app --reload --port 8000
