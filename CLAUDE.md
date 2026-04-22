# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Full pipeline (run in order)
```bash
python -m src.preprocess          # fit preprocessor, split data → models/preprocessor.pkl + data/processed/
python -m src.train               # train 4 models, save best → models/best_model.pkl
uvicorn api.main:app --reload --port 8000
```

### Tests
```bash
pytest tests/ -v                  # all tests (requires trained models for test_api.py)
pytest tests/test_preprocess.py   # no models needed
pytest tests/test_train.py        # no models needed
pytest tests/test_api.py          # requires models/best_model.pkl + models/preprocessor.pkl
```

### Docker
```bash
docker compose up --build         # models/ must exist before building
```

## Architecture

The project is split into two independent layers that communicate through serialized `.pkl` files in `models/`.

**ML Pipeline (`src/`)** — must be run before the API:
- `utils.py` — single source of truth for path constants (`ROOT_DIR`, `DATA_DIR`, `MODEL_DIR`, `REPORT_DIR`) and `TARGET_COL`/`RANDOM_SEED`. All other modules import from here.
- `preprocess.py` — `build_preprocessor()` returns a `ColumnTransformer` (median impute + StandardScaler for numerics; most_frequent impute + OneHotEncoder for categoricals). `run_preprocessing()` fits it on train data and saves `preprocessor.pkl`. The preprocessor is intentionally fitted only on train data and reused in both training and the API.
- `train.py` — loads `preprocessor.pkl`, transforms the processed CSVs, trains all four models in `MODELS` dict, ranks by ROC-AUC, saves the winner as `best_model.pkl`. Adding a new model only requires adding it to `MODELS`.
- `evaluate.py` — pure utility: `compute_metrics()` returns a dict, `plot_*` functions save PNGs to `reports/figures/`.

**API (`api/`)** — stateless FastAPI app:
- `api/main.py` — app factory. The `lifespan` context manager calls `predict.load_artifacts()` on startup, which sets the module-level `_model` and `_preprocessor` globals in `predict.py`. This is the only place models are loaded.
- `api/schemas.py` — `CustomerFeatures` uses `Literal` types to enumerate all valid categorical values, matching the exact strings in the Kaggle CSV. The `SeniorCitizen` column is `int` (0/1), not `"Yes"/"No"` — this is a known quirk of the dataset.
- `api/routes/predict.py` — converts the Pydantic model to a single-row DataFrame, runs `preprocessor.transform()` (not `fit_transform`), then calls `predict_proba`. The threshold is a module constant (`THRESHOLD = 0.5`).

**Key constraint:** `customerID` is dropped in `preprocess.py:run_preprocessing()` before saving CSVs. `SeniorCitizen` is kept as a raw feature but is not in `CATEGORICAL_FEATURES` — it stays numeric as-is.

## Data

Raw CSV expected at: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

`TotalCharges` has whitespace entries for new customers (tenure=0) — `load_raw_data()` coerces these to `NaN`, which the median imputer then fills. `Churn` is binary-encoded to `int` immediately on load.
