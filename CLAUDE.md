# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

End-to-end ML + web service for Telco Customer Churn prediction.

```
[Raw CSV]
   ↓ src/preprocess.py
[models/preprocessor.pkl]  +  [data/processed/train.csv, test.csv]
   ↓ src/train.py
[models/best_model.pkl]  +  [models/model_metadata.json]
   ↓ uvicorn api.main:app
[http://localhost:8000]  →  Web UI  +  REST API
```

**The ML pipeline must run before the API starts.** The two layers share no in-memory state — only `.pkl` and `.json` files in `models/`.

---

## Commands

### Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### ML Pipeline

```bash
python -m src.pipeline          # tek komut: preprocess → train
# veya adım adım:
python -m src.preprocess        # → models/preprocessor.pkl + data/processed/
python -m src.train             # → models/best_model.pkl + model_metadata.json
```

### API + UI

```bash
uvicorn api.main:app --reload --port 8000
# http://localhost:8000       → Web arayüzü
# http://localhost:8000/docs  → Swagger
```

### Tests

```bash
pytest tests/ -v                     # 20 test, veri dosyası gerekmez
pytest tests/test_api.py -v          # API, batch, startup failure
pytest tests/test_preprocess.py -v   # Feature engineering, preprocessor
pytest tests/test_train.py -v        # Metrikler, threshold, SMOTE pipeline
pytest tests/test_drift.py -v        # PSI hesaplama
```

### Docker (opsiyonel)

```bash
python -m src.pipeline && docker build -t churn-api . && docker run -p 8000:8000 churn-api
```

### Drift (opsiyonel, prediction log biriktikten sonra)

```bash
python -m src.drift   # → reports/drift/drift_report_<tarih>.json
```

---

## Architecture

### `src/` — ML Pipeline

**`src/utils.py` — tüm path sabitleri buradan import edilir**

```python
ROOT_DIR, DATA_DIR, MODEL_DIR, REPORTS_DIR, REPORT_DIR
DRIFT_REPORT_DIR, PREDICTION_LOG_DIR
MODEL_METADATA_PATH   # models/model_metadata.json
MODEL_COMPARISON_PATH # models/model_comparison.csv
TARGET_COL = "Churn"
RANDOM_SEED = 42
append_jsonl(path, payload)  # prediction logging için
```

Başka hiçbir modül path string'i hardcode etmez.

---

**`src/config.py` — eğitim hiperparametreleri**

```python
@dataclass(frozen=True)
class TrainingConfig:
    cv_folds: int = 5
    optuna_trials: int = 60
    top_k_for_tuning: int = 2       # screening'den kaç aday tune edilir
    min_f1_improvement: float = 0.01
    threshold_points: int = 181     # eşik optimizasyonu için adım sayısı
    search_space_version: str = "linear_v2"
```

Eğitim davranışını değiştirmek için sadece bu dosyaya dokunmak yeterlidir.

---

**`src/features.py` — feature engineering**

`add_engineered_features(df)` fonksiyonu ham DataFrame'e şu kolonları ekler:

| Kolon | Açıklama |
|-------|----------|
| `ContractMonths` | Sözleşme tipi → ay sayısı (1 / 12 / 24) |
| `ChargePerTenure` | TotalCharges / tenure (0'dan koruma var) |
| `ServiceCount` | Aktif hizmet sayısı (0–8) |
| `HasFamily` | Partner veya Dependents = "Yes" ise "Yes" |
| `IsFiberMonthToMonth` | Fiber + aylık sözleşme — yüksek churn sinyali |
| `NoSupportFiber` | Fiber + TechSupport = No |
| `HighChargeShortTenure` | MonthlyCharges ≥ 80 AND tenure ≤ 6 |

Son 3 kolon "extended features" — base features her zaman kullanılır.
`get_feature_lists(include_extended_features=False)` hangi kolonların preprocessor'a gireceğini döner.

---

**`src/preprocess.py` — veri bölme ve preprocessor fit**

- `load_raw_data()`: CSV yükler, `TotalCharges` boşluklarını → NaN, `Churn` → 0/1
- `add_engineered_features()` çağrıldıktan sonra feature listelerine göre `ColumnTransformer` fit edilir
- Preprocessor **yalnızca train verisi üzerinde** fit edilir, test ve API'de sadece `transform()` kullanılır
- `customerID` her zaman düşürülür, feature matrix'e girmez
- Çıktılar: `models/preprocessor.pkl`, `data/processed/train.csv`, `data/processed/test.csv`

---

**`src/train.py` — model seçimi ve eğitim**

Adımlar:
1. **Screening**: 4 model × 3 strateji = 12 aday, 5-fold stratified CV
2. **Tuning**: En iyi 2 aday Optuna ile tune edilir (60 deneme, TPE sampler)
3. **Threshold opt.**: OOF olasılıklarından F1-maksimum eşik bulunur
4. **Final fit**: Kazanan model tüm train seti üzerinde yeniden fit edilir

Modeller:
- `logreg_liblinear_l2` — LogisticRegression (liblinear solver)
- `logreg_saga_elasticnet` — LogisticRegression (saga + elasticnet)
- `calibrated_sgd_logloss` — SGDClassifier + CalibratedClassifierCV
- `calibrated_linear_svc` — LinearSVC + CalibratedClassifierCV

Stratejiler:
- `baseline` — sınıf ağırlıksız
- `class_weight` — `class_weight="balanced"`
- `smote` — imblearn Pipeline içinde SMOTE (sadece fold-train verisine uygulanır)

**Yeni model eklemek için:** `MODEL_NAMES` tuple'ına ekle, `_base_estimator()` ve `_suggest_params()` fonksiyonlarına ilgili case'i ekle.

Çıktılar: `models/best_model.pkl`, `models/preprocessor.pkl` (re-saved), `models/model_metadata.json`, `models/model_comparison.csv`, `reports/figures/cm_*.png`, `reports/figures/roc_*.png`

---

**`src/evaluate.py`**

```python
compute_metrics(model, X_test, y_test, threshold=0.5) → dict
# accuracy, f1, precision, recall, roc_auc

plot_confusion_matrix(model, X_test, y_test, model_name, threshold)
plot_roc(model, X_test, y_test, model_name)
# → reports/figures/ klasörüne PNG
```

---

**`src/drift.py`**

`run_drift_check()`: `data/processed/train.csv`'yi referans alır, `logs/predictions/*.jsonl` dosyalarından gelen production verisini karşılaştırır. PSI (Population Stability Index) hesaplar.

- PSI < 0.1 → stable
- PSI 0.1–0.2 → moderate_shift
- PSI > 0.2 → high_shift

Sadece tahmin logları oluştuktan sonra anlamlıdır.

---

### `api/` — FastAPI Servisi

**`api/main.py`**

- `lifespan` context manager → `predict.load_artifacts()` startup'ta çağrılır
- CORS middleware (`allow_origins=["*"]`) — dış erişim için
- Jinja2 `GET /` → `frontend/templates/index.html`
- `/static` → `frontend/static/` (CSS, JS)
- `/health` ve `/api/v1/*` router'ları

---

**`api/schemas.py`**

`CustomerFeatures`: 19 alan, Pydantic v2 `Literal` ile enum validation. Kaggle CSV'deki string değerleriyle **birebir** eşleşmelidir — değiştirirsen OHE bozulur.

```python
SeniorCitizen: int  # 0 veya 1 — diğer binary kolonların aksine string değil
```

`BatchPredictionRequest`: max 500 `CustomerFeatures` listesi.
`BatchPredictionResponse`: her satır için `row_index` + prediction + `summary.predicted_churn`.

---

**`api/routes/predict.py`**

- `load_artifacts()`: `best_model.pkl`, `preprocessor.pkl`, `model_metadata.json` yükler; `_threshold = metadata["selected_threshold"]`
- Her tahmin öncesi `add_engineered_features()` çağrılır (tek satır DataFrame)
- Her tahmin `logs/predictions/<tarih>.jsonl`'e yazılır (`append_jsonl`)
- `_threshold` eğitim sırasında optimize edilmiş değer — her yanıtta `threshold_used` olarak döner

---

**`api/routes/health.py`**

```
GET /health → {"status": "ok", "model_loaded": bool}
```

`model_loaded`: best_model.pkl + preprocessor.pkl + model_metadata.json — üçü de mevcut olmalı.

---

### `frontend/` — Web Arayüzü

FastAPI ile aynı port, aynı process. Ayrı bir server yok.

- `frontend/templates/index.html` — Jinja2 template, tek sayfa
- `frontend/static/style.css` — CSS custom properties, dark theme (`#0f1117`)
- `frontend/static/app.js` — Vanilla JS; sayfa yüklenince `/health`, form submit'te `/api/v1/predict`

Arayüz bileşenleri:
- Sol: müşteri formu (4 grup: demografik, hizmetler, sözleşme/ödeme, finansal)
- Sağ: olasılık yüzdesi + gradient gauge bar + churn/safe badge

---

### `tests/`

Tüm testler monkeypatch kullanır, veri dosyası gerektirmez:

| Dosya | Ne test eder |
|-------|-------------|
| `test_api.py` | health, single predict, batch, 422 on bad input, startup fail without metadata |
| `test_preprocess.py` | `build_preprocessor` output shape, feature coverage, engineered columns |
| `test_train.py` | `compute_metrics` keys/range, `optimize_threshold`, SMOTE pipeline, MODEL_NAMES içeriği |
| `test_drift.py` | PSI non-negative, PSI shift detection |

`test_api.py` — `DummyModel` / `DummyPreprocessor` sınıfları gerçek sklearn objelerinin interface'ini taklit eder; `monkeypatch` ile gerçek path'ler `tmp_path`'e yönlendirilir.

---

## Data Flow — Adım Adım

```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
   ↓ load_raw_data()
   │  TotalCharges: whitespace → NaN
   │  Churn: "Yes"/"No" → 1/0
   │  customerID: düşürülür
   ↓ add_engineered_features()    (4 base + 3 extended kolon eklenir)
   ↓ train_test_split(80/20, stratify=Churn, seed=42)
   ├── data/processed/train.csv   (~5634 satır)
   └── data/processed/test.csv    (~1409 satır)
   ↓ ColumnTransformer.fit(train) — SADECE train verisi üzerinde
   │  Numeric: median impute → StandardScaler
   │  Categorical: mode impute → OneHotEncoder(handle_unknown="ignore")
   ↓ models/preprocessor.pkl
   ↓ src/train.py
   │  12 aday screening (5-fold CV) → top 2 Optuna tuning → threshold opt
   │  Final fit(X_train_full)
   ↓ models/best_model.pkl  +  model_metadata.json  +  model_comparison.csv
   ↓ API request: CustomerFeatures
   │  .model_dump() → DataFrame → add_engineered_features()
   │  preprocessor.transform() → model.predict_proba()[:, 1]
   │  proba >= selected_threshold → churn_prediction
   ↓ PredictionResponse + prediction log
```

---

## Kritik Kısıtlar

1. **Preprocessor frozen**: API'de asla `fit_transform` çağrılmaz, sadece `transform`. Yeni feature eklersen pipeline'ı yeniden çalıştırman gerekir.

2. **`CustomerFeatures` Literal değerleri** Kaggle CSV'deki string'lerle birebir eşleşmeli. Değiştirirsen OHE "unknown" olarak sıfır vektörü üretir — sessiz hata.

3. **`SeniorCitizen` integer**: Diğer binary kolonlar `"Yes"/"No"` string, bu `0/1` int. `BASE_NUMERIC_FEATURES` içinde tanımlı.

4. **Docker model bağımlılığı**: `Dockerfile` `models/` klasörünü image'a kopyalar. Retrain sonrası `docker build` yeniden çalıştırılmalı.

5. **Railway deployment**: `models/*.pkl` gitignore'da. Deploy için ya model dosyalarını commit et ya da CI'da pipeline çalıştır.

---

## Genişletme Rehberi

### Yeni model ekle
`src/train.py`: `MODEL_NAMES` tuple'ına ekle → `_base_estimator()` → `_suggest_params()`

### Yeni feature ekle
1. `src/features.py`: `add_engineered_features()` içine mantık ekle
2. `src/features.py`: `BASE_NUMERIC_FEATURES` veya `BASE_CATEGORICAL_FEATURES`'a ekle
3. `api/schemas.py`: `CustomerFeatures`'a Pydantic field ekle (Literal değerleri CSV'deki ile eşleşmeli)
4. `python -m src.pipeline` ile yeniden eğit

### Classification threshold'u değiştir
`model_metadata.json` içindeki `selected_threshold` okunur — statik sabit değil. Eşiği değiştirmek için `src/train.py`'deki `optimize_threshold()` mantığını değiştir ve yeniden eğit.

### Web arayüzüne bölüm ekle
`frontend/templates/index.html` + `frontend/static/app.js` — Jinja2 template, vanilla JS. Build süreci yok.
