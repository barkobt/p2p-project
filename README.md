# Telco Customer Churn — ML + REST API

Telekom müşterilerinin hizmeti bırakıp bırakmayacağını tahmin eden uçtan uca makine öğrenmesi sistemi. Veri ön işlemeden model eğitimine, HTTP API servisinden Docker dağıtımına kadar eksiksiz bir pipeline içerir.

---

## İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Veri Seti](#veri-seti)
- [Kurulum](#kurulum)
- [Pipeline Çalıştırma](#pipeline-çalıştırma)
- [API Kullanımı](#api-kullanımı)
- [Model Karşılaştırması](#model-karşılaştırması)
- [Docker](#docker)
- [Testler](#testler)
- [Proje Yapısı](#proje-yapısı)
- [Yol Haritası](#yol-haritası)

---

## Proje Hakkında

Bu proje, bir telekom şirketinin müşteri verilerini kullanarak hangi müşterilerin hizmeti bırakabileceğini (churn) tahmin eder. Sistem üç ana katmandan oluşur:

- **Veri Katmanı:** Ham CSV → temizleme → eğitim/test bölme
- **Model Katmanı:** Linear model ailesi ile CV + Optuna tuning ve otomatik en iyi model seçimi
- **Servis Katmanı:** FastAPI tabanlı REST API, `/predict` endpoint'i üzerinden gerçek zamanlı tahmin

### Teknik Stack

| Katman | Teknoloji |
|--------|-----------|
| ML Pipeline | scikit-learn, imbalanced-learn, Optuna |
| API | FastAPI + Uvicorn |
| Veri İşleme | pandas, numpy |
| Serileştirme | joblib |
| Test | pytest + httpx |
| Konteyner | Docker + Docker Compose |

---

## Veri Seti

[Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

IBM örnek verisi olan bu set, 7043 müşterinin demografik, hizmet ve sözleşme bilgilerini içerir. Hedef değişken `Churn`, müşterinin son ay içinde hizmeti bırakıp bırakmadığını gösterir.

**Önemli Veri Notları:**
- `TotalCharges` sütunu yeni müşterilerde (tenure=0) boşluk içerir → `NaN` olarak işlenir, medyan ile doldurulur
- `SeniorCitizen` 0/1 integer formatındadır, diğer kategorikler "Yes"/"No" string'idir
- Sınıf dengesizliği mevcuttur (~26% churn, ~74% non-churn)

CSV dosyasını indirip şu konuma yerleştir:

```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## Kurulum

Python 3.11+ önerilir.

```bash
# Sanal ortam oluştur
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Bağımlılıkları kur
pip install -r requirements.txt
```

---

## Pipeline Çalıştırma

Adımlar sırasıyla çalıştırılmalıdır; her adım bir sonrakinin girişini üretir.

```bash
# Adım 1 — Veri ön işleme
# Çıktı: models/preprocessor.pkl, data/processed/train.csv, data/processed/test.csv
python -m src.preprocess

# Adım 2 — Model eğitimi ve karşılaştırma
# Çıktı:
#   models/best_model.pkl
#   models/model_metadata.json
#   models/model_comparison.csv
#   reports/figures/*.png
python -m src.train

# Adım 3 — API servisi
uvicorn api.main:app --reload --port 8000
```

Tek komutla local pipeline:

```bash
python -m src.pipeline
# veya
bash scripts/run_local_pipeline.sh
```

### Keşifsel Veri Analizi (EDA)

```bash
jupyter notebook notebooks/01_eda.ipynb
```

EDA notebook'u churn dağılımı, sayısal değişken histogramları, kategorik değişkenler bazında churn oranları ve korelasyon matrisini otomatik olarak `reports/figures/` klasörüne kaydeder.

### Drift Kontrolü (PSI)

Prediction logları oluştuktan sonra:

```bash
python -m src.drift
```

Çıktı `reports/drift/` klasörüne JSON raporu olarak yazılır.

---

## API Kullanımı

API başlatıldıktan sonra interactive dokümantasyon için: **http://localhost:8000/docs**

### Endpoint'ler

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| `GET` | `/health` | Servis sağlığı ve model yüklü olup olmadığını döner |
| `POST` | `/api/v1/predict` | Müşteri bilgilerini alır, churn olasılığı döner |
| `POST` | `/api/v1/predict/batch` | Tek istekte en fazla 500 müşteri için tahmin döner |

### POST `/api/v1/predict`

**İstek Gövdesi:**

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.5,
  "TotalCharges": 1020.0
}
```

**Yanıt:**

```json
{
  "churn_probability": 0.7342,
  "churn_prediction": true,
  "threshold_used": 0.42
}
```

`threshold_used`, eğitim sırasında validation üzerinde optimize edilip `models/model_metadata.json` içine yazılan değerdir.

**Alan Açıklamaları:**

| Alan | Tip | Kabul Edilen Değerler |
|------|-----|----------------------|
| `gender` | string | `"Male"`, `"Female"` |
| `SeniorCitizen` | int | `0`, `1` |
| `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` | string | `"Yes"`, `"No"` |
| `MultipleLines` | string | `"Yes"`, `"No"`, `"No phone service"` |
| `InternetService` | string | `"DSL"`, `"Fiber optic"`, `"No"` |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | string | `"Yes"`, `"No"`, `"No internet service"` |
| `Contract` | string | `"Month-to-month"`, `"One year"`, `"Two year"` |
| `PaymentMethod` | string | `"Electronic check"`, `"Mailed check"`, `"Bank transfer (automatic)"`, `"Credit card (automatic)"` |
| `MonthlyCharges` | float | > 0 |
| `TotalCharges` | float | ≥ 0 |

### cURL Örneği

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d @- << 'EOF'
{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
  "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
  "MultipleLines": "No", "InternetService": "Fiber optic",
  "OnlineSecurity": "No", "OnlineBackup": "No",
  "DeviceProtection": "No", "TechSupport": "No",
  "StreamingTV": "Yes", "StreamingMovies": "Yes",
  "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.5, "TotalCharges": 1020.0
}
EOF
```

---

## Model Karşılaştırması

`python -m src.train` akışı:

- Stratified 5-fold CV ile model/strateji karşılaştırması yapar
- Linear family adayları için `baseline`, `class_weight`, `smote` stratejilerini dener
- İlk 2 adayı Optuna ile (`60 trial`) tune eder
- Out-of-fold olasılıklardan en iyi F1 threshold'unu seçer
- Sonuçları `models/model_comparison.csv` dosyasına yazar

Confusion matrix ve ROC eğrisi grafikleri otomatik olarak `reports/figures/` klasörüne kaydedilir.

---

## Docker

Modeller eğitildikten sonra Docker ile çalıştırılabilir. `models/` klasörü build sırasında imaja kopyalanır.

```bash
# Önce modelleri eğit
python -m src.pipeline

# Image oluştur ve başlat
docker compose up --build

# Arka planda çalıştırmak için
docker compose up -d --build
```

Servis `http://localhost:8000` adresinde erişilebilir olur.

---

## Testler

```bash
# Tüm testler
pytest tests/ -v

# Unit testler
pytest tests/test_preprocess.py tests/test_train.py tests/test_drift.py -v

# API smoke testleri
pytest tests/test_api.py -v
```

---

## Proje Yapısı

```
p2p-project/
│
├── data/
│   ├── raw/                    ← Kaggle CSV buraya (gitignore'd)
│   └── processed/              ← train.csv, test.csv (preprocess sonrası üretilir)
│
├── notebooks/
│   └── 01_eda.ipynb            ← Keşifsel veri analizi
│
├── src/
│   ├── utils.py                ← Path sabitleri (ROOT_DIR, DATA_DIR, MODEL_DIR), logger
│   ├── features.py             ← Ortak feature engineering dönüşümleri
│   ├── preprocess.py           ← ColumnTransformer pipeline + veri bölme
│   ├── train.py                ← CV + dengesizlik stratejileri + Optuna + artefakt üretimi
│   ├── pipeline.py             ← Tek komutluk preprocess + train akışı
│   ├── drift.py                ← Prediction logları üzerinden PSI drift raporu
│   └── evaluate.py             ← compute_metrics, plot_roc, plot_confusion_matrix
│
├── api/
│   ├── main.py                 ← FastAPI app + lifespan (model yükleme)
│   ├── schemas.py              ← CustomerFeatures + batch request/response şemaları
│   └── routes/
│       ├── predict.py          ← /predict ve /predict/batch + JSONL prediction logging
│       └── health.py           ← GET /health
│
├── models/                     ← best_model.pkl, preprocessor.pkl, model_metadata.json, model_comparison.csv
├── tests/                      ← pytest testleri
├── reports/figures/            ← Otomatik kaydedilen grafikler
├── reports/drift/              ← Drift raporları
├── logs/predictions/           ← Günlük JSONL prediction logları
├── mlruns/                     ← MLflow local file store
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── CLAUDE.md                   ← Claude Code için kılavuz
```

---

## Yol Haritası

Projenin ilerleyen versiyonlarında planlanıyor:

- [x] **SMOTE / sınıf ağırlığı** — sınıf dengesizliği için karşılaştırmalı strateji
- [x] **Hiperparametre optimizasyonu** — Optuna ile otomatik tuning
- [ ] **Feature importance görselleştirme** — SHAP değerleri ile model yorumlanabilirliği
- [x] **Batch predict endpoint** — tek seferde çok sayıda müşteri tahmini (`POST /api/v1/predict/batch`)
- [x] **Model versiyonlama** — MLflow local store ile deney takibi
- [ ] **Streamlit arayüzü** — sürükle-bırak CSV yükleme ve tahmin dashboard'u
- [x] **CI/CD** — GitHub Actions ile lint + unit + API smoke test
- [x] **Monitoring** — prediction logları ve PSI tabanlı drift raporu
