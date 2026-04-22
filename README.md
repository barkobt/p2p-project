# Telco Customer Churn — ML + Web Servisi

Telekom müşterilerinin hizmeti bırakıp bırakmayacağını tahmin eden uçtan uca sistem.
Veri ön işleme → model eğitimi → REST API → web arayüzü tek pakette.

---

## İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Veri Seti](#veri-seti)
- [Kurulum](#kurulum)
- [Pipeline Çalıştırma](#pipeline-çalıştırma)
- [Web Arayüzü ve API](#web-arayüzü-ve-api)
- [Web'den Erişim (Başkaları da Erişebilsin)](#webden-erişim)
- [Docker ile Çalıştırma](#docker-ile-çalıştırma)
- [Testler](#testler)
- [Proje Yapısı](#proje-yapısı)
- [Model Karşılaştırması](#model-karşılaştırması)
- [Yol Haritası](#yol-haritası)

---

## Proje Hakkında

Sistem üç ana katmandan oluşur:

| Katman | Ne yapar |
|--------|---------|
| **ML Pipeline** (`src/`) | Ham CSV'yi işler, 4 model × 3 strateji × Optuna tuning ile eğitir |
| **API** (`api/`) | FastAPI; tek tahmin, toplu tahmin, sağlık endpoint'leri |
| **Web Arayüzü** (`frontend/`) | Aynı porttan servis edilen sade, koyu temalı tahmin formu |

### Teknik Stack

| | |
|---|---|
| ML | scikit-learn, imbalanced-learn (SMOTE), Optuna |
| API & UI | FastAPI, Uvicorn, Jinja2 |
| Veri | pandas, numpy |
| Test | pytest (20 test, veri dosyası gerektirmez) |
| Konteyner | Docker (opsiyonel) |

---

## Veri Seti

[Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

7 043 müşteri kaydı. Hedef: `Churn` (son ay hizmeti bırakma).

**Önemli notlar:**
- `TotalCharges` — yeni müşterilerde boşluk → `NaN` → medyan ile doldurulur
- `SeniorCitizen` — `0/1` integer (diğer kolonlar `"Yes"/"No"` string)
- Sınıf dengesizliği: ~%26 churn / ~%74 non-churn → SMOTE + ağırlıklı eğitim ile giderilir

```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## Kurulum

Python **3.11+** önerilir.

```bash
git clone https://github.com/barkobt/p2p-project.git
cd p2p-project

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Pipeline Çalıştırma

> Adımlar **sırasıyla** çalıştırılmalı; her adım bir sonrakinin girişini üretir.

### Hızlı başlangıç (tek komut)

```bash
python -m src.pipeline
# Sırasıyla preprocess → train çalıştırır
```

### Adım adım

```bash
# 1. Veri ön işleme
#    Çıktı: models/preprocessor.pkl
#           data/processed/train.csv
#           data/processed/test.csv
python -m src.preprocess

# 2. Model eğitimi
#    Çıktı: models/best_model.pkl
#           models/model_metadata.json   (eşik, metrikler, model adı)
#           models/model_comparison.csv  (tüm aday karşılaştırması)
#           reports/figures/             (ROC + confusion matrix grafikleri)
python -m src.train

# 3. Servisi başlat
uvicorn api.main:app --reload --port 8000
```

### EDA Notebook

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Drift Kontrolü (opsiyonel)

Tahmin logları biriktikten sonra:

```bash
python -m src.drift
# → reports/drift/drift_report_<tarih>.json
```

---

## Web Arayüzü ve API

Servis başladıktan sonra aynı port üzerinden her şeye erişilir:

| URL | Ne |
|-----|----|
| `http://localhost:8000/` | **Web Arayüzü** — tahmin formu |
| `http://localhost:8000/docs` | Swagger / interaktif API dökümantasyonu |
| `http://localhost:8000/health` | Servis sağlığı |
| `POST /api/v1/predict` | Tek müşteri tahmini |
| `POST /api/v1/predict/batch` | Toplu tahmin (max 500 müşteri) |

### POST `/api/v1/predict` — Örnek İstek

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0,
    "Partner": "Yes", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5, "TotalCharges": 1020.0
  }'
```

### Örnek Yanıt

```json
{
  "churn_probability": 0.7342,
  "churn_prediction": true,
  "threshold_used": 0.42
}
```

`threshold_used` — eğitim sırasında F1 optimizasyonu ile bulunan değer, her yanıtta döner.

---

## Web'den Erişim

> **Soru: Başkaları da bu servise erişebilir mi?**
> Evet — aşağıdaki yöntemlerden biri ile.

---

### Seçenek A — Geçici Paylaşım: ngrok *(sunum/demo için)*

Kendi bilgisayarınızdan çalışırken herkese açık geçici URL alırsınız.

```bash
# ngrok kurulumu: https://ngrok.com/download
# Kurulumdan sonra:

uvicorn api.main:app --port 8000 &   # API'yi arka planda başlat
ngrok http 8000                       # Genel URL oluştur
```

Çıktı şöyle görünür:

```
Forwarding  https://abc123.ngrok-free.app  →  http://localhost:8000
```

`https://abc123.ngrok-free.app` adresini istediğiniz kişiyle paylaşın.
ngrok penceresi açık kaldığı sürece erişim aktif olur.

---

### Seçenek B — Kalıcı Deployment: Railway *(ücretsiz, önerilen)*

GitHub'a push edilen kod otomatik deploy olur.

**Adımlar:**

```
1. railway.app → "New Project" → "Deploy from GitHub repo"
2. barkobt/p2p-project reposunu seç
3. Settings → Start Command:
      uvicorn api.main:app --host 0.0.0.0 --port $PORT
4. "Deploy" tıkla → ~2-3 dakika sonra URL hazır
   Örnek: https://p2p-project-production.up.railway.app
```

**⚠️ Önemli — Model dosyaları:**
`models/*.pkl` ve `models/*.json` gitignore'da, bu yüzden Railway'de model bulunamaz.
Çözüm: Deployment öncesinde `.gitignore`'dan şu satırı kaldır, model dosyalarını commit et:

```
# Bu satırı sil veya yorum yap:
# models/*.pkl
```

```bash
python -m src.pipeline          # modelleri train et
git add models/                 # commit'e ekle
git commit -m "chore: add trained model artifacts"
git push                        # Railway otomatik redeploy yapar
```

---

### Seçenek C — Docker ile (VPS / Render.com)

```bash
# Önce modelleri eğit
python -m src.pipeline

# Docker image oluştur ve çalıştır
docker build -t churn-api .
docker run -p 8000:8000 churn-api
# → http://localhost:8000
```

**Render.com deployment:**

```
render.com → "New Web Service" → GitHub repoyu seç
Environment: Docker
Deploy → URL otomatik atanır
```

---

## Docker ile Çalıştırma

```bash
python -m src.pipeline          # önce model eğit
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

---

## Testler

Tüm testler veri dosyası **gerektirmez** (monkeypatch ile izole edilmiştir).

```bash
pytest tests/ -v                  # 20 test, hepsi çalışır

pytest tests/test_api.py -v       # API + batch + startup failure
pytest tests/test_preprocess.py   # Feature engineering, preprocessor
pytest tests/test_train.py        # Metrikler, eşik optimizasyonu, SMOTE
pytest tests/test_drift.py        # PSI hesaplama
```

---

## Proje Yapısı

```
p2p-project/
│
├── data/
│   ├── raw/                  ← Kaggle CSV (gitignore'd)
│   └── processed/            ← train.csv, test.csv (pipeline çıktısı)
│
├── src/
│   ├── config.py             ← Eğitim hiperparametreleri (TrainingConfig dataclass)
│   ├── features.py           ← Feature engineering (ContractMonths, ChargePerTenure, ...)
│   ├── preprocess.py         ← ColumnTransformer pipeline + veri bölme
│   ├── train.py              ← 4 model × 3 strateji, Optuna tuning, en iyi modeli seç
│   ├── evaluate.py           ← compute_metrics, ROC/CM grafikleri
│   ├── pipeline.py           ← preprocess + train tek komutla
│   ├── drift.py              ← PSI tabanlı veri kayması raporu (opsiyonel)
│   └── utils.py              ← Path sabitleri, logger, append_jsonl
│
├── api/
│   ├── main.py               ← FastAPI app, CORS, Jinja2, static mount
│   ├── schemas.py            ← CustomerFeatures, PredictionResponse, Batch...
│   └── routes/
│       ├── predict.py        ← POST /api/v1/predict  ve  /predict/batch
│       └── health.py         ← GET /health
│
├── frontend/
│   ├── templates/
│   │   └── index.html        ← Tek sayfa, koyu temalı tahmin formu
│   └── static/
│       ├── style.css         ← CSS custom properties, dark theme
│       └── app.js            ← Vanilla JS, fetch → sonuç paneli
│
├── models/                   ← .pkl + .json (gitignore'd, pipeline üretir)
├── notebooks/
│   └── 01_eda.ipynb
├── tests/                    ← 20 pytest testi
├── reports/figures/          ← Grafik çıktıları
├── requirements.txt
├── Dockerfile
└── CLAUDE.md
```

---

## Model Karşılaştırması

`python -m src.train` çıktısı eğitim sonrasında doldurulacak.

Eğitim süreci:
1. 4 model × 3 strateji (baseline, class_weight, SMOTE) = 12 aday, 5-katlı CV
2. En iyi 2 aday Optuna ile (60 deneme) tune edilir
3. OOF olasılıklarından en iyi F1 eşiği seçilir
4. Sonuçlar `models/model_comparison.csv`'ye yazılır

| Model | Strateji | CV F1 | CV ROC-AUC | Eşik |
|-------|----------|-------|------------|------|
| — | — | — | — | — |

> Eğitim tamamlandıktan sonra `models/model_comparison.csv` veya terminal çıktısından doldur.

---

## Yol Haritası

- [x] Veri ön işleme pipeline'ı
- [x] Feature engineering (ContractMonths, ChargePerTenure, ServiceCount, HasFamily)
- [x] Çoklu model eğitimi + Optuna tuning
- [x] SMOTE ile sınıf dengesizliği giderme
- [x] FastAPI REST servisi (tek + toplu tahmin)
- [x] Metadata-tabanlı dinamik threshold
- [x] Prediction logging
- [x] Web arayüzü (koyu tema, probability gauge)
- [x] CORS — dış erişime açık
- [x] PSI tabanlı drift detection
- [ ] SHAP değerleri ile model yorumlanabilirliği
- [ ] CI/CD (GitHub Actions — push'ta otomatik test)
