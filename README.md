# Telco Customer Churn — ML + REST API

Telekom müşterilerinin hizmeti bırakıp bırakmayacağını tahmin eden uçtan uca bir makine öğrenmesi sistemi.

## Veri Seti

[Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

CSV dosyasını `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` yoluna yerleştir.

---

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Pipeline Çalıştırma

```bash
# 1. Veri ön işleme (preprocessor.pkl + train/test CSV üretir)
python -m src.preprocess

# 2. Model eğitimi (4 model karşılaştırır, best_model.pkl kaydeder)
python -m src.train

# 3. API'yi başlat
uvicorn api.main:app --reload --port 8000
```

### EDA Notebook

```bash
jupyter notebook notebooks/01_eda.ipynb
```

---

## API Referansı

| Method | Endpoint          | Açıklama                          |
|--------|-------------------|-----------------------------------|
| GET    | `/health`         | Servis sağlığı + model durumu     |
| POST   | `/api/v1/predict` | Churn olasılığı tahmini           |

**Swagger UI:** http://localhost:8000/docs

### Örnek İstek

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Örnek Yanıt

```json
{
  "churn_probability": 0.7342,
  "churn_prediction": true,
  "threshold_used": 0.5
}
```

---

## Model Karşılaştırması

| Model              | Accuracy | F1   | ROC-AUC |
|--------------------|----------|------|---------|
| LightGBM           | —        | —    | —       |
| XGBoost            | —        | —    | —       |
| RandomForest       | —        | —    | —       |
| LogisticRegression | —        | —    | —       |

> Eğitim tamamlandıktan sonra `python -m src.train` çıktısından doldur.

---

## Docker

```bash
# Önce modelleri eğit (models/ klasörü gerekli)
python -m src.preprocess && python -m src.train

# Docker ile başlat
docker compose up --build
```

---

## Testler

```bash
pytest tests/ -v
```

> `test_api.py` çalışması için modellerin eğitilmiş olması gerekir.

---

## Proje Yapısı

```
p2p-project/
├── data/            # Ham ve işlenmiş veri
├── notebooks/       # EDA notebook
├── src/             # ML pipeline (preprocess, train, evaluate)
├── api/             # FastAPI servisi
├── models/          # Eğitilmiş model dosyaları (.pkl)
├── tests/           # Birim ve entegrasyon testleri
└── reports/figures/ # Otomatik kaydedilen grafikler
```
