# Fraud Detection API

This FastAPI service is designed for the Kaggle **Credit Card Fraud Detection** dataset (`creditcard.csv`).

## What changed
- Training script now expects the Kaggle dataset columns: `Time`, `V1`-`V28`, `Amount`, `Class`
- SHAP explainability is built into the model artifact and exposed through the API
- A browser UI is included for loading a sample row, scoring JSON, or uploading a CSV row
- Render deployment config is included

## 1. Put the Kaggle CSV in place
Place `creditcard.csv` in:

```text
fraud_detection_api/data/creditcard.csv
```

Or set:

```bash
FRAUD_DATASET_PATH=/full/path/to/creditcard.csv
```

## 2. Train the model

```bash
pip install -r requirements.txt
python train_model.py
```

Artifacts created:
- `artifacts/fraud_pipeline.joblib`
- `artifacts/sample_rows.json`

## 3. Run locally

```bash
uvicorn app.main:app --reload
```

Open:
- API docs: `http://127.0.0.1:8000/docs`
- Frontend: `http://127.0.0.1:8000/`

## 4. Render deploy
Use the root `render.yaml` or create a Python Web Service with:
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Required environment variable on Render
- `FRAUD_DATASET_PATH=/opt/render/project/src/data/creditcard.csv`

Because Kaggle files are not stored in the repo in this package, upload `creditcard.csv` into your service or commit it to a private repository if licensing and storage policies allow.
