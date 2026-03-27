from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / 'artifacts'
MODEL_PATH = ARTIFACT_DIR / 'fraud_pipeline.joblib'
SAMPLES_PATH = ARTIFACT_DIR / 'sample_rows.json'
TEMPLATES_DIR = BASE_DIR / 'app' / 'templates'
STATIC_DIR = BASE_DIR / 'app' / 'static'

FEATURE_NAMES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']


class FraudRequest(BaseModel):
    Time: float = Field(..., description='Seconds elapsed between this transaction and the first transaction in the dataset')
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0, description='Transaction amount')

class PredictionResponse(BaseModel):
    fraud_probability: float
    predicted_label: str
    risk_level: str
    shap_top_features: list[dict[str, Any]]


app = FastAPI(
    title='Fraud Detection API',
    version='2.0.0',
    description='Fraud scoring service trained for the Kaggle credit card fraud dataset with SHAP explainability and a browser UI.',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

_model_bundle: dict[str, Any] | None = None


def load_bundle() -> dict[str, Any]:
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f'Model artifact not found at {MODEL_PATH}. Run train_model.py after placing the Kaggle creditcard.csv file in the project.'
            )
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


def payload_to_frame(payload: FraudRequest) -> pd.DataFrame:
    row = payload.model_dump()
    return pd.DataFrame([[row[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)


def compute_prediction(df: pd.DataFrame) -> PredictionResponse:
    bundle = load_bundle()
    model = bundle['model']

    probability = float(model.predict_proba(df)[0, 1])
    label = 'fraud' if probability >= bundle.get('threshold', 0.5) else 'legitimate'
    risk = 'high' if probability >= 0.8 else 'medium' if probability >= 0.5 else 'low'

    explainer = shap.Explainer(model)
    shap_values = explainer(df)
    shap_array = shap_values.values
    if shap_array.ndim == 3:
        shap_for_class = shap_array[0, : , 1]
    else:
        shap_for_class = shap_array[0]

    feature_pairs = [
        {
            'feature': feature,
            'value': float(df.iloc[0][feature]),
            'shap_value': float(shap_val),
        }
        for feature, shap_val in zip(FEATURE_NAMES, shap_for_class)
    ]
    feature_pairs.sort(key=lambda item: abs(item['shap_value']), reverse=True)

    return PredictionResponse(
        fraud_probability=round(probability, 6),
        predicted_label=label,
        risk_level=risk,
        shap_top_features=feature_pairs[:5],
    )


@app.on_event('startup')
def startup_event() -> None:
    if MODEL_PATH.exists():
        load_bundle()


@app.get('/', response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse('index.html', {'request': request, 'feature_names': FEATURE_NAMES})


@app.get('/health')
def health() -> dict[str, Any]:
    return {
        'status': 'ok',
        'model_loaded': MODEL_PATH.exists(),
        'dataset_expected': 'creditcard.csv',
        'feature_count': len(FEATURE_NAMES),
    }


@app.get('/sample')
def sample() -> dict[str, Any]:
    if not SAMPLES_PATH.exists():
        raise HTTPException(status_code=500, detail='Sample rows not found. Train the model first.')
    rows = json.loads(SAMPLES_PATH.read_text())
    idx = int(np.random.randint(0, len(rows)))
    return {'sample_index': idx, 'payload': rows[idx]}


@app.post('/predict', response_model=PredictionResponse)
def predict(payload: FraudRequest) -> PredictionResponse:
    try:
        return compute_prediction(payload_to_frame(payload))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post('/explain', response_model=PredictionResponse)
def explain(payload: FraudRequest) -> PredictionResponse:
    return predict(payload)


@app.post('/predict-csv')
async def predict_csv(file: UploadFile = File(...), row_index: int = Form(0)) -> JSONResponse:
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail='Upload a CSV file.')
    content = await file.read()
    df = pd.read_csv(pd.io.common.BytesIO(content))
    missing = [col for col in FEATURE_NAMES if col not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f'Missing required columns: {missing}')
    if row_index < 0 or row_index >= len(df):
        raise HTTPException(status_code=400, detail='row_index out of range')
    prediction = compute_prediction(df.loc[[row_index], FEATURE_NAMES].copy())
    return JSONResponse({'row_index': row_index, **prediction.model_dump()})


if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv('PORT', '8000'))
    uvicorn.run('app.main:app', host='0.0.0.0', port=port, reload=True)
