from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / 'artifacts'
ARTIFACT_DIR.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACT_DIR / 'fraud_pipeline.joblib'
SAMPLES_PATH = ARTIFACT_DIR / 'sample_rows.json'
DATASET_PATH = Path(os.getenv('FRAUD_DATASET_PATH', BASE_DIR / 'data' / 'creditcard.csv'))
FEATURE_NAMES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
TARGET = 'Class'


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f'Could not find {DATASET_PATH}. Download creditcard.csv from Kaggle and place it there, or set FRAUD_DATASET_PATH.'
        )
    df = pd.read_csv(DATASET_PATH)
    missing = [c for c in FEATURE_NAMES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f'Dataset missing expected columns: {missing}')
    return df


def main() -> None:
    df = load_dataset()
    X = df[FEATURE_NAMES].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, digits=4))
    print(f'ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}')
    print(f'PR-AUC: {average_precision_score(y_test, y_prob):.4f}')

    background = shap.sample(X_train, 200, random_state=42)
    explainer = shap.TreeExplainer(model, data=background, feature_perturbation='interventional')

    sample_rows = X_test.head(20).to_dict(orient='records')
    SAMPLES_PATH.write_text(json.dumps(sample_rows, indent=2))

    joblib.dump(
        {
            'model': model,
            'explainer': explainer,
            'feature_names': FEATURE_NAMES,
            'threshold': 0.5,
            'dataset_path': str(DATASET_PATH),
        },
        MODEL_PATH,
    )
    print(f'Saved model bundle to {MODEL_PATH}')
    print(f'Saved example rows to {SAMPLES_PATH}')


if __name__ == '__main__':
    main()
