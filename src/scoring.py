import json

import joblib
import pandas as pd

from config import META_PATH, MODEL_PATH, RAW_COLUMNS
from explain import field_contributions


def load_artifacts():
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())


def score_transaction(pipeline, meta, row: dict, threshold=None):
    threshold = meta["threshold"] if threshold is None else threshold
    proba = float(pipeline.predict_proba(pd.DataFrame([row])[RAW_COLUMNS])[0, 1])
    contrib = field_contributions(pipeline, row, meta["reference_row"])
    return {
        "probability": proba,
        "label": "High Risk" if proba >= threshold else "Low Risk",
        "threshold": threshold,
        "contributions": contrib,
    }
