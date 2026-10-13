"""
TASK 4: FastAPI Local Inference API
Kepler Exoplanet Classification -- Explainable ML System

Loads the same saved final model and preprocessor used by the Streamlit app.
Exposes three endpoints:
  GET  /health         -- liveness check
  POST /predict        -- single candidate inference
  POST /predict/batch  -- batch inference

Run with:
  uvicorn api:app --reload --port 8000

Sample curl (single prediction):
  curl -X POST http://localhost:8000/predict \\
    -H "Content-Type: application/json" \\
    -d '{"koi_period":9.49,"koi_duration":3.45,"koi_depth":506,
         "koi_prad":2.2,"koi_impact":0.12,"koi_model_snr":45.0,
         "koi_teq":865,"koi_insol":67,"koi_steff":5440,
         "koi_slogg":4.45,"koi_srad":0.93}'
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Paths (api.py lives at project root; models/ is a sibling directory)
# ---------------------------------------------------------------------------
ROOT            = Path(__file__).resolve().parent
MODEL_PATH      = ROOT / "models" / "final" / "final_model.pkl"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.pkl"
THRESHOLD_PATH  = ROOT / "reports" / "phase_7" / "final_threshold.json"
METADATA_PATH   = ROOT / "models" / "final" / "final_model_metadata.json"

# ---------------------------------------------------------------------------
# Load model, preprocessor, threshold at startup
# ---------------------------------------------------------------------------
_model       = joblib.load(MODEL_PATH)
_preprocessor = joblib.load(PREPROCESSOR_PATH)

with open(THRESHOLD_PATH) as f:
    _threshold = float(json.load(f)["threshold"])   # 0.61

with open(METADATA_PATH) as f:
    _metadata = json.load(f)

# The exact column order the booster was trained on
_booster_cols: list[str] = _model.get_booster().feature_names

# ---------------------------------------------------------------------------
# Feature order expected as raw inputs (pre-preprocessing)
# These are the 11 KOI table columns -- order is arbitrary here because
# we build a DataFrame with column names before passing to preprocessor.
# ---------------------------------------------------------------------------
RAW_FEATURES = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_impact",
    "koi_model_snr",
    "koi_teq",
    "koi_insol",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
]

# ---------------------------------------------------------------------------
# Pydantic input model
# Validation ranges derived from the actual training data's 1st–99th percentiles
# (from reports/phase_1/feature_quantiles.csv) with generous outer padding.
# ---------------------------------------------------------------------------
class KOIFeatures(BaseModel):
    koi_period:    float = Field(..., ge=0.1,    le=2000.0,
                                  description="Orbital period (days). "
                                              "Training range: 0.24–1071")
    koi_duration:  float = Field(..., ge=0.05,   le=200.0,
                                  description="Transit duration (hours). "
                                              "Training range: 0.10–138.5")
    koi_depth:     float = Field(..., ge=0.0,    le=2_000_000.0,
                                  description="Transit depth (ppm). "
                                              "Training range: 0.8–1.54M")
    koi_prad:      float = Field(..., ge=0.0,    le=500_000.0,
                                  description="Planet radius (Earth radii). "
                                              "Training range: 0.08–200346")
    koi_impact:    float = Field(..., ge=0.0,    le=200.0,
                                  description="Impact parameter. "
                                              "Training range: 0.0–100.8")
    koi_model_snr: float = Field(..., ge=0.0,    le=20_000.0,
                                  description="Transit signal-to-noise ratio. "
                                              "Training range: 0.0–9054.7")
    koi_teq:       float = Field(..., ge=50.0,   le=20_000.0,
                                  description="Equilibrium temperature (K). "
                                              "Training range: 92–14667")
    koi_insol:     float = Field(..., ge=0.0,    le=15_000_000.0,
                                  description="Insolation flux (Earth flux). "
                                              "Training range: 0.02–10.9M")
    koi_steff:     float = Field(..., ge=2000.0, le=20_000.0,
                                  description="Stellar effective temperature (K). "
                                              "Training range: 2661–15896")
    koi_slogg:     float = Field(..., ge=0.0,    le=6.0,
                                  description="Stellar surface gravity (log g). "
                                              "Training range: 0.047–5.283")
    koi_srad:      float = Field(..., ge=0.05,   le=300.0,
                                  description="Stellar radius (solar radii). "
                                              "Training range: 0.116–229.9")

    model_config = {"json_schema_extra": {
        "example": {
            "koi_period":    9.49,
            "koi_duration":  3.45,
            "koi_depth":     506.0,
            "koi_prad":      2.2,
            "koi_impact":    0.12,
            "koi_model_snr": 45.0,
            "koi_teq":       865.0,
            "koi_insol":     67.0,
            "koi_steff":     5440.0,
            "koi_slogg":     4.45,
            "koi_srad":      0.93,
        }
    }}


class PredictionResult(BaseModel):
    prediction:    str   # "CONFIRMED" or "FALSE POSITIVE"
    probability:   float
    threshold_used: float


# ---------------------------------------------------------------------------
# Core inference logic (shared by single and batch)
# ---------------------------------------------------------------------------
def _run_inference(features_list: list[dict]) -> list[PredictionResult]:
    """
    Apply the saved preprocessor then the final model.
    Reorders columns to match the booster's exact feature order.
    """
    df = pd.DataFrame(features_list, columns=RAW_FEATURES)

    # 1. Preprocess (imputation + log1p on 4 features + StandardScaler)
    X_proc = _preprocessor.transform(df)

    # 2. Re-wrap as DataFrame with column names the booster knows
    X_proc_df = pd.DataFrame(X_proc, columns=_booster_cols)

    # 3. Explicit column order guard
    X_ordered = X_proc_df[_booster_cols]

    # 4. Predict
    proba = _model.predict_proba(X_ordered)[:, 1]
    labels = np.where(proba >= _threshold, "CONFIRMED", "FALSE POSITIVE")

    return [
        PredictionResult(
            prediction=str(label),
            probability=float(p),
            threshold_used=_threshold,
        )
        for label, p in zip(labels, proba)
    ]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Kepler Exoplanet Classification API",
    description=(
        "Local inference API for the Kepler KOI XGBoost classifier. "
        "Classifies a Kepler Object of Interest as **CONFIRMED** exoplanet or "
        "**FALSE POSITIVE** using 11 stellar and transit features."
    ),
    version="1.0.0",
)


@app.get("/health", summary="Liveness check")
def health():
    """Returns service status and model name."""
    return {
        "status": "ok",
        "model":  "XGBoost",
        "threshold": _threshold,
        "model_version": _metadata.get("version", "1.0"),
    }


@app.post("/predict", response_model=PredictionResult,
          summary="Single candidate prediction")
def predict(features: KOIFeatures):
    """
    Classify a single Kepler Object of Interest.

    Returns the predicted disposition (CONFIRMED or FALSE POSITIVE),
    the model's probability score, and the decision threshold used.
    """
    try:
        results = _run_inference([features.model_dump()])
        return results[0]
    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"Inference error: {str(exc)}") from exc


@app.post("/predict/batch", response_model=List[PredictionResult],
          summary="Batch candidate prediction")
def predict_batch(candidates: List[KOIFeatures]):
    """
    Classify a list of Kepler Objects of Interest in one call.

    Returns a list of prediction objects in the same order as the input.
    Maximum recommended batch size: 10,000 candidates.
    """
    if len(candidates) == 0:
        raise HTTPException(status_code=422, detail="Batch must not be empty.")
    if len(candidates) > 10_000:
        raise HTTPException(status_code=422,
                            detail="Batch size exceeds maximum (10,000).")
    try:
        records = [c.model_dump() for c in candidates]
        return _run_inference(records)
    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"Batch inference error: {str(exc)}") from exc
