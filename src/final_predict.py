"""
final_predict.py
Kepler Exoplanet Classification -- Explainable ML System

Exposes predict_exoplanet(X) for production-style inference.

Usage:
    from src.final_predict import predict_exoplanet, load_pipeline
    result = predict_exoplanet(X_df)

The function:
  1. Validates required features are present.
  2. Enforces feature order.
  3. Loads models/final/final_model.pkl.
  4. Generates predicted probability.
  5. Applies the frozen threshold from models/final/final_model_metadata.json.
  6. Returns a DataFrame with prediction, probability, and threshold.

Note: Predictions are machine-learning outputs, not scientific confirmations.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to project root, resolved from this file's location)
# ---------------------------------------------------------------------------
_SRC_DIR    = Path(__file__).resolve().parent
_ROOT       = _SRC_DIR.parent
_MODEL_PATH = _ROOT / "models" / "final" / "final_model.pkl"
_META_PATH  = _ROOT / "models" / "final" / "final_model_metadata.json"

# ---------------------------------------------------------------------------
# Required feature order
# ---------------------------------------------------------------------------
REQUIRED_FEATURES = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_impact", "koi_model_snr", "koi_teq", "koi_insol",
    "koi_steff", "koi_slogg", "koi_srad",
]


def load_pipeline():
    """
    Load the final model and threshold from disk.
    Returns (model, threshold, metadata_dict).
    """
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model not found: {_MODEL_PATH}\n"
            "Run src/phase_7_finalization.py first."
        )
    if not _META_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {_META_PATH}"
        )

    model = joblib.load(_MODEL_PATH)
    with open(_META_PATH) as fh:
        meta = json.load(fh)

    threshold = float(meta["threshold"])
    return model, threshold, meta


def predict_exoplanet(X: pd.DataFrame) -> pd.DataFrame:
    """
    Classify KOI objects as CONFIRMED or FALSE POSITIVE.

    Parameters
    ----------
    X : pd.DataFrame
        Input features. Must contain all 11 required columns
        (see REQUIRED_FEATURES). Additional columns are ignored.
        Must already be preprocessed (imputed, log-transformed, scaled)
        using the Phase 2 preprocessor (models/preprocessor.pkl).

    Returns
    -------
    pd.DataFrame with columns:
        - predicted_class   : "CONFIRMED" or "FALSE_POSITIVE"
        - predicted_probability : P(CONFIRMED), float in [0, 1]
        - threshold         : frozen classification threshold
    """
    # --- 1. Validate required features ---
    missing = [f for f in REQUIRED_FEATURES if f not in X.columns]
    if missing:
        raise ValueError(
            f"Missing required feature(s): {missing}\n"
            f"Required: {REQUIRED_FEATURES}"
        )

    # --- 2. Enforce feature order ---
    X_ordered = X[REQUIRED_FEATURES].copy()

    # --- 3. Load pipeline ---
    model, threshold, meta = load_pipeline()

    # --- 4. Generate probabilities ---
    y_proba = model.predict_proba(X_ordered)[:, 1]

    # Validate
    assert not np.any(np.isnan(y_proba)), "NaN in predicted probabilities"
    assert float(y_proba.min()) >= 0.0 and float(y_proba.max()) <= 1.0

    # --- 5. Apply threshold ---
    y_pred = (y_proba >= threshold).astype(int)
    labels = ["CONFIRMED" if p == 1 else "FALSE_POSITIVE" for p in y_pred]

    # --- 6. Return results ---
    return pd.DataFrame({
        "predicted_class":       labels,
        "predicted_probability": y_proba.round(4),
        "threshold":             threshold,
    }, index=X.index)


def predict_from_raw(X_raw: pd.DataFrame,
                     preprocessor_path: str = None) -> pd.DataFrame:
    """
    End-to-end prediction from raw (un-preprocessed) KOI features.
    Applies the Phase 2 preprocessor then runs predict_exoplanet.

    Parameters
    ----------
    X_raw : pd.DataFrame
        Raw KOI feature columns (un-imputed, un-scaled).
    preprocessor_path : str or None
        Path to models/preprocessor.pkl. Defaults to the standard location.

    Returns
    -------
    pd.DataFrame (same as predict_exoplanet)
    """
    import joblib as _joblib
    import json as _json

    if preprocessor_path is None:
        preprocessor_path = str(_ROOT / "models" / "preprocessor.pkl")

    preprocessor = _joblib.load(preprocessor_path)

    with open(_META_PATH) as fh:
        meta = _json.load(fh)
    features = meta["features"]

    missing = [f for f in features if f not in X_raw.columns]
    if missing:
        raise ValueError(f"Raw input missing features: {missing}")

    X_proc_arr = preprocessor.transform(X_raw[features])
    X_proc = pd.DataFrame(X_proc_arr, columns=features, index=X_raw.index)

    return predict_exoplanet(X_proc)


# ---------------------------------------------------------------------------
# CLI / demo usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("final_predict.py -- Kepler Exoplanet Classification")
    print("-" * 50)

    # Demo: load processed candidate data and predict
    cand_path = _ROOT / "data" / "processed" / "X_candidate_processed.csv"
    if not cand_path.exists():
        print(f"Candidate data not found: {cand_path}")
        sys.exit(1)

    X_cand = pd.read_csv(cand_path)
    print(f"Loaded {len(X_cand):,} candidate rows")

    results = predict_exoplanet(X_cand)

    conf_n = (results["predicted_class"] == "CONFIRMED").sum()
    fp_n   = (results["predicted_class"] == "FALSE_POSITIVE").sum()
    thr    = float(results["threshold"].iloc[0])

    print(f"Threshold used     : {thr:.2f}")
    print(f"Predicted CONFIRMED: {conf_n:,}")
    print(f"Predicted FALSE POS: {fp_n:,}")
    print("\nFirst 5 predictions:")
    print(results.head(5).to_string())
    print("\n[NOTE] These are model predictions, not astronomical confirmations.")
