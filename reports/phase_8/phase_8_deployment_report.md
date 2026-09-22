# Phase 8 Deployment Report
## Kepler Exoplanet Classification — Explainable ML System

*Streamlit application built on finalized Phase 7 artifacts.*

---

## 1. Application Overview

A Streamlit web application that exposes the Phase 7 finalized XGBoost model
for interactive KOI classification. The application is inference-only — no
model training, threshold optimization, or preprocessing fitting occurs at
runtime.

**Entry point:** `app.py`
**Model:** XGBoost (scale_pos_weight = 1.7611)
**Threshold:** 0.61 (frozen from Phase 7 OOF optimization)
**Streamlit version:** 1.64.0

---

## 2. Pages

| Page | Description |
|------|-------------|
| Single Prediction | Enter 11 feature values, get a single KOI prediction with probability visualization |
| Batch Prediction | Upload a CSV, run predictions on all rows, download results |
| Candidate Explorer | Browse and filter the 1,977 CANDIDATE predictions from Phase 7 |
| Model Information | Final model config, test metrics, confusion matrix, ROC/PR/calibration plots |
| Explainability | Phase 6 SHAP summary/bar/dependence plots with feature descriptions |
| About | Project overview, pipeline phases, dataset statistics, limitations |

---

## 3. Prediction Workflow (Single)

```text
User enters 11 feature values (preprocessed scale)
            ↓
Input validation (NaN/Inf check, all 11 features present)
            ↓
predict_single() via joblib-loaded XGBoost model
            ↓
Threshold applied (0.61 frozen)
            ↓
CONFIRMED / FALSE POSITIVE + probability displayed
            ↓
Probability bar visualization
            ↓
Feature summary table
```

---

## 4. Batch Workflow

```text
User uploads CSV
            ↓
Column validation (all 11 required features present)
            ↓
Numeric / NaN / Inf validation per column
            ↓
predict_batch() on all rows
            ↓
Prediction distribution chart + probability histogram
            ↓
Results table (first 100 rows)
            ↓
Full CSV download
```

---

## 5. Candidate Explorer

- Loads `data/processed/X_candidate_processed.csv` (1,977 rows)
- Loads `reports/phase_7/candidate_predictions.csv` (Phase 7 pre-generated)
- Summary metrics: 1,977 total, 525 predicted CONFIRMED, 1,452 predicted FP
- Probability slider for interactive filtering (exploration only, does not change model threshold)
- Probability distribution histogram with threshold reference line

---

## 6. Explainability

- Displays Phase 6 SHAP summary and bar plots for all 4 models
- Interactive model selector (LR, DT, RF, XGBoost)
- SHAP dependence plots for top 5 features (koi_prad, koi_model_snr, koi_period, koi_duration, koi_insol)
- Feature importance table with Phase 6 notes
- Clear language distinguishing model behavior from physical causation

---

## 7. Model Information

- Final model configuration table
- Hyperparameters from Phase 4
- Threshold details from `reports/phase_7/final_threshold.json`
- Test metrics from `reports/phase_7/final_test_metrics.csv`
- Confusion matrix, ROC curve, PR curve, calibration curve (all Phase 7 images)

---

## 8. Validation

| Test | Result |
|------|--------|
| app.py syntax check | PASS |
| Single prediction (test row 0) | PASS — CONFIRMED, prob=0.6410 |
| Batch prediction (1,977 candidates) | PASS — CONF=525, FP=1452 (matches Phase 7) |
| Model reload (joblib) | PASS — max diff 0.00e+00 |
| Missing feature validation | PASS — error raised |
| requirements.txt complete | PASS |

---

## 9. Deployment Requirements

```text
Python:          3.12+
streamlit:       >=1.28.0
pandas:          >=2.0.0
numpy:           >=1.24.0
scikit-learn:    >=1.2.0
xgboost:         >=1.7.0
joblib:          >=1.2.0
matplotlib:      >=3.7.0
shap:            >=0.41.0
imbalanced-learn:>=0.11.0
```

Run with: `streamlit run app.py`

---

## 10. Limitations

1. The application accepts preprocessed (standardized) feature values.
   Raw KOI values require preprocessing via `models/preprocessor.pkl` first.
2. Predictions are ML model outputs, not scientific astronomical confirmations.
3. The classification threshold (0.61) is fixed; it cannot be changed via the UI.
4. SHAP plots are static images from Phase 6; they do not recompute at runtime.
5. The candidate probability filter slider is for exploration only and does not
   affect the production classification threshold.
6. On Windows, the app must be launched from the user's own terminal (not from
   a restricted shell process) due to AppControl DLL policies.

---

## 11. Conclusion

Phase 8 delivers a production-style Streamlit application that exposes the
complete Kepler exoplanet classification pipeline through an interactive web UI.
The application is inference-only, uses relative paths for portability, caches
model loading, and includes a prominent scientific disclaimer on every page.

**Phase 8 is complete. Phase 9 will handle README, GitHub setup, and project documentation.**
