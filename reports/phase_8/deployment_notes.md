# Phase 8 Deployment Notes
## Kepler Exoplanet Classification -- Streamlit Application

---

## Local Setup

### Prerequisites

- Python 3.12+
- pip

### Installation

```bash
cd "Exoplanet Classification"
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### Running Streamlit

```bash
streamlit run app.py
```

The application will open at [http://localhost:8501](http://localhost:8501).

On Windows PowerShell:

```powershell
$env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

---

## Required Files

The application requires the following files at runtime:

```text
app.py                                          # Streamlit entry point
src/final_predict.py                            # Prediction helpers
models/final/final_model.pkl                    # Fitted XGBoost model (665 KB)
models/final/final_model_metadata.json          # Threshold, hyperparameters, versions
data/processed/X_candidate_processed.csv        # 1,977 preprocessed candidate rows
data/processed/processed_feature_names.json     # Feature order
reports/phase_7/candidate_predictions.csv       # Pre-generated candidate predictions
reports/phase_7/final_test_metrics.csv          # Test performance metrics
reports/phase_7/final_threshold.json            # Frozen threshold = 0.61
plots/phase_7/final_confusion_matrix.png        # Confusion matrix image
plots/phase_7/final_roc_curve.png               # ROC curve image
plots/phase_7/final_pr_curve.png                # PR curve image
plots/phase_7/calibration_curve.png             # Calibration image
plots/phase_6/shap_summary_*.png                # SHAP summary plots (4 models)
plots/phase_6/shap_bar_*.png                    # SHAP bar plots (4 models)
plots/phase_6/dependence/shap_dependence_*.png  # Dependence plots (5 features)
```

The following are NOT required at runtime:

```text
data/raw/cumulative_kois.csv     # Raw NASA dataset (not needed for inference)
src/phase_*.py                   # Training scripts (no retraining in app)
models/baseline/                 # Baseline models (not used in production)
models/imbalance/                # Experimental models (not used in production)
```

---

## Deployment Notes

### Streamlit Community Cloud

1. Push the project to a public GitHub repository.
2. Ensure `requirements.txt` is in the repository root.
3. Ensure `app.py` is in the repository root.
4. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
5. Set the main file to `app.py`.

**Note:** The `models/final/final_model.pkl` file (665 KB) must be included in
the repository or hosted separately and loaded via URL.

**Note:** Large files (>100 MB) should use Git LFS. The random forest model in
`models/imbalance/` is ~20 MB and should be excluded from deployment if not needed.

### Environment Variables

No environment variables or API keys are required.

### .streamlit/config.toml

A dark space-inspired theme is configured in `.streamlit/config.toml`.
Delete this file to revert to the Streamlit default theme.

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: final_model.pkl` | Model file missing | Run Phase 7 first or ensure `models/final/` is present |
| `ModuleNotFoundError: streamlit` | Not installed | Run `pip install streamlit` |
| `ValueError: Missing required features` | CSV missing columns | Ensure all 11 features are in the uploaded CSV |
| `ImportError: DLL load failed` | Windows AppControl policy | Run from your own terminal, not from KiroCrew shell |
| Blank page on startup | Model loading cached; wait | Refresh the page |
| `UnicodeDecodeError` | Encoding issue | Set `$env:PYTHONIOENCODING="utf-8"` before running |

---

## Batch Prediction CSV Format

The uploaded CSV must contain exactly these 11 column names:

```
koi_period, koi_duration, koi_depth, koi_prad, koi_impact,
koi_model_snr, koi_teq, koi_insol, koi_steff, koi_slogg, koi_srad
```

These values should be in the **preprocessed (standardized) scale** produced by
the Phase 2 preprocessing pipeline (`models/preprocessor.pkl`).

To predict from raw KOI data, use `predict_from_raw()` in `src/final_predict.py`.

---

## Scientific Disclaimer

This application provides machine-learning predictions based on the trained
Kepler KOI dataset. A prediction labeled CONFIRMED does not constitute
independent astronomical confirmation of an exoplanet.
