# Kepler Exoplanet Classification
## Explainable ML System for Planet Candidate Verification

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64-red)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4-green)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.52-orange)](https://shap.readthedocs.io)

An end-to-end explainable machine learning system for Kepler exoplanet candidate verification, combining leakage-aware preprocessing, cross-validated model selection, class-imbalance handling, SHAP-based interpretation, threshold optimization, and an interactive Streamlit deployment.

> **Important:** Model predictions for CANDIDATE observations are **machine-learning predictions, not scientific confirmations of exoplanets**. This system is an educational and research tool — it does not replace independent astronomical validation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Features Used](#features-used)
- [Data Leakage Prevention](#data-leakage-prevention)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Class Imbalance Handling](#class-imbalance-handling)
- [Explainability](#explainability)
- [Final Model](#final-model)
- [Candidate Predictions](#candidate-predictions)
- [Streamlit Application](#streamlit-application)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Reproducibility](#reproducibility)
- [Technology Stack](#technology-stack)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

## Project Overview

NASA's Kepler space telescope observed over 150,000 stars from 2009–2018, detecting periodic brightness dips caused by objects transiting in front of their host stars. These transit events produce **Kepler Objects of Interest (KOIs)** — candidate signals that may represent real exoplanets, stellar eclipses, or instrumental artefacts.

Expert astronomers classify KOIs as:

| Label | Meaning |
|-------|---------|
| **CONFIRMED** | The transit signal is consistent with a real exoplanet after thorough validation |
| **FALSE POSITIVE** | The transit signal is caused by a non-planetary source (e.g., eclipsing binary, background star) |
| **CANDIDATE** | The classification is still under investigation — labels are not yet known |

Machine learning can assist candidate verification by learning patterns from already-classified KOIs and applying them to the unclassified CANDIDATE pool. This project builds such a system with a focus on **explainability, leakage prevention, and scientific responsibility**.

---

## Key Features

- NASA Kepler KOI cumulative dataset (9,564 observations)
- Thorough exploratory data analysis (Phase 1)
- Leakage-aware preprocessing: imputation, log-transforms, scaling — fitted on training data only (Phase 2)
- Four ML baselines: Logistic Regression, Decision Tree, Random Forest, XGBoost (Phase 3)
- Stratified 5-fold CV with PR-AUC as the primary tuning metric (Phase 4)
- Hyperparameter tuning with GridSearchCV and RandomizedSearchCV (Phase 4)
- Class imbalance experiments: baseline, class weighting, SMOTE-in-pipeline (Phase 5)
- SHAP explainability with global, local, and dependence analysis (Phase 6)
- Out-of-fold threshold optimization (maximize F1 subject to recall ≥ 0.90) (Phase 7)
- Final model evaluation on a fully held-out test set (Phase 7)
- Predictions for 1,977 unlabeled CANDIDATE KOIs (Phase 7)
- Interactive Streamlit web app: single prediction, batch CSV, candidate explorer, model info, explainability dashboard (Phase 8)

---

## Dataset

**Source:** [NASA Exoplanet Archive — Cumulative KOI Table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)

**Raw file:** `data/raw/cumulative_kois.csv`

| Statistic | Value |
|-----------|-------|
| Total rows | 9,564 |
| Total columns | 153 |
| Labeled observations (CONFIRMED + FALSE POSITIVE) | 7,587 |
| CANDIDATE observations (unlabeled) | 1,977 |

**Labeled class distribution:**

| Class | Count | Percentage |
|-------|-------|------------|
| CONFIRMED (target = 1) | 2,748 | 36.2% |
| FALSE POSITIVE (target = 0) | 4,839 | 63.8% |

CANDIDATE rows were kept completely separate throughout the project. They were **never** used to fit preprocessing steps, imputers, scalers, or models. They appear only in the candidate prediction section.

---

## Features Used

Eleven numerical features were selected based on transit physics and domain knowledge:

| # | Feature | Description |
|---|---------|-------------|
| 1 | `koi_period` | Orbital period of the KOI candidate |
| 2 | `koi_duration` | Transit duration |
| 3 | `koi_depth` | Transit depth (fractional flux decrease) |
| 4 | `koi_prad` | Estimated planetary radius |
| 5 | `koi_impact` | Transit impact parameter |
| 6 | `koi_model_snr` | Transit model signal-to-noise ratio |
| 7 | `koi_teq` | Estimated equilibrium temperature |
| 8 | `koi_insol` | Incident stellar flux / insolation |
| 9 | `koi_steff` | Stellar effective temperature |
| 10 | `koi_slogg` | Stellar surface gravity |
| 11 | `koi_srad` | Stellar radius |

---

## Data Leakage Prevention

The following columns were **explicitly removed** before any analysis or modelling:

| Column | Why Removed |
|--------|-------------|
| `koi_score` | Directly encodes a disposition confidence score |
| `koi_pdisposition` | Preliminary disposition — derived from or predictive of the target |
| `koi_fpflag_nt` | False positive flag (not transit-like) |
| `koi_fpflag_ss` | False positive flag (stellar eclipse) |
| `koi_fpflag_co` | False positive flag (centroid offset) |
| `koi_fpflag_ec` | False positive flag (ephemeris contamination) |

Retaining any of these columns would constitute **target leakage** — the model would learn from information that encodes the label itself rather than learning from physical transit properties.

---

## Preprocessing Pipeline

**File:** `models/preprocessor.pkl`  
**Script:** `src/phase_2_preprocessing.py`

The preprocessing pipeline follows a strict leakage-safe design:

```
Training data
      ↓
Stratified 80/20 train-test split (random_state=42, stratify=y)
      ↓
SimpleImputer(strategy="median")  ← fitted on X_train ONLY
      ↓
log1p transformation (koi_period, koi_depth, koi_prad, koi_insol)
      ↓
StandardScaler()  ← fitted on X_train ONLY
      ↓
ColumnTransformer (sklearn Pipeline)
```

| Property | Value |
|----------|-------|
| Train size | 6,069 (80%) |
| Test size | 1,518 (20%) |
| random_state | 42 |
| Output features | 11 |
| NaN after preprocessing | 0 |
| Log-transformed features | koi_period, koi_depth, koi_prad, koi_insol |

Test and candidate data were **never** used to fit the imputer or scaler — only to transform.

---

## Machine Learning Models

Four baseline classifiers were evaluated (Phase 3), then tuned using stratified 5-fold cross-validation with **PR-AUC** as the primary metric (Phase 4).

### Baseline Results (Phase 3)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|----|---------|--------|
| Logistic Regression | 0.8254 | 0.7403 | 0.7982 | 0.7682 | 0.8955 | 0.7520 |
| Decision Tree | 0.8834 | 0.8255 | 0.8600 | 0.8424 | 0.8783 | 0.7606 |
| Random Forest | 0.9315 | 0.9084 | 0.9018 | 0.9051 | 0.9772 | 0.9557 |
| XGBoost | 0.9341 | 0.9091 | 0.9091 | 0.9091 | 0.9797 | 0.9620 |

---

## Hyperparameter Tuning

**Script:** `src/phase_4_model_comparison.py`

Tuned using 5-fold StratifiedKFold, primary scoring: `average_precision`.

| Model | Best Hyperparameters |
|-------|---------------------|
| Logistic Regression | `C=100, penalty=l2, solver=lbfgs` |
| Decision Tree | `criterion=entropy, max_depth=12, min_samples_leaf=10, min_samples_split=2` |
| Random Forest | `n_estimators=400, max_depth=None, min_samples_split=5, min_samples_leaf=2, max_features=log2` |
| XGBoost | `n_estimators=200, max_depth=8, learning_rate=0.05, subsample=0.7, colsample_bytree=0.9, min_child_weight=3` |

### Tuned Model CV Results

| Model | CV PR-AUC | ±std | CV ROC-AUC | CV F1 |
|-------|-----------|------|------------|-------|
| Logistic Regression | 0.7582 | ±0.0275 | 0.8963 | 0.7765 |
| Decision Tree | 0.8760 | ±0.0223 | 0.9408 | 0.8583 |
| Random Forest | 0.9463 | ±0.0095 | 0.9717 | 0.8846 |
| XGBoost | **0.9556** | **±0.0086** | 0.9768 | 0.9024 |

---

## Class Imbalance Handling

**Script:** `src/phase_5_imbalance.py`

The labeled dataset is imbalanced (36.2% CONFIRMED, 63.8% FALSE POSITIVE). Three strategies were evaluated for all four models (12 configurations total):

| Strategy | Description |
|----------|-------------|
| Baseline | No imbalance handling |
| Class weighting | `class_weight="balanced"` for LR/DT/RF; `scale_pos_weight=1.7611` for XGBoost |
| SMOTE | Synthetic Minority Over-sampling, applied **inside** each CV fold via `imblearn.pipeline.Pipeline` |

**XGBoost scale_pos_weight calculation:**
```
scale_pos_weight = FP_training_count / CONF_training_count = 3871 / 2198 = 1.7611
```

> SMOTE was applied exclusively within the training portion of each cross-validation fold. Validation folds and the test set were never resampled. This prevents synthetic data from contaminating model evaluation.

Both class weighting and SMOTE generally increased minority-class recall (+0.02 to +0.09) at a modest cost to precision. XGBoost with class weighting achieved the highest CV PR-AUC (0.9556) and was selected as the final configuration.

---

## Explainability

**Script:** `src/phase_6_interpretability.py`

Four complementary interpretation methods were applied to the Phase 4 tuned models:

| Method | Scope | Description |
|--------|-------|-------------|
| Logistic Regression coefficients | Global | Signed direction of each feature's contribution |
| Native tree importance | Global | Impurity-based importance (DT, RF, XGBoost) |
| Permutation importance | Global | PR-AUC drop when each feature is shuffled (n_repeats=10) |
| SHAP (TreeExplainer / LinearExplainer) | Global + Local | Shapley values showing feature contributions per prediction |

### Top SHAP Features (Mean |SHAP|)

| Model | Top 3 Features |
|-------|---------------|
| Logistic Regression | `koi_prad`, `koi_depth`, `koi_slogg` |
| Decision Tree | `koi_prad`, `koi_period`, `koi_model_snr` |
| Random Forest | `koi_prad`, `koi_model_snr`, `koi_period` |
| XGBoost | `koi_prad`, `koi_model_snr`, `koi_period` |

**`koi_prad` (estimated planetary radius) was the top global SHAP feature across all four models.** This is consistent with transit physics: very large inferred radii often indicate diluted eclipsing binary scenarios rather than planet-sized transits.

> These findings describe **model behavior**, not causal astronomical relationships.

---

## Final Model

**Script:** `src/phase_7_finalization.py`  
**Model file:** `models/final/final_model.pkl`  
**Metadata:** `models/final/final_model_metadata.json`

### Selected Configuration

| Property | Value |
|----------|-------|
| Model | XGBoost |
| Imbalance strategy | scale_pos_weight = 1.7611 |
| Selection basis | Highest CV PR-AUC (training data only — test set not used for selection) |
| CV PR-AUC | 0.9556 ± 0.0086 |
| CV ROC-AUC | 0.9767 |

### Threshold Optimization

The classification threshold was optimized using **out-of-fold predictions** from the training set only:

- Method: 5-fold StratifiedKFold OOF probabilities
- Objective: maximize F1 subject to recall ≥ 0.90
- **Frozen threshold: 0.61**
- OOF recall at threshold: 0.9081
- OOF F1 at threshold: 0.9040

The test set was accessed **only once**, after the configuration and threshold were completely frozen.

### Final Test-Set Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.9354 |
| Precision | 0.9124 |
| Recall | 0.9091 |
| F1 | 0.9107 |
| ROC-AUC | 0.9824 |
| PR-AUC | 0.9661 |
| Balanced Accuracy | 0.9298 |
| Brier Score | 0.0461 |

**Confusion matrix (test set, threshold = 0.61):**

```
                  Predicted FP    Predicted CONF
Actual FP              920              48
Actual CONF             50             500
```

TN = 920 | FP = 48 | FN = 50 | TP = 500

---

## Candidate Predictions

After the final model was frozen, predictions were generated for the **1,977 CANDIDATE KOI rows** that were withheld throughout the entire project:

| Prediction | Count | Percentage |
|------------|-------|------------|
| CONFIRMED | 525 | 26.6% |
| FALSE POSITIVE | 1,452 | 73.4% |

**File:** `reports/phase_7/candidate_predictions.csv`

> These are **model predictions only**. They do not constitute independent scientific confirmation of exoplanets. All candidate predictions should be treated as preliminary machine-learning outputs subject to expert astronomical review.

---

## Streamlit Application

**Script:** `app.py`  
**Theme:** `.streamlit/config.toml`

### Launch

```bash
# Windows PowerShell
cd "C:\Users\Adhiya\Desktop\Projects\Exoplanet Classification"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

```bash
# macOS / Linux
cd "Exoplanet Classification"
source .venv/bin/activate
streamlit run app.py
```

Application opens at: [http://localhost:8501](http://localhost:8501)

### Application Pages

| Page | Description |
|------|-------------|
| **Single Prediction** | Enter 11 preprocessed feature values, receive CONFIRMED / FALSE POSITIVE prediction with probability bar |
| **Batch Prediction** | Upload CSV with 11 features, download predictions for all rows |
| **Candidate Explorer** | Browse 1,977 Phase 7 candidate predictions with interactive probability filter |
| **Model Information** | Final model config, test metrics, confusion matrix, ROC/PR/calibration plots |
| **Explainability** | Phase 6 SHAP summary, bar, and dependence plots for all four models |
| **About** | Project overview, pipeline phases, dataset statistics, limitations |

---

## Project Structure

```text
Exoplanet Classification/
│
├── app.py                              # Streamlit application entry point
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── .streamlit/
│   └── config.toml                     # Streamlit dark theme
│
├── src/
│   ├── phase_1_eda.py                  # Data acquisition & EDA
│   ├── phase_2_preprocessing.py        # Preprocessing pipeline
│   ├── phase_3_baseline_models.py      # Baseline ML models
│   ├── phase_4_model_comparison.py     # Hyperparameter tuning & CV
│   ├── phase_5_imbalance.py            # Class imbalance experiments
│   ├── phase_6_interpretability.py     # SHAP & feature importance
│   ├── phase_7_finalization.py         # Model selection & threshold
│   └── final_predict.py               # Production prediction function
│
├── models/
│   ├── preprocessor.pkl                # Phase 2 fitted preprocessing pipeline
│   ├── baseline/                       # Phase 3 baseline model PKLs
│   ├── tuned/                          # Phase 4 tuned model PKLs
│   ├── imbalance/                      # Phase 5 experimental model PKLs
│   └── final/
│       ├── final_model.pkl             # Final XGBoost model (665 KB)
│       └── final_model_metadata.json   # Threshold, features, versions
│
├── data/
│   ├── raw/
│   │   └── cumulative_kois.csv         # NASA Exoplanet Archive (9,564 rows)
│   └── processed/
│       ├── X_train_processed.csv
│       ├── X_test_processed.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       ├── X_candidate_processed.csv
│       ├── koi_labeled.csv
│       ├── koi_candidates.csv
│       ├── feature_columns.json
│       └── processed_feature_names.json
│
├── reports/
│   ├── phase_1/                        # EDA reports & summaries
│   ├── phase_2/                        # Preprocessing summaries
│   ├── phase_3/                        # Baseline model comparisons
│   ├── phase_4/                        # Tuning results, best parameters
│   ├── phase_5/                        # Imbalance experiment results
│   ├── phase_6/                        # Explainability reports
│   ├── phase_7/                        # Final model selection & metrics
│   ├── phase_8/                        # Deployment notes
│   └── phase_9/                        # Documentation validation
│
└── plots/
    ├── phase_1/                        # EDA plots
    ├── phase_2/                        # Preprocessing plots
    ├── phase_3/                        # Baseline model plots
    ├── phase_4/                        # Tuning comparison plots
    ├── phase_5/                        # Imbalance experiment plots
    ├── phase_6/                        # SHAP & importance plots
    └── phase_7/                        # Final model evaluation plots
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/R-Adhiya/Exoplanet_Classification.git
cd Exoplanet_Classification

# 2. Create virtual environment
python -m venv .venv

# 3. Activate (Windows)
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch the Streamlit app
streamlit run app.py
```

> You do **not** need to retrain any model. The final model (`models/final/final_model.pkl`) and all processed data are included in the repository.

### Regenerating from Scratch (optional)

If you want to reproduce the full pipeline:

```bash
# Phase 1: Download NASA data and run EDA
python src/phase_1_eda.py

# Phase 2: Build preprocessing pipeline
python src/phase_2_preprocessing.py

# Phase 3: Baseline models
python src/phase_3_baseline_models.py

# Phase 4: Hyperparameter tuning
python src/phase_4_model_comparison.py

# Phase 5: Class imbalance experiments
python src/phase_5_imbalance.py

# Phase 6: SHAP interpretability
python src/phase_6_interpretability.py

# Phase 7: Final model selection
python src/phase_7_finalization.py
```

---

## Reproducibility

| Element | Value |
|---------|-------|
| `random_state` | 42 (used throughout all phases) |
| Preprocessing | Saved as `models/preprocessor.pkl` |
| Final model | Saved as `models/final/final_model.pkl` |
| Threshold | Saved as `reports/phase_7/final_threshold.json` (= 0.61) |
| Candidate predictions | Saved as `reports/phase_7/candidate_predictions.csv` |
| Test evaluation | Performed exactly once, after threshold was frozen |
| Feature order | Saved as `data/processed/processed_feature_names.json` |

The NASA Exoplanet Archive cumulative KOI table is publicly available.
License and usage terms are governed by the archive's data use policy.

---

## Technology Stack

| Category | Technologies |
|----------|-------------|
| Language | Python 3.12 |
| Data | Pandas 2.x, NumPy 2.x |
| Machine Learning | Scikit-learn 1.9, XGBoost 3.4 |
| Class Imbalance | imbalanced-learn 0.14 |
| Explainability | SHAP 0.52 |
| Visualization | Matplotlib 3.11, Seaborn 0.13 |
| Deployment | Streamlit 1.64 |
| Model Persistence | Joblib |
| Version Control | Git / GitHub |

---

## Limitations

1. **ML predictions are not astronomical confirmation.** A CONFIRMED prediction means the model assigns a high probability of agreement with existing KOI labels — it does not independently verify the existence of an exoplanet.
2. **Dataset quality** depends on Kepler pipeline measurements and may contain heterogeneous cases within each class.
3. **Distribution shift:** The model is trained on historical Kepler data; performance on observations from other telescopes or future data may differ.
4. **Feature scope:** Only 11 photometric/stellar features are used. Richer feature sets (e.g., light-curve morphology) could improve performance.
5. **Threshold 0.61** was optimized for this project's objective (maximize F1, recall ≥ 0.90) and may not be appropriate for all use cases.
6. **Interpretability methods** describe model behavior, not causal physical relationships.
7. **External astronomical validation** is required before any prediction can be treated as a scientific claim.

---

## Future Work

- Incorporate light-curve-level features (transit shape, duration ratios)
- CNN or Transformer-based transit signal analysis
- Uncertainty quantification (conformal prediction, Bayesian models)
- Distribution shift evaluation on K2 / TESS data
- Probability calibration improvements (Platt scaling, isotonic regression)
- Active learning for efficient expert review of borderline candidates
- Larger labeled datasets incorporating K2 and TESS KOIs
- Human-in-the-loop scientific review interface
- Temporal validation across Kepler observation quarters

---

## Portfolio Summary

**An end-to-end explainable machine learning system for Kepler exoplanet candidate verification, combining leakage-aware preprocessing, cross-validated model selection, class-imbalance handling, SHAP-based interpretation, threshold optimization, and an interactive Streamlit deployment.**

The project covers the complete ML lifecycle — from raw NASA data acquisition through EDA, feature engineering, baseline modelling, hyperparameter tuning, imbalance experiments, interpretability analysis, final model selection, and a deployable web application — with a consistent focus on scientific responsibility and reproducibility.

---

## License

This project uses publicly available NASA data from the Exoplanet Archive.
The code is intended for educational and research purposes.
