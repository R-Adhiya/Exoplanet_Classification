# Phase 7 Model Selection & Finalization Report
## Kepler Exoplanet Classification — Explainable ML System

*All values derived from actual data. Test set accessed exactly once.*

---

## 1. Objective

Phase 7 converts the experimental results of Phases 1-6 into a reproducible finalized prediction pipeline. The final model is selected using training/CV evidence only, a classification threshold is optimized on out-of-fold predictions, and the system is evaluated on the untouched test set once.

---

## 2. Candidate Models

12 configurations evaluated (4 models x 3 strategies): baseline, class_weight/scale_pos_weight, SMOTE. See `reports/phase_5/imbalance_cv_results.csv`.

---

## 3. Selection Method

Selection based entirely on 5-fold stratified CV metrics from Phase 5. Primary criterion: mean CV PR-AUC. Secondary (tie-break): lower PR-AUC std deviation.

The test set was NOT inspected during model selection.

---

## 4. Selected Configuration

| Item | Value |
|------|-------|
| Model | XGBoost |
| Strategy | class_weight |
| CV PR-AUC | 0.9556 +/- 0.0086 |
| CV ROC-AUC | 0.9767 +/- 0.0031 |
| CV Recall | 0.9304 |
| CV F1 | 0.9007 |

---

## 5. Threshold Optimization

Out-of-fold (5-fold) predicted probabilities generated on training data only.
Threshold range: 0.05 to 0.95 (step 0.01).
Criterion: maximize F1 subject to recall >= 0.9.
Recall constraint met: True.
Selected threshold: **0.61**

---

## 6. Final Test Evaluation

Test set accessed once, after threshold was frozen.

| Metric | Value |
|--------|-------|
| Accuracy | 0.9354 |
| Precision | 0.9124 |
| Recall | 0.9091 |
| F1 | 0.9107 |
| ROC-AUC | 0.9824 |
| PR-AUC | 0.9661 |
| Specificity | 0.9504 |
| Balanced Accuracy | 0.9298 |
| False Positive Rate | 0.0496 |
| False Negative Rate | 0.0909 |
| TN | 920 |
| FP | 48 |
| FN | 50 |
| TP | 500 |

---

## 7. Calibration

| Metric | Value |
|--------|-------|
| Brier Score | 0.0461 |

The Brier score measures probability calibration (0 = perfect, 1 = worst). No automatic recalibration was applied in this phase.

---

## 8. Candidate Predictions

| Item | Value |
|------|-------|
| Total candidates | 1,977 |
| Predicted CONFIRMED | 525 |
| Predicted FALSE POSITIVE | 1,452 |
| Threshold used | 0.61 |

**Important**: These are model predictions for currently unlabeled CANDIDATE rows. They are NOT scientifically confirmed exoplanets.

---

## 9. Reproducibility

| Item | Value |
|------|-------|
| random_state | 42 |
| Python | 3.12.10 |
| scikit-learn | 1.9.1 |
| XGBoost | 3.4.1 |
| imbalanced-learn | 0.14.2 |
| Final model | `models/final/final_model.pkl` |
| Threshold | `reports/phase_7/final_threshold.json` |

---

## 10. Limitations

- Model predictions are not astronomical scientific confirmations
- Dataset labels originate from the NASA Kepler KOI classification data
- Threshold reflects the selected operational objective (F1 subject to recall constraint)
- Probability calibration may have limitations
- Model performance depends on the feature distribution in training data
- Candidate predictions are preliminary and should be reviewed by domain experts

---

## 11. Conclusion

A finalized prediction pipeline has been created using XGBoost with class_weight strategy. The pipeline achieves PR-AUC=0.9661 and F1=0.9107 on the held-out test set. The classification threshold of 0.61 was selected using out-of-fold predictions from the training set only.

**Phase 7 is complete. Phase 8 will handle Streamlit deployment.**