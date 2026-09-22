# Phase 3 Baseline Modeling Report
## Kepler Exoplanet Classification — Explainable ML System

*All metrics computed from actual model predictions on the held-out test set.*

---

## 1. Objective

Phase 3 establishes baseline classification performance for four standard classifiers trained on the Phase 2 preprocessed dataset. No hyperparameter tuning, class rebalancing, threshold optimization, or feature selection is performed in this phase. These baselines serve as the reference point for all subsequent phases.

---

## 2. Dataset

| Item | Value |
|------|-------|
| Training samples | 6,069 |
| Test samples | 1,518 |
| Features | 11 |
| CONFIRMED in test (target=1) | 550 (36.2%) |
| FALSE POSITIVE in test (target=0) | 968 (63.8%) |
| Positive class prevalence | 0.3623 |

Features used:

- `koi_period`
- `koi_depth`
- `koi_prad`
- `koi_insol`
- `koi_duration`
- `koi_impact`
- `koi_model_snr`
- `koi_teq`
- `koi_steff`
- `koi_slogg`
- `koi_srad`

---

## 3. Models

All models use `random_state=42`. No hyperparameter tuning applied.

| Model | Key Parameters |
|-------|----------------|
| Logistic Regression | `max_iter=2000`, default regularization |
| Decision Tree | sklearn defaults |
| Random Forest | `n_jobs=-1`, sklearn defaults |
| XGBoost | `eval_metric='logloss'`, `n_jobs=-1`, XGBoost defaults |

---

## 4. Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Proportion of all correct predictions |
| Precision | Of predicted CONFIRMED, fraction truly CONFIRMED |
| Recall | Of true CONFIRMED, fraction correctly identified |
| F1 | Harmonic mean of Precision and Recall |
| ROC-AUC | Area under the ROC curve (threshold-independent) |
| PR-AUC / AP | Area under the Precision-Recall curve (sensitive to imbalance) |

PR-AUC is emphasized because the dataset is imbalanced (36.2% positive class).

---

## 5. Baseline Metric Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|----|---------|--------|
| Logistic Regression | 0.8254 | 0.7403 | 0.7982 | 0.7682 | 0.8955 | 0.7520 |
| Decision Tree | 0.8834 | 0.8255 | 0.8600 | 0.8424 | 0.8783 | 0.7606 |
| Random Forest | 0.9315 | 0.9084 | 0.9018 | 0.9051 | 0.9772 | 0.9557 |
| XGBoost | 0.9341 | 0.9091 | 0.9091 | 0.9091 | 0.9797 | 0.9620 |

---

## 6. Confusion Matrices

| Model | TN | FP | FN | TP |
|-------|----|----|----|----|
| Logistic Regression | 814 | 154 | 111 | 439 |
| Decision Tree | 868 | 100 | 77 | 473 |
| Random Forest | 918 | 50 | 54 | 496 |
| XGBoost | 918 | 50 | 50 | 500 |

TN = True Negative (FALSE POSITIVE correctly identified)
FP = False Positive (FALSE POSITIVE misclassified as CONFIRMED)
FN = False Negative (CONFIRMED misclassified as FALSE POSITIVE)
TP = True Positive (CONFIRMED correctly identified)

---

## 7. ROC Analysis

| Model | ROC-AUC |
|-------|---------|
| Logistic Regression | 0.8955 |
| Decision Tree | 0.8783 |
| Random Forest | 0.9772 |
| XGBoost | 0.9797 |

ROC-AUC values are reported for comparison across models. A higher ROC-AUC indicates better discrimination overall, but ROC-AUC can be optimistic under class imbalance. No model is declared a winner based on this metric alone.

---

## 8. Precision-Recall Analysis

| Model | PR-AUC / Average Precision |
|-------|---------------------------|
| Logistic Regression | 0.7520 |
| Decision Tree | 0.7606 |
| Random Forest | 0.9557 |
| XGBoost | 0.9620 |

Positive class prevalence = 0.3623. A random classifier would achieve a PR-AUC approximately equal to the prevalence. Models substantially above prevalence demonstrate meaningful discrimination.

---

## 9. Baseline Observations

- **Recall**: XGBoost produced the highest recall (0.9091); Logistic Regression produced the lowest (0.7982).

- **Precision**: XGBoost produced the highest precision (0.9091); Logistic Regression produced the lowest (0.7403).

- **PR-AUC**: XGBoost produced the highest PR-AUC (0.9620); Logistic Regression produced the lowest (0.7520).

- **False Negatives**: Logistic Regression produced the most false negatives (111), meaning the most CONFIRMED planets were misclassified as FALSE POSITIVE.

- **False Positives**: Logistic Regression produced the most false positives (154), meaning the most FALSE POSITIVE objects were misclassified as CONFIRMED.

These are factual observations from baseline models. No ranking or final selection is made at this stage.

---

## 10. Limitations

The following techniques have NOT been applied in Phase 3:

- No hyperparameter tuning
- No SMOTE or oversampling
- No class weighting (`class_weight='balanced'`)
- No threshold optimization
- No SHAP or feature importance analysis
- No cross-validation
- No final model selection

These will be addressed in later phases.

---

## 11. Environment

| Item | Version |
|------|---------|
| Python | 3.12.10 |
| scikit-learn | 1.9.1 |
| XGBoost | 3.4.1 |
| random_state | 42 |
| Train samples | 6,069 |
| Test samples | 1,518 |
| Features | 11 |

---

## 12. Conclusion

Phase 3 has established baseline classification metrics for four standard models on the Kepler KOI dataset. The metrics above serve as the reference point for Phases 4 and beyond, where class balancing, hyperparameter tuning, threshold optimization, and SHAP analysis will be applied.

**Phase 3 is complete. Phase 4 will handle advanced model comparison and hyperparameter tuning.**