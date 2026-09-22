# Phase 5 Class Imbalance Handling Report
## Kepler Exoplanet Classification — Explainable ML System

*All metrics computed from actual data.*

---

## 1. Objective

The labeled training dataset is imbalanced: the minority class (CONFIRMED planets) represents only 36.2% of labeled samples. Phase 5 investigates whether explicit imbalance handling improves classification performance, especially minority-class recall and PR-AUC.

---

## 2. Dataset Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| CONFIRMED (target=1) | 2,198 | 36.22% |
| FALSE POSITIVE (target=0) | 3,871 | 63.78% |
| Total | 6,069 | 100% |

---

## 3. Methods

**Group A: Baseline** -- Phase 4 tuned hyperparameters, no imbalance handling.

**Group B: Class weighting**
- Logistic Regression, Decision Tree, Random Forest: `class_weight='balanced'`
- XGBoost: `scale_pos_weight=1.7611` (= 3871/2198 = negative/positive training count)

**Group C: SMOTE**
- `SMOTE(random_state=42)` wrapped in `imblearn.pipeline.Pipeline`
- Applied ONLY to the training portion of each CV fold
- Validation folds and the test set are never resampled

---

## 4. Leakage Prevention

SMOTE was applied ONLY inside each CV training fold via `imblearn.pipeline.Pipeline`. The pipeline ensures SMOTE runs on the training split of each fold, and the validation split remains untouched. The final test set was accessed only after all CV experiments were complete.

---

## 5. CV Results

| Model | Strategy | CV PR-AUC | +/-std | CV ROC-AUC | CV F1 | CV Recall |
|-------|----------|-----------|--------|------------|-------|-----------|
| Logistic Regression | baseline | 0.7582 | 0.0275 | 0.8963 | 0.7765 | 0.8226 |
| Decision Tree | baseline | 0.8760 | 0.0223 | 0.9408 | 0.8583 | 0.8763 |
| Random Forest | baseline | 0.9463 | 0.0095 | 0.9717 | 0.8846 | 0.8890 |
| XGBoost | baseline | 0.9550 | 0.0095 | 0.9768 | 0.9024 | 0.9104 |
| Logistic Regression | class_weight | 0.7540 | 0.0280 | 0.8955 | 0.7837 | 0.9063 |
| Decision Tree | class_weight | 0.8737 | 0.0146 | 0.9415 | 0.8589 | 0.9176 |
| Random Forest | class_weight | 0.9462 | 0.0080 | 0.9719 | 0.8861 | 0.9163 |
| XGBoost | class_weight | 0.9556 | 0.0086 | 0.9767 | 0.9007 | 0.9304 |
| Logistic Regression | SMOTE | 0.7563 | 0.0283 | 0.8962 | 0.7841 | 0.9045 |
| Decision Tree | SMOTE | 0.8569 | 0.0203 | 0.9366 | 0.8567 | 0.8899 |
| Random Forest | SMOTE | 0.9439 | 0.0089 | 0.9714 | 0.8857 | 0.9090 |
| XGBoost | SMOTE | 0.9530 | 0.0096 | 0.9762 | 0.8995 | 0.9263 |

---

## 6. Test Set Results

| Model | Strategy | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|----------|----------|-----------|--------|----|---------|--------|
| Logistic Regression | baseline | 0.8314 | 0.7517 | 0.7982 | 0.7743 | 0.8978 | 0.7566 |
| Decision Tree | baseline | 0.9038 | 0.8544 | 0.8855 | 0.8696 | 0.9451 | 0.8796 |
| Random Forest | baseline | 0.9302 | 0.9066 | 0.9000 | 0.9033 | 0.9773 | 0.9589 |
| XGBoost | baseline | 0.9354 | 0.9094 | 0.9127 | 0.9111 | 0.9817 | 0.9654 |
| Logistic Regression | class_weight | 0.8202 | 0.6981 | 0.8873 | 0.7814 | 0.8975 | 0.7530 |
| Decision Tree | class_weight | 0.8999 | 0.8149 | 0.9364 | 0.8714 | 0.9454 | 0.8767 |
| Random Forest | class_weight | 0.9302 | 0.8841 | 0.9291 | 0.9060 | 0.9772 | 0.9577 |
| XGBoost | class_weight | 0.9387 | 0.8988 | 0.9364 | 0.9172 | 0.9824 | 0.9661 |
| Logistic Regression | SMOTE | 0.8235 | 0.7032 | 0.8873 | 0.7846 | 0.8982 | 0.7565 |
| Decision Tree | SMOTE | 0.8986 | 0.8333 | 0.9000 | 0.8654 | 0.9456 | 0.8804 |
| Random Forest | SMOTE | 0.9308 | 0.8897 | 0.9236 | 0.9063 | 0.9768 | 0.9532 |
| XGBoost | SMOTE | 0.9420 | 0.9038 | 0.9400 | 0.9216 | 0.9824 | 0.9676 |

---

## 7. Precision-Recall Behavior

Class weighting and SMOTE typically increase minority-class recall at the cost of precision. Observed changes are documented in `reports/phase_5/baseline_vs_imbalance.csv`.

---

## 8. Confusion Matrix Summary

| Model | Strategy | TN | FP | FN | TP |
|-------|----------|----|----|----|----|
| Logistic Regression | baseline | 823 | 145 | 111 | 439 |
| Decision Tree | baseline | 885 | 83 | 63 | 487 |
| Random Forest | baseline | 917 | 51 | 55 | 495 |
| XGBoost | baseline | 918 | 50 | 48 | 502 |
| Logistic Regression | class_weight | 757 | 211 | 62 | 488 |
| Decision Tree | class_weight | 851 | 117 | 35 | 515 |
| Random Forest | class_weight | 901 | 67 | 39 | 511 |
| XGBoost | class_weight | 910 | 58 | 35 | 515 |
| Logistic Regression | SMOTE | 762 | 206 | 62 | 488 |
| Decision Tree | SMOTE | 869 | 99 | 55 | 495 |
| Random Forest | SMOTE | 905 | 63 | 42 | 508 |
| XGBoost | SMOTE | 913 | 55 | 33 | 517 |

---

## 9. SMOTE Balance Verification

| Fold | Orig Pos | Orig Neg | SMOTE Pos | SMOTE Neg | Val Pos | Val Neg |
|------|----------|----------|-----------|-----------|---------|---------|
| 1 | 1759 | 3096 | 3096 | 3096 | 439 | 775 |
| 2 | 1758 | 3097 | 3097 | 3097 | 440 | 774 |
| 3 | 1758 | 3097 | 3097 | 3097 | 440 | 774 |
| 4 | 1758 | 3097 | 3097 | 3097 | 440 | 774 |
| 5 | 1759 | 3097 | 3097 | 3097 | 439 | 774 |

SMOTE creates balanced training folds. Validation folds remain untouched.

---

## 10. Limitations

- Classification threshold remains at 0.5 (not optimized)
- No final model selection performed
- No SHAP or feature importance analysis
- Test set accessed only once, after all CV experiments
- Preprocessing pipeline unchanged from Phase 2

---

## 11. Environment

| Item | Value |
|------|-------|
| imbalanced-learn | 0.14.2 |
| random_state | 42 |
| CV folds | 5 |
| Configurations | 12 (4 models x 3 strategies) |

---

## 12. Conclusion

Phase 5 has measured the impact of class-weighting and SMOTE on all four classifiers using a leakage-safe evaluation framework. The results provide context for Phase 6 (model interpretation) and Phase 7 (threshold optimization and final model selection).

**Phase 5 is complete. Phase 6 will handle SHAP and feature importance.**