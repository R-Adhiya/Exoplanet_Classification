# Phase 4 Model Comparison & Robust Evaluation Report
## Kepler Exoplanet Classification — Explainable ML System

*All metrics computed from actual data. No hard-coded values.*

---

## 1. Objective

Phase 4 performs stratified cross-validation and hyperparameter tuning on the four baseline models from Phase 3. The goal is to establish more rigorous, generalisation-aware performance estimates before applying class-imbalance techniques (Phase 5) and model interpretation (Phase 6). No final model selection occurs here.

---

## 2. Methodology

| Item | Value |
|------|-------|
| CV strategy | StratifiedKFold(n_splits=5, shuffle=True) |
| random_state | 42 |
| Primary metric | average_precision (PR-AUC) |
| LR search | GridSearchCV |
| DT search | GridSearchCV |
| RF search | RandomizedSearchCV (n_iter=20) |
| XGB search | RandomizedSearchCV (n_iter=20) |

**Parameter spaces:**

- **Logistic Regression** (grid): `{'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs'], 'penalty': ['l2']}`
- **Decision Tree** (grid): `{'max_depth': [3, 5, 8, 12, None], 'min_samples_split': [2, 5, 10, 20], 'min_samples_leaf': [1, 2, 5, 10], 'criterion': ['gini', 'entropy']}`
- **Random Forest** (random): `{'n_estimators': [200, 400], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2], 'max_features': ['sqrt', 'log2']}`
- **XGBoost** (random): `{'n_estimators': [100, 200, 300, 500], 'max_depth': [3, 4, 5, 6, 8], 'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2], 'subsample': [0.7, 0.8, 0.9, 1.0], 'colsample_bytree': [0.7, 0.8, 0.9, 1.0], 'min_child_weight': [1, 3, 5]}`

---

## 3. Cross-Validation Results

| Model | CV PR-AUC mean | CV PR-AUC std | CV ROC-AUC | CV F1 |
|-------|---------------|---------------|------------|-------|
| Logistic Regression | 0.7582 | 0.0275 | 0.8963 | 0.7765 |
| Decision Tree | 0.8760 | 0.0223 | 0.9408 | 0.8583 |
| Random Forest | 0.9463 | 0.0095 | 0.9717 | 0.8846 |
| XGBoost | 0.9550 | 0.0095 | 0.9768 | 0.9024 |

---

## 4. Best Hyperparameters

**Logistic Regression:** `{'C': 100, 'penalty': 'l2', 'solver': 'lbfgs'}`

**Decision Tree:** `{'criterion': 'entropy', 'max_depth': 12, 'min_samples_leaf': 10, 'min_samples_split': 2}`

**Random Forest:** `{'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': None}`

**XGBoost:** `{'subsample': 0.7, 'n_estimators': 200, 'min_child_weight': 3, 'max_depth': 8, 'learning_rate': 0.05, 'colsample_bytree': 0.9}`

---

## 5. Tuned Test-Set Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|----|---------|--------|
| Logistic Regression | 0.8314 | 0.7517 | 0.7982 | 0.7743 | 0.8978 | 0.7566 |
| Decision Tree | 0.9038 | 0.8544 | 0.8855 | 0.8696 | 0.9451 | 0.8796 |
| Random Forest | 0.9302 | 0.9066 | 0.9000 | 0.9033 | 0.9773 | 0.9589 |
| XGBoost | 0.9354 | 0.9094 | 0.9127 | 0.9111 | 0.9817 | 0.9654 |

---

## 6. Baseline vs Tuned

| Model | Metric | Baseline | Tuned | Delta |
|-------|--------|----------|-------|-------|
| Logistic Regression | accuracy | 0.8254 | 0.8314 | +0.0060 |
| Logistic Regression | precision | 0.7403 | 0.7517 | +0.0114 |
| Logistic Regression | recall | 0.7982 | 0.7982 | +0.0000 |
| Logistic Regression | f1 | 0.7682 | 0.7743 | +0.0061 |
| Logistic Regression | roc_auc | 0.8955 | 0.8978 | +0.0023 |
| Logistic Regression | pr_auc | 0.7520 | 0.7566 | +0.0046 |
| Decision Tree | accuracy | 0.8834 | 0.9038 | +0.0204 |
| Decision Tree | precision | 0.8255 | 0.8544 | +0.0289 |
| Decision Tree | recall | 0.8600 | 0.8855 | +0.0255 |
| Decision Tree | f1 | 0.8424 | 0.8696 | +0.0272 |
| Decision Tree | roc_auc | 0.8783 | 0.9451 | +0.0668 |
| Decision Tree | pr_auc | 0.7606 | 0.8796 | +0.1190 |
| Random Forest | accuracy | 0.9315 | 0.9302 | -0.0013 |
| Random Forest | precision | 0.9084 | 0.9066 | -0.0018 |
| Random Forest | recall | 0.9018 | 0.9000 | -0.0018 |
| Random Forest | f1 | 0.9051 | 0.9033 | -0.0018 |
| Random Forest | roc_auc | 0.9772 | 0.9773 | +0.0001 |
| Random Forest | pr_auc | 0.9557 | 0.9589 | +0.0032 |
| XGBoost | accuracy | 0.9341 | 0.9354 | +0.0013 |
| XGBoost | precision | 0.9091 | 0.9094 | +0.0003 |
| XGBoost | recall | 0.9091 | 0.9127 | +0.0036 |
| XGBoost | f1 | 0.9091 | 0.9111 | +0.0020 |
| XGBoost | roc_auc | 0.9797 | 0.9817 | +0.0020 |
| XGBoost | pr_auc | 0.9620 | 0.9654 | +0.0034 |

---

## 7. Stability (CV PR-AUC)

| Model | Mean | Std | Min Fold | Max Fold |
|-------|------|-----|----------|----------|
| Logistic Regression | 0.7582 | 0.0275 | 0.7118 | 0.7863 |
| Decision Tree | 0.8760 | 0.0223 | 0.8379 | 0.9036 |
| Random Forest | 0.9463 | 0.0095 | 0.9294 | 0.9581 |
| XGBoost | 0.9550 | 0.0095 | 0.9375 | 0.9637 |

---

## 8. Limitations

- No SMOTE or oversampling applied
- No class weighting (`class_weight='balanced'`) applied
- No threshold optimization performed
- No SHAP or feature importance analysis performed
- No final model selection made

These will be addressed in Phases 5, 6, and 7.

---

## 9. Environment

| Item | Value |
|------|-------|
| Python | 3.12.10 |
| scikit-learn | 1.9.1 |
| XGBoost | 3.4.1 |
| Train samples | 6,069 |
| Test samples | 1,518 |
| Features | 11 |

---

## 10. Conclusion

Phase 4 has established cross-validated and hyperparameter-tuned performance estimates for all four classifiers. The tuned metrics serve as the reference point for Phase 5 (class-imbalance handling) and Phase 6 (model interpretation). No final production model has been selected.

**Phase 4 is complete. Phase 5 will investigate class-imbalance techniques.**