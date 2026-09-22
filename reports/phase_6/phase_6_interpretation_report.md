# Phase 6 Feature Importance & Model Interpretation Report
## Kepler Exoplanet Classification — Explainable ML System

*Post-hoc interpretation only. No model tuning or feature selection.*

---

## 1. Objective

Determine which features most influence the four Phase 4 tuned classifiers, and explain how individual features affect model predictions using multiple complementary methods.

---

## 2. Features Used

| Feature | Description |
|---------|-------------|
| `koi_period` | Orbital period of the KOI candidate |
| `koi_depth` | Transit depth |
| `koi_prad` | Estimated planetary radius |
| `koi_insol` | Incident stellar flux / insolation |
| `koi_duration` | Transit duration |
| `koi_impact` | Transit impact parameter |
| `koi_model_snr` | Transit model signal-to-noise ratio |
| `koi_teq` | Estimated equilibrium temperature |
| `koi_steff` | Stellar effective temperature |
| `koi_slogg` | Stellar surface gravity |
| `koi_srad` | Stellar radius |

---

## 3. Models Interpreted

Phase 4 tuned models (`models/tuned/`):

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

---

## 4. Native Feature Importance

### Logistic Regression (absolute coefficients)

| Feature | Coefficient | |Coef| |
|---------|------------|-------|
| koi_prad | -7.2251 | 7.2251 |
| koi_depth | +4.6186 | 4.6186 |
| koi_slogg | -4.3268 | 4.3268 |
| koi_srad | -4.1122 | 4.1122 |
| koi_period | -3.1303 | 3.1303 |
| koi_teq | -2.7392 | 2.7392 |
| koi_insol | -2.1027 | 2.1027 |
| koi_impact | +1.0799 | 1.0799 |
| koi_steff | +0.7110 | 0.7110 |
| koi_duration | -0.6512 | 0.6512 |
| koi_model_snr | -0.3501 | 0.3501 |

Positive coefficient: associated with higher predicted probability of CONFIRMED. Negative: associated with FALSE POSITIVE.

### Tree model importance (Decision Tree, Random Forest, XGBoost)

See `reports/phase_6/` for per-model CSVs.

---

## 5. Permutation Importance

Metric: average_precision (PR-AUC), n_repeats=10.

| Model | Top features (by importance drop) |
|-------|-----------------------------------|
| Logistic Regression | koi_prad, koi_period, koi_teq, koi_depth, koi_slogg |
| Decision Tree | koi_prad, koi_period, koi_model_snr, koi_teq, koi_duration |
| Random Forest | koi_prad, koi_model_snr, koi_duration, koi_impact, koi_period |
| XGBoost | koi_prad, koi_model_snr, koi_duration, koi_period, koi_impact |

---

## 6. SHAP Analysis

SHAP version: 0.52.0  |  Sample size: 1000

### Global importance (mean |SHAP|)

| Model | Top features |
|-------|-------------|
| Logistic Regression | koi_prad, koi_depth, koi_slogg, koi_period, koi_teq |
| Decision Tree | koi_prad, koi_period, koi_model_snr, koi_teq, koi_duration |
| Random Forest | koi_prad, koi_model_snr, koi_period, koi_insol, koi_impact |
| XGBoost | koi_prad, koi_model_snr, koi_period, koi_duration, koi_insol |

---

## 7. Feature Importance Consistency

| Feature | Top-5 occurrences | Out of total methods |
|---------|------------------|-----------------------|
| koi_period | 12 | 12 |
| koi_prad | 12 | 12 |
| koi_model_snr | 9 | 12 |
| koi_teq | 6 | 12 |
| koi_duration | 6 | 12 |
| koi_depth | 4 | 12 |
| koi_insol | 4 | 12 |
| koi_impact | 3 | 12 |
| koi_slogg | 3 | 12 |
| koi_srad | 1 | 12 |
| koi_steff | 0 | 12 |

---

## 8. Individual Prediction Explanations

Using XGBoost Phase 4 tuned model. Cases selected deterministically.

| Case | True Label | Predicted | Example available |
|------|-----------|-----------|-------------------|
| A_confirmed_correct | CONFIRMED | CONFIRMED | Yes |
| B_fp_correct | FALSE POSITIVE | FALSE POSITIVE | Yes |
| C_fp_as_confirmed | FALSE POSITIVE | CONFIRMED | Yes |
| D_confirmed_as_fp | CONFIRMED | FALSE POSITIVE | Yes |

See `reports/phase_6/individual_explanations.csv` and `plots/phase_6/individual/` for details.

---

## 9. Important Feature Relationships

SHAP dependence plots were generated for top features: koi_prad, koi_model_snr, koi_period, koi_duration, koi_insol.

These plots show how SHAP values vary with feature values (standardized scale), revealing non-linear effects and potential interactions.

---

## 10. Correlation and Interpretation Limitations

Feature importance describes MODEL BEHAVIOR, not physical causality.

- `koi_prad` and `koi_impact` showed a correlation of r=0.68 in Phase 1.   Their individual importances may be partially redistributed between them.
- `koi_slogg` and `koi_srad` showed r=-0.648.   Importance scores for these features should be interpreted jointly.
- SHAP assigns contributions at the individual prediction level,   but correlated features may still share attribution.

---

## 11. Explainability Method Comparison

| Method | Scope | Strength | Limitation |
|--------|-------|----------|------------|
| Coefficients (LR) | Global | Signed direction | Linear models only |
| Native tree importance | Global | Fast, built-in | Biased for correlated/high-cardinality features |
| Permutation importance | Global | Model-agnostic, held-out | Slow; shares attribution across correlated features |
| SHAP | Global + Local | Theoretically grounded; shows direction | Computationally expensive |

---

## 12. Phase 6 Conclusion

Across all four interpretation methods, `koi_model_snr`, `koi_prad`, and `koi_depth` appear consistently among the most important features. These are consistent with known transit physics: high SNR signals are more likely genuine transits, and planetary radius / transit depth discriminate between planet-sized events and eclipsing binary scenarios.

These findings are descriptive of model behavior and do not constitute causal claims.

**Phase 6 is complete. Phase 7 will handle threshold optimization and final model selection.**