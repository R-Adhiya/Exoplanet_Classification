# Phase 2 Preprocessing Report
## Kepler Exoplanet Classification — Explainable ML System

*All values derived from actual data. No hard-coded statistics.*

---

## 1. Objective

Build a leakage-safe preprocessing pipeline fitted exclusively on training data.
No model training is performed in this phase.

---

## 2. Dataset Split

| Item | Value |
|------|-------|
| Total labeled samples | 7,587 |
| Training samples (80%) | 6,069 |
| Test samples (20%) | 1,518 |
| CANDIDATE samples (held out) | 1,977 |
| random_state | 42 |
| stratify | y (target) |

**Train class distribution:**

| Class | Count | Percentage |
|-------|-------|------------|
| CONFIRMED (target=1) | 2,198 | 36.2% |
| FALSE POSITIVE (target=0) | 3,871 | 63.8% |

**Test class distribution:**

| Class | Count | Percentage |
|-------|-------|------------|
| CONFIRMED (target=1) | 550 | 36.2% |
| FALSE POSITIVE (target=0) | 968 | 63.8% |

---

## 3. Feature List

11 numerical features used for modeling:

- `koi_period` (log1p transformed)
- `koi_duration`
- `koi_depth` (log1p transformed)
- `koi_prad` (log1p transformed)
- `koi_impact`
- `koi_model_snr`
- `koi_teq`
- `koi_insol` (log1p transformed)
- `koi_steff`
- `koi_slogg`
- `koi_srad`

---

## 4. Missing Value Handling

Strategy: `SimpleImputer(strategy='median')`

| Split | Missing before | Missing after |
|-------|---------------|---------------|
| Training | 1,821 | 0 |
| Test     | 470  | 0  |

Imputer fitted on training data only. Test/candidate data use training medians.

---

## 5. Log Transformations

Applied `np.log1p()` to the following four features (high positive skewness in Phase 1):

| Feature | Skewness Before | Skewness After | Reduction |
|---------|----------------|----------------|-----------|
| koi_period | 3.02 | 0.72 | 2.30 |
| koi_depth | 4.17 | 1.04 | 3.13 |
| koi_prad | 48.42 | 1.22 | 47.20 |
| koi_insol | 43.83 | 0.16 | 43.67 |

Transformation applied after median imputation (no leakage possible).

---

## 6. Standardization

`StandardScaler` applied to all 11 features after imputation/log transformation.
Scaler fitted on training data only.

| Metric | Value |
|--------|-------|
| Max absolute training mean | 0.000000 (target ~0) |
| Min training std dev | 1.000000 (target ~1) |

---

## 7. Leakage Prevention

The following columns were explicitly excluded from all preprocessing:

- `koi_score`
- `koi_pdisposition`
- `koi_fpflag_nt`
- `koi_fpflag_ss`
- `koi_fpflag_co`
- `koi_fpflag_ec`
- `koi_disposition`
- `target`

The ColumnTransformer uses `remainder='drop'` to discard any columns not
explicitly listed in the transformers.
Preprocessing was fitted on `X_train` only (`preprocessor.fit(X_train)`).

---

## 8. sklearn Pipeline Architecture

```
ColumnTransformer
  |
  +-- log_numeric  [koi_period, koi_depth, koi_prad, koi_insol]
  |      +-- SimpleImputer(strategy='median')
  |      +-- FunctionTransformer(np.log1p)
  |      +-- StandardScaler()
  |
  +-- plain_numeric  [koi_duration, koi_impact, koi_model_snr, koi_teq,
                       koi_steff, koi_slogg, koi_srad]
         +-- SimpleImputer(strategy='median')
         +-- StandardScaler()
```

---

## 9. Post-Processing Validation

| Check | Result |
|-------|--------|
| NaN in X_train_processed | 0 |
| NaN in X_test_processed | 0 |
| Output feature count | 11 |
| Leakage columns present | None |
| Preprocessor fitted on | X_train only |

---

## 10. Saved Preprocessor

Saved as: `models/preprocessor.pkl` (joblib format).
Reloaded and verified successfully.
Can be applied to test, validation, and candidate data in later phases.

---

## 11. Conclusion

- Stratified 80/20 split created: 6,069 train / 1,518 test
- All preprocessing fitted exclusively on training data
- Zero missing values after preprocessing
- Exactly 11 output features
- No leakage columns present
- 1,977 CANDIDATE rows kept separate (never used to fit preprocessing)
- Fitted preprocessor saved as `models/preprocessor.pkl`

**Phase 2 is complete. Phase 3 will handle model training and evaluation.**