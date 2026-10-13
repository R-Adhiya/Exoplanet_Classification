# Benchmark Comparison: Kepler KOI Classification

> **Leakage note**: This project explicitly removes `koi_score`, `koi_pdisposition`,
> `koi_fpflag_nt`, `koi_fpflag_ss`, `koi_fpflag_co`, `koi_fpflag_ec`, and
> `koi_disposition` from all training data. Most published benchmarks and Kaggle
> notebooks retain some of these columns, making direct metric comparison imperfect
> — but the approach here is more scientifically honest.

---

## Comparison Table

| Source | Model/Approach | Metric | Benchmark Value | This Project | Delta |
|--------|---------------|--------|-----------------|--------------|-------|
| Ayaan0620 (GitHub, 2025) — Kepler ML Project | XGBoost (all physical features), 10-fold CV | ROC-AUC | 0.9870 | 0.9824 | -0.0046 |
| Devpost: Leakage-Free Kepler Classifier (2025) | Random Forest (raw physical features only), 3-class task | F1 (macro) | 0.8230 | 0.9107* | +0.0877 |
| Saha (2021) arXiv:2101.01904 | Decision Tree ensemble (~7.5% misclassification rate) | Accuracy | 0.9250 | 0.9354 | +0.0104 |

> \* The F1 comparison with Devpost is not direct: this project is binary (CONFIRMED vs FALSE POSITIVE) while Devpost solves a harder 3-class task (including CANDIDATE). The project's binary F1=0.9107 on the CONFIRMED class is expected to be higher.

---

## Interpretation

This project's final XGBoost model (F1=0.9107 on the CONFIRMED class, ROC-AUC=0.9824)
is directly comparable to the leakage-free GitHub benchmark (Ayaan0620, 2025:
AUC=0.9870, XGBoost, binary task) and is within 0.005 AUC of it, validating that 11
carefully selected physical features are sufficient for strong binary classification.
The Devpost leakage-free classifier (macro F1=0.823) solves a harder 3-class problem
(including CANDIDATE), so the lower reported metric is expected and does not indicate a
weaker model. Saha (2021) reports ~92.5% accuracy but likely includes `koi_fpflag_*`
and `koi_score` as features — the Devpost project directly quantified that including
these columns inflates macro F1 by +0.073, which explains the gap. This project's
explicit removal of all six leakage columns makes direct metric comparison imperfect
but ensures the model learns from genuine astrophysical signal rather than NASA's
pre-computed verdicts.

---

## Error Analysis

### Confusion Matrix at Threshold 0.61

```
                    Predicted FP    Predicted CONFIRMED
Actual FP               920                 48
Actual CONFIRMED         50                500
```

| Metric | Value |
|--------|-------|
| True Positives (TP) — CONFIRMED correctly identified | 500 |
| True Negatives (TN) — FALSE POSITIVE correctly identified | 920 |
| False Positives (FP) — FALSE POSITIVE predicted as CONFIRMED | 48 |
| False Negatives (FN) — CONFIRMED predicted as FALSE POSITIVE | 50 |
| False Positive Rate (FPR) | 0.0496 (4.96%) |
| False Negative Rate (FNR) | 0.0909 (9.09%) |
| Precision (CONFIRMED class) | 0.9124 |
| Recall (CONFIRMED class) | 0.9091 |

### What Do the Errors Mean Astronomically?

**48 False Positives (FP predicted as CONFIRMED):**
These are KOIs that are genuine astrophysical false positives (e.g., eclipsing
binaries, background stars) but the model predicted them as exoplanet candidates.
These are the more dangerous errors in a scientific context — they would send
astronomers to spend follow-up time on non-planet targets. At 4.96% FPR, this is
within acceptable bounds for a triage tool.

**50 False Negatives (CONFIRMED predicted as FALSE POSITIVE):**
These are real confirmed exoplanets that the model incorrectly classified as false
positives. These have slightly higher numbers (50 vs 48) because the threshold was
deliberately tuned to 0.61 with a `recall >= 0.90` constraint, meaning the model
was allowed to miss up to 10% of confirmed planets while maximizing F1. The 9.09%
false negative rate satisfies this constraint exactly.

### Why Do These Errors Occur?

The 11 features used (stellar parameters + transit properties) have overlapping
distributions for difficult borderline cases:

1. **Low-SNR transits**: KOIs with `koi_model_snr` near 10–20 are ambiguous — the
   transit signal is weak enough to be consistent with either a small planet or
   instrumental noise.
2. **Grazing transits**: High impact parameter (`koi_impact` > 0.8) creates shallow,
   V-shaped transits resembling eclipsing binary stars — a classic false positive
   source.
3. **Giant star hosts**: Large `koi_srad` (> 3 solar radii) makes transit depth
   (`koi_depth`) inconsistent — a deep transit on a giant star is a different
   physical scenario than the same depth on a Sun-like star.
4. **Feature boundary cases**: The top SHAP feature `koi_prad` shows strong
   separability above ~3 Earth radii, but sub-Earth and super-Earth sizes (1–4 R⊕)
   overlap significantly between confirmed planets and false positives.

### Threshold Sensitivity

The threshold was optimized from the default 0.50 to 0.61 using OOF predictions:

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.50 | 0.883 | 0.927 | 0.904 |
| **0.61** | **0.912** | **0.909** | **0.911** |
| 0.70 | 0.938 | 0.878 | 0.907 |

Moving from 0.50 to 0.61 improved precision by +2.9 pp at a cost of only 1.8 pp
recall, resulting in the best F1 while maintaining recall above the 0.90 constraint.

---

## My Model Metrics (for reference)

| Metric | Value |
|--------|-------|
| accuracy | 0.9354 |
| precision | 0.9124 |
| recall | 0.9091 |
| f1 | 0.9107 |
| roc_auc | 0.9824 |
| pr_auc | 0.9661 |

---

## Source Citations

### 1. Ayaan0620 (GitHub, 2025)
- **Citation**: Ayaan0620/Kepler-MachineLearning-Project, GitHub 2025
- **URL**: https://github.com/Ayaan0620/Kepler-MachineLearning-Project
- **Model**: XGBoost (all physical features), 10-fold stratified CV
- **Features**: Physical measurements + error bars (~20–30 columns)
- **Dataset**: Cumulative KOI table — binary CONFIRMED vs FP
- **Leakage**: ✅ Clean — no koi_score, koi_pdisposition, koi_fpflag_*

### 2. Devpost: Leakage-Free Kepler Exoplanet Classifier (2025)
- **Citation**: Leakage-Free Kepler Exoplanet Classifier, Devpost Hackathon 2025
- **URL**: https://devpost.com/software/leakage-free-kepler-exoplanet-classifier
- **Model**: Random Forest (raw physical features only), 3-class task
- **Features**: Raw physical features only (~11–15 columns)
- **Dataset**: Cumulative KOI table — 3-class (CONFIRMED, FALSE POSITIVE, CANDIDATE)
- **Leakage**: ✅ Explicitly leakage-aware; koi_fpflag_* isolated to comparison only. WITH flags: macro F1=0.896, WITHOUT: 0.823 (delta=+0.073 quantified)

### 3. Saha (2021) arXiv:2101.01904
- **Citation**: Saha 2021, arXiv:2101.01904
- **URL**: https://arxiv.org/abs/2101.01904
- **Model**: Decision Tree ensemble (~7.5% misclassification rate)
- **Features**: Tabular KOI features including flag columns
- **Dataset**: Cumulative KOI table — binary classification
- **Leakage**: ⚠️ Likely leaked — paper includes koi_fpflag_nt, koi_fpflag_ss as features; koi_score mentioned as important

---

*Chart: `plots/benchmarking/benchmark_comparison.png` — run `python src/task3_benchmarking.py` to generate.*
