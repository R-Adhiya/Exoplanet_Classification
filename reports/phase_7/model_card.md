# Model Card
## Kepler Exoplanet Classification

---

## Model

- **Type**: XGBoost with class_weight imbalance strategy
- **Version**: Phase 7 finalized model
- **Saved as**: `models/final/final_model.pkl`

---

## Intended Use

Classify NASA Kepler Objects of Interest (KOIs) into CONFIRMED or FALSE POSITIVE categories. The system is intended as an exploratory machine-learning analysis tool. It is NOT a replacement for peer-reviewed astronomical analysis.

---

## Input Features

11 preprocessed (imputed, log-transformed where applicable, standardized) numerical features:

- `koi_period`: Orbital period
- `koi_duration`: Transit duration
- `koi_depth`: Transit depth
- `koi_prad`: Planetary radius estimate
- `koi_impact`: Transit impact parameter
- `koi_model_snr`: Signal-to-noise ratio
- `koi_teq`: Equilibrium temperature
- `koi_insol`: Stellar insolation
- `koi_steff`: Stellar effective temperature
- `koi_slogg`: Stellar surface gravity
- `koi_srad`: Stellar radius

---

## Output

- **Predicted class**: CONFIRMED (1) or FALSE POSITIVE (0)
- **Predicted probability**: P(CONFIRMED)
- **Classification threshold**: 0.61

---

## Training Data

NASA Exoplanet Archive Cumulative KOI Table (cumulative_kois.csv, downloaded September 2026).
Labeled set: 7,587 rows (2,748 CONFIRMED, 4,839 FALSE POSITIVE).
Train split: 6,069 rows (80%).

---

## Evaluation Method

- 5-fold stratified cross-validation for model/threshold selection
- Untouched 20% test set (1,518 rows) for final evaluation
- Primary metric: PR-AUC (Average Precision)

---

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 0.9354 |
| Precision | 0.9124 |
| Recall | 0.9091 |
| F1 | 0.9107 |
| ROC-AUC | 0.9824 |
| PR-AUC | 0.9661 |
| Brier Score | 0.0461 |

---

## Threshold

Classification threshold: **0.61**

Selected to maximize F1 subject to recall >= 0.90, using 5-fold out-of-fold predictions on training data.

---

## Limitations

1. Labels are sourced from the NASA Kepler KOI catalogue, which itself reflects the state of analysis at time of download.
2. The model may not generalize to KOI observations from other telescopes or with different noise characteristics.
3. Probability outputs are not perfectly calibrated.
4. Feature distributions in the training set may not represent all possible planetary systems.

---

## Ethical and Scientific Considerations

**This model is a machine-learning classifier, not a scientific confirmation instrument.**

- A prediction of CONFIRMED by this model means the model assigns a high probability of agreement with the existing KOI CONFIRMED label. It does not constitute independent scientific confirmation of the existence of an exoplanet.
- A prediction of FALSE POSITIVE does not mean the object has been scientifically ruled out as an exoplanet.
- All candidate predictions should be treated as preliminary machine-learning outputs, subject to expert review.
- The system should not be used as the sole basis for scientific publications or follow-up observation decisions.