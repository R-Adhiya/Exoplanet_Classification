# Phase 1 EDA Report
## Kepler Exoplanet Classification — Explainable ML System

*Generated from actual NASA Exoplanet Archive cumulative KOI table.*

---

## Dataset Overview

| Item | Value |
|------|-------|
| Total rows | 9,564 |
| Total columns | 153 |
| Labeled rows (CONFIRMED + FP) | 7,587 |
| Candidate rows | 1,977 |
| Source | NASA Exoplanet Archive cumulative KOI table |

---

## Target Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| CONFIRMED (target=1) | 2,748 | 36.2% |
| FALSE POSITIVE (target=0) | 4,839 | 63.8% |
| CANDIDATE (no label) | 1,977 | — |

---

## Leakage Columns Removed

The following columns were removed prior to feature analysis because they are derived from or directly encode the disposition decision:

- `koi_score`
- `koi_pdisposition`
- `koi_fpflag_nt`
- `koi_fpflag_ss`
- `koi_fpflag_co`
- `koi_fpflag_ec`

---

## Missingness

| Feature | Missing Count | Missing % |
|---------|--------------|-----------|
| koi_depth | 259 | 3.4% |
| koi_impact | 259 | 3.4% |
| koi_prad | 259 | 3.4% |
| koi_steff | 259 | 3.4% |
| koi_slogg | 259 | 3.4% |
| koi_model_snr | 259 | 3.4% |
| koi_teq | 259 | 3.4% |
| koi_srad | 259 | 3.4% |
| koi_insol | 219 | 2.9% |
| koi_duration | 0 | 0.0% |
| koi_period | 0 | 0.0% |

No features exceed 20% missing.

---

## Summary Statistics

See `reports/phase_1/summary_statistics.csv` for full descriptive statistics (count, mean, std, min, 25%, median, 75%, max, skewness) for all 11 features.

---

## Skewness

| Feature | Skewness |
|---------|----------|
| koi_prad | 53.72 |
| koi_insol | 44.68 |
| koi_impact | 25.18 |
| koi_srad | 20.42 |
| koi_duration | 6.12 |

Features with |skewness| > 1 are candidates for log or power transformation in Phase 2. No transformations are applied in Phase 1.

---

## Correlation Analysis

Pearson correlation was computed for all 11 features on the labeled dataset.

**Notable pairs (|r| > 0.5):**

| Feature A | Feature B | r |
|-----------|-----------|---|
| koi_prad | koi_impact | 0.680 |
| koi_slogg | koi_srad | -0.648 |
| koi_insol | koi_srad | 0.579 |
| koi_depth | koi_model_snr | 0.572 |
| koi_teq | koi_slogg | -0.567 |

No features are automatically removed based on correlation alone; decisions will be made in Phase 2 with domain context.

---

## Distribution Analysis

Histogram and KDE plots were generated for: `koi_period`, `koi_depth`, `koi_prad`, `koi_duration`.

See `plots/phase_1/dist_*.png` for visualisations.

---

## Class Separability

| Feature | Median CONFIRMED | Median FP | Mann-Whitney p |
|---------|-----------------|-----------|----------------|
| koi_period | 11.3556 | 5.2439 | 0.0 |
| koi_depth | 448.8 | 575.95 | 0.0 |
| koi_prad | 2.16 | 8.97 | 0.0 |
| koi_duration | 3.492 | 4.057 | 0.0 |

Mann-Whitney p-values indicate whether the class distributions differ statistically. A small p-value suggests different distributions, but does not guarantee predictive separability on its own.

---

## Data Quality Issues

1. **Missingness**: Some features have missing values; imputation strategy    to be determined in Phase 2.
2. **Skewness**: Several features are heavily right-skewed; transformation    candidates flagged for Phase 2.
3. **Outliers**: Extreme values present in period, depth, and radius;    clipping/Winsorising strategy deferred to Phase 2.
4. **Class imbalance**: Present — will require class-weighting or resampling in Phase 2.

---

## Physical Interpretation

This section provides cautious physical context for the observed feature distributions based on transit photometry principles.

- **Transit depth (`koi_depth`)**: Measures the fractional flux decrease   during transit. Planets produce characteristically shallower,   more consistent depths than eclipsing binaries. Differences in depth   distributions between classes may reflect this, but overlap is expected.

- **Planetary radius (`koi_prad`)**: Derived from transit depth and stellar   radius. True exoplanets cluster at sub-Neptune to Jupiter radii;   very large inferred radii often indicate false positives   (diluted eclipsing binaries). Observed class differences are consistent   with this, but EDA alone does not confirm exoplanet status.

- **Transit duration (`koi_duration`)**: Depends on orbital velocity and   stellar radius. Extremely short or long durations can indicate   non-planetary scenarios. Both classes show broad distributions.

- **Orbital period (`koi_period`)**: Kepler's sampling window favours   shorter-period planets. The distribution reflects both detection bias   and physical occurrence rates; false positives can occur at any period.

- **Signal-to-noise ratio (`koi_model_snr`)**: Higher SNR signals tend to   be genuine transits. Confirmed planets are expected to show higher   median SNR than false positives.

- **Stellar radius (`koi_srad`)**: Affects derived planetary radius and   transit depth interpretation. Uncertainty in stellar parameters   propagates to planetary parameters.

**Important caveat**: These physical interpretations are consistent with known transit science but cannot be confirmed from EDA alone. The ML model in later phases will learn from the joint feature space rather than any single physical argument.

---

## Phase 1 Conclusion

- Dataset loaded: **9,564 KOI objects** across **153 columns**
- Labeled for modelling: **7,587** (2,748 CONFIRMED, 4,839 FALSE POSITIVE)
- Held out for prediction: **1,977 CANDIDATEs**
- All 11 feature columns validated as present
- All 6 leakage columns removed
- Missingness, skewness, and correlation quantified (no transformations applied)
- Processed datasets and feature list saved

**Phase 1 is complete. Phase 2 will handle: imputation, transformation, train/test split, and baseline modelling.**