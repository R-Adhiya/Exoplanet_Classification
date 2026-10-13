---
title: Kepler Exoplanet Classification API
emoji: 🪐
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
license: mit
short_description: FastAPI inference API for Kepler KOI XGBoost classifier
---

# Kepler Exoplanet Classification API

FastAPI inference API for the Kepler KOI (Kepler Objects of Interest) XGBoost classifier.
Classifies a KOI as **CONFIRMED** exoplanet or **FALSE POSITIVE** using 11 stellar and transit features.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/predict` | Single KOI prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/docs` | Interactive Swagger UI |

## Sample Request

```bash
curl -X POST https://<your-space-url>/predict \
  -H "Content-Type: application/json" \
  -d '{"koi_period":9.49,"koi_duration":3.45,"koi_depth":506,
       "koi_prad":2.2,"koi_impact":0.12,"koi_model_snr":45.0,
       "koi_teq":865,"koi_insol":67,"koi_steff":5440,
       "koi_slogg":4.45,"koi_srad":0.93}'
```

## Features Used

`koi_period`, `koi_duration`, `koi_depth`, `koi_prad`, `koi_impact`,
`koi_model_snr`, `koi_teq`, `koi_insol`, `koi_steff`, `koi_slogg`, `koi_srad`

## Model

XGBoost with `scale_pos_weight=1.7611`, threshold 0.61.
F1=0.9107, PR-AUC=0.9661, ROC-AUC=0.9824 on held-out test set.
