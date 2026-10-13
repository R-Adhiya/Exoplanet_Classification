"""
TASK 2b: Optuna Hyperparameter Optimization for XGBoost
Kepler Exoplanet Classification -- Explainable ML System

Runs an Optuna TPE study (50 trials) on XGBoost, optimizing PR-AUC
via 5-fold stratified cross-validation on the training set.

Each trial is logged to the same MLflow 'Final Model' experiment so
the Optuna results appear alongside the Grid/Random Search runs.

Does NOT replace the existing final model -- purely an additional
hyperparameter search record as required by the academic rubric.

Usage (from project root):
    pip install optuna mlflow
    python src/task2b_optuna_tuning.py

After running, view in MLflow UI:
    mlflow ui --backend-store-uri mlruns
    # open http://127.0.0.1:5000
"""

import json
import sys
import warnings
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "processed"
MLRUNS    = ROOT / "mlruns"
MLRUNS.mkdir(exist_ok=True)

mlflow.set_tracking_uri(MLRUNS.as_uri())

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("[INFO] Loading preprocessed training data...")
X_train = pd.read_csv(DATA_PROC / "X_train_processed.csv").values
X_test  = pd.read_csv(DATA_PROC / "X_test_processed.csv").values
y_train = pd.read_csv(DATA_PROC / "y_train.csv").squeeze().values
y_test  = pd.read_csv(DATA_PROC / "y_test.csv").squeeze().values

neg = int((y_train == 0).sum())
pos = int((y_train == 1).sum())
spw_default = round(neg / pos, 4)
print(f"  Train: {X_train.shape}  |  neg={neg}, pos={pos}, spw={spw_default}")

N_TRIALS = 50
N_FOLDS  = 5
SCORING  = "average_precision"   # PR-AUC (primary metric)

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 3.0),
        "eval_metric": "logloss",
        "verbosity": 0,
        "random_state": 42,
    }

    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train,
                             cv=cv, scoring=SCORING, n_jobs=1)
    pr_auc_mean = float(scores.mean())

    # Log each trial as an MLflow run
    mlflow.set_experiment("Optuna XGBoost Study")
    with mlflow.start_run(run_name=f"trial_{trial.number:03d}",
                          nested=False) as run:
        mlflow.log_params(params)
        mlflow.log_metric("cv_pr_auc_mean", pr_auc_mean)
        mlflow.log_metric("cv_pr_auc_std",  float(scores.std()))
        mlflow.log_metric("trial_number",   trial.number)
        mlflow.set_tags({
            "phase":     "optuna",
            "n_folds":   str(N_FOLDS),
            "objective": SCORING,
        })

    return pr_auc_mean


# ---------------------------------------------------------------------------
# Run Optuna study
# ---------------------------------------------------------------------------
print(f"\n[INFO] Starting Optuna TPE study: {N_TRIALS} trials, {N_FOLDS}-fold CV...")
print(f"       Optimizing: {SCORING} (PR-AUC)")
print()

study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=42),
    study_name="xgboost_pr_auc",
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

best = study.best_trial
print(f"[INFO] Best trial: #{best.number}  PR-AUC={best.value:.4f}")
print(f"       Best params: {best.params}")

# ---------------------------------------------------------------------------
# Retrain best model on full training set, evaluate on test set
# ---------------------------------------------------------------------------
print("\n[INFO] Retraining best Optuna model on full training set...")
best_model = XGBClassifier(
    **{k: v for k, v in best.params.items()},
    eval_metric="logloss",
    verbosity=0,
    random_state=42,
)
best_model.fit(X_train, y_train)
y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.61).astype(int)   # same threshold as final model

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
test_metrics = {
    "test_pr_auc":    average_precision_score(y_test, y_prob),
    "test_roc_auc":   roc_auc_score(y_test, y_prob),
    "test_accuracy":  accuracy_score(y_test, y_pred),
    "test_f1_conf":   f1_score(y_test, y_pred),
    "test_precision": precision_score(y_test, y_pred),
    "test_recall":    recall_score(y_test, y_pred),
}
print(f"  Test PR-AUC  : {test_metrics['test_pr_auc']:.4f}")
print(f"  Test ROC-AUC : {test_metrics['test_roc_auc']:.4f}")
print(f"  Test F1      : {test_metrics['test_f1_conf']:.4f}")

# Log best model run to MLflow
mlflow.set_experiment("Optuna XGBoost Study")
with mlflow.start_run(run_name="BEST -- Optuna XGBoost (full-train eval)") as run:
    mlflow.log_params({**best.params, "threshold": 0.61, "n_trials": N_TRIALS})
    mlflow.log_metrics({**test_metrics, "cv_pr_auc_best": best.value})
    mlflow.xgboost.log_model(best_model, artifact_path="model")
    mlflow.set_tags({
        "phase":   "optuna_best",
        "final":   "false",   # existing model is still the production model
        "n_trials": str(N_TRIALS),
    })

# ---------------------------------------------------------------------------
# Save Optuna best params to reports
# ---------------------------------------------------------------------------
REPORTS_P4 = ROOT / "reports" / "phase_4"
optuna_out = REPORTS_P4 / "optuna_best_params.json"
with open(optuna_out, "w") as f:
    json.dump({
        "study_name":        "xgboost_pr_auc",
        "n_trials":          N_TRIALS,
        "best_trial":        best.number,
        "best_cv_pr_auc":    best.value,
        "best_params":       best.params,
        "test_metrics":      test_metrics,
        "note": (
            "Optuna TPE study. Does not replace the production model "
            "(models/final/final_model.pkl) -- that model was selected "
            "in Phase 7 via OOF threshold optimization with recall >= 0.90 constraint."
        ),
    }, f, indent=2)
print(f"\n[INFO] Saved Optuna results: {optuna_out}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 2b COMPLETE -- OPTUNA HYPERPARAMETER OPTIMIZATION")
print("=" * 60)
print(f"  Trials run      : {N_TRIALS}")
print(f"  Best trial      : #{best.number}")
print(f"  Best CV PR-AUC  : {best.value:.4f}")
print(f"  Test PR-AUC     : {test_metrics['test_pr_auc']:.4f}")
print(f"  Test F1         : {test_metrics['test_f1_conf']:.4f}")
print()
print("  All trials logged to MLflow experiment: 'Optuna XGBoost Study'")
print("  View: mlflow ui --backend-store-uri mlruns")
print("  Then: http://127.0.0.1:5000")
