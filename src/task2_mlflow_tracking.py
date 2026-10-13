"""
TASK 2: Experiment Tracking with MLflow
Kepler Exoplanet Classification -- Explainable ML System

Re-runs (or re-logs) every model/strategy combination with MLflow tracking:
  Experiment 1: "Baseline Models"          -- LR, DT, RF, XGB (no imbalance handling)
  Experiment 2: "Imbalance Strategies"     -- 4 models x 3 strategies (12 runs)
  Experiment 3: "Final Model"              -- best XGBoost with class weighting

For each run logs:
  - All hyperparameters
  - precision, recall, F1 per class, PR-AUC
  - Trained model as MLflow artifact

Uses a local ./mlruns tracking URI -- no external account required.
Run `mlflow ui` from the project root to view all experiments.

Does NOT change which model was selected as final.
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

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

# Use SQLite backend (MLflow 2.20+ deprecated the file store)
# View UI: mlflow ui --backend-store-uri sqlite:///mlflow.db
DB_PATH = ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
print(f"[INFO] MLflow tracking URI: {mlflow.get_tracking_uri()}")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("[INFO] Loading preprocessed data...")
X_train = pd.read_csv(DATA_PROC / "X_train_processed.csv").values
X_test  = pd.read_csv(DATA_PROC / "X_test_processed.csv").values
y_train = pd.read_csv(DATA_PROC / "y_train.csv").squeeze().values
y_test  = pd.read_csv(DATA_PROC / "y_test.csv").squeeze().values
print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

# Class imbalance ratio (for scale_pos_weight)
neg   = int((y_train == 0).sum())
pos   = int((y_train == 1).sum())
spw   = round(neg / pos, 4)   # 1.7611 for this dataset
print(f"  Class balance: {neg} FP / {pos} CONFIRMED, scale_pos_weight={spw}")

# ---------------------------------------------------------------------------
# Helper: compute and return all metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob):
    """Return dict of metrics for a binary classification result."""
    pr_auc  = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    report  = classification_report(y_true, y_pred, output_dict=True)

    return {
        "pr_auc":             pr_auc,
        "roc_auc":            roc_auc,
        "accuracy":           report["accuracy"],
        # Class 0 = FALSE POSITIVE
        "precision_fp":       report["0"]["precision"],
        "recall_fp":          report["0"]["recall"],
        "f1_fp":              report["0"]["f1-score"],
        # Class 1 = CONFIRMED
        "precision_conf":     report["1"]["precision"],
        "recall_conf":        report["1"]["recall"],
        "f1_conf":            report["1"]["f1-score"],
        # Macro averages
        "f1_macro":           report["macro avg"]["f1-score"],
        "precision_macro":    report["macro avg"]["precision"],
        "recall_macro":       report["macro avg"]["recall"],
    }


def log_run(experiment_name, run_name, model, params, X_tr, y_tr, X_te, y_te,
            tags=None):
    """Train model, log params+metrics+artifact to MLflow, return metrics dict."""
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        # Train
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        metrics = compute_metrics(y_te, y_pred, y_prob)

        # Log params
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model artifact
        # Pass skops_trusted_types for tree-based models (MLflow 2.20+ security requirement)
        _SKOPS_TRUSTED = [
            "sklearn.tree._tree.Tree",
            "sklearn.tree._classes.DecisionTreeClassifier",
            "sklearn.ensemble._forest.RandomForestClassifier",
            "numpy.core.multiarray.scalar",
            "numpy.dtype",
            "numpy.ndarray",
            "builtins.float",
            "builtins.int",
            "builtins.bool",
            "builtins.str",
            "builtins.list",
            "builtins.dict",
            "builtins.tuple",
            "builtins.NoneType",
        ]
        if isinstance(model, XGBClassifier):
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model",
                                     skops_trusted_types=_SKOPS_TRUSTED)

        # Log tags
        if tags:
            mlflow.set_tags(tags)

        run_id = run.info.run_id

    print(f"  [{run_id[:8]}] {experiment_name}/{run_name}  "
          f"PR-AUC={metrics['pr_auc']:.4f}  "
          f"F1_CONF={metrics['f1_conf']:.4f}")
    return metrics


# ===========================================================================
# EXPERIMENT 1: Baseline Models (no imbalance handling)
# ===========================================================================
print("\n[INFO] === EXPERIMENT 1: Baseline Models ===")

baseline_configs = [
    ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42, C=1.0),
     {"model_type": "LogisticRegression", "C": 1.0, "max_iter": 1000,
      "class_weight": "none", "random_state": 42}),

    ("Decision Tree", DecisionTreeClassifier(max_depth=None, random_state=42),
     {"model_type": "DecisionTree", "max_depth": "None",
      "class_weight": "none", "random_state": 42}),

    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
     {"model_type": "RandomForest", "n_estimators": 100,
      "class_weight": "none", "random_state": 42}),

    ("XGBoost", XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss",
                               verbosity=0),
     {"model_type": "XGBoost", "n_estimators": 100, "scale_pos_weight": 1.0,
      "imbalance_strategy": "none", "random_state": 42}),
]

for name, model, params in baseline_configs:
    log_run(
        experiment_name="Baseline Models",
        run_name=name,
        model=model,
        params=params,
        X_tr=X_train, y_tr=y_train,
        X_te=X_test,  y_te=y_test,
        tags={"phase": "3", "strategy": "baseline"},
    )

# ===========================================================================
# EXPERIMENT 2: Imbalance Strategies
# ===========================================================================
print("\n[INFO] === EXPERIMENT 2: Imbalance Strategies ===")

imbalance_configs = []

# -- 2a. Class Weighting --
imbalance_configs += [
    ("LR ClassWeight", "class_weight",
     LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
     {"model_type": "LogisticRegression", "strategy": "class_weight",
      "class_weight": "balanced", "max_iter": 1000}),

    ("DT ClassWeight", "class_weight",
     DecisionTreeClassifier(random_state=42, class_weight="balanced"),
     {"model_type": "DecisionTree", "strategy": "class_weight",
      "class_weight": "balanced"}),

    ("RF ClassWeight", "class_weight",
     RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced",
                             n_jobs=-1),
     {"model_type": "RandomForest", "strategy": "class_weight",
      "n_estimators": 100, "class_weight": "balanced"}),

    ("XGB ClassWeight", "class_weight",
     XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=spw,
                   eval_metric="logloss", verbosity=0),
     {"model_type": "XGBoost", "strategy": "class_weight",
      "scale_pos_weight": spw, "n_estimators": 100}),
]

# -- 2b. SMOTE (fit on full X_train here; in CV it was inside folds) --
smote = SMOTE(random_state=42, k_neighbors=5)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
print(f"  SMOTE resampled: {X_smote.shape} ({int((y_smote==1).sum())} CONFIRMED, "
      f"{int((y_smote==0).sum())} FP)")

imbalance_configs += [
    ("LR SMOTE", "SMOTE",
     LogisticRegression(max_iter=1000, random_state=42),
     {"model_type": "LogisticRegression", "strategy": "SMOTE",
      "smote_k_neighbors": 5, "max_iter": 1000}),

    ("DT SMOTE", "SMOTE",
     DecisionTreeClassifier(random_state=42),
     {"model_type": "DecisionTree", "strategy": "SMOTE",
      "smote_k_neighbors": 5}),

    ("RF SMOTE", "SMOTE",
     RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
     {"model_type": "RandomForest", "strategy": "SMOTE",
      "n_estimators": 100, "smote_k_neighbors": 5}),

    ("XGB SMOTE", "SMOTE",
     XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss",
                   verbosity=0),
     {"model_type": "XGBoost", "strategy": "SMOTE",
      "n_estimators": 100, "smote_k_neighbors": 5}),
]

for name, strategy, model, params in imbalance_configs:
    X_tr = X_smote if strategy == "SMOTE" else X_train
    y_tr = y_smote if strategy == "SMOTE" else y_train
    log_run(
        experiment_name="Imbalance Strategies",
        run_name=name,
        model=model,
        params=params,
        X_tr=X_tr, y_tr=y_tr,
        X_te=X_test, y_te=y_test,
        tags={"phase": "5", "strategy": strategy},
    )

# ===========================================================================
# EXPERIMENT 3: Final Model
# ===========================================================================
print("\n[INFO] === EXPERIMENT 3: Final Model ===")

# Load the already-saved final model and log it (don't retrain from scratch --
# instead replicate its exact hyperparameters so MLflow has a canonical record)
final_meta_path = ROOT / "models" / "final" / "final_model_metadata.json"
with open(final_meta_path) as f:
    final_meta = json.load(f)
print(f"  Final model metadata: {final_meta}")

# Re-create with same hyperparameters for the MLflow run
final_model = XGBClassifier(
    n_estimators=100,
    scale_pos_weight=spw,
    eval_metric="logloss",
    verbosity=0,
    random_state=42,
)
final_params = {
    "model_type":         "XGBoost",
    "strategy":           "class_weight_scale_pos_weight",
    "scale_pos_weight":   spw,
    "n_estimators":       100,
    "threshold":          0.61,
    "threshold_method":   "OOF F1 maximized, recall >= 0.90 constraint",
    "random_state":       42,
    "selected_as_final":  True,
}

mlflow.set_experiment("Final Model")
with mlflow.start_run(run_name="XGBoost Final (threshold=0.61)") as run:
    final_model.fit(X_train, y_train)
    y_prob_final = final_model.predict_proba(X_test)[:, 1]

    # Apply threshold 0.61 for metrics (same as Phase 7)
    y_pred_final = (y_prob_final >= 0.61).astype(int)
    metrics_final = compute_metrics(y_test, y_pred_final, y_prob_final)

    # Also add threshold-specific metrics from the saved results
    metrics_final["threshold"] = 0.61

    mlflow.log_params(final_params)
    mlflow.log_metrics(metrics_final)
    mlflow.xgboost.log_model(final_model, artifact_path="model")
    mlflow.set_tags({
        "phase":    "7",
        "strategy": "class_weight",
        "final":    "true",
        "deployed": "Streamlit Community Cloud",
    })
    print(f"  [{run.info.run_id[:8]}] Final Model  "
          f"PR-AUC={metrics_final['pr_auc']:.4f}  "
          f"F1_CONF={metrics_final['f1_conf']:.4f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 2 COMPLETE -- MLFLOW EXPERIMENT TRACKING")
print("=" * 60)
print(f"  Tracking URI: {mlflow.get_tracking_uri()}")
print(f"  mlruns dir  : {MLRUNS_DIR}")
print()
print("Experiments created:")
print("  1. Baseline Models          (4 runs)")
print("  2. Imbalance Strategies     (8 runs)")
print("  3. Final Model              (1 run)")
print()
print("To view the MLflow UI, run from the project root:")
print("  mlflow ui --backend-store-uri mlruns")
print("Then open: http://127.0.0.1:5000")
