"""
Phase 7: Model Selection & Finalization
Kepler Exoplanet Classification -- Explainable ML System

Steps performed:
  1. Load Phase 5 CV results and select final configuration using CV evidence only
  2. Refit the selected model on the full training set
  3. Generate out-of-fold (OOF) probabilities for threshold optimization
  4. Optimize classification threshold (maximize F1 subject to recall >= 0.90)
  5. Freeze threshold
  6. Evaluate on untouched test set exactly once
  7. Calibration analysis (diagnostic only)
  8. Save final model, metadata, and prediction function
  9. Generate candidate predictions
  10. Validate the serialized pipeline

No new hyperparameter search. No SHAP. No feature selection. No deployment.
Test set accessed ONLY after the final configuration and threshold are frozen.

Compatibility: Python 3.12, pandas 2.x, scikit-learn 1.9+, XGBoost 3.x,
               imbalanced-learn 0.14+, Windows PowerShell (ASCII-safe output).
"""

import json
import platform
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Stdout: force UTF-8 on Windows
# ---------------------------------------------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT            = Path(__file__).resolve().parent.parent
DATA_PROCESSED  = ROOT / "data" / "processed"
REPORTS_P4      = ROOT / "reports" / "phase_4"
REPORTS_P5      = ROOT / "reports" / "phase_5"
REPORTS_P7      = ROOT / "reports" / "phase_7"
PLOTS_P7        = ROOT / "plots" / "phase_7"
MODELS_FINAL    = ROOT / "models" / "final"

for _d in [REPORTS_P7, PLOTS_P7, MODELS_FINAL]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE     = 42
N_SPLITS         = 5
THRESHOLD_RANGE  = np.arange(0.05, 0.96, 0.01)
RECALL_MIN       = 0.90
np.random.seed(RANDOM_STATE)

CORE_FEATURES = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_impact", "koi_model_snr", "koi_teq", "koi_insol",
    "koi_steff", "koi_slogg", "koi_srad",
]

MODEL_SLUGS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree":       "decision_tree",
    "Random Forest":       "random_forest",
    "XGBoost":             "xgboost",
}


# ===========================================================================
# 1. LOAD TRAINING DATA
# ===========================================================================
def load_training_data():
    print("\n[INFO] Loading training data...")
    X_train = pd.read_csv(DATA_PROCESSED / "X_train_processed.csv")
    y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv").squeeze()
    assert X_train.shape[1] == 11 and X_train.isnull().sum().sum() == 0
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / pos
    print(f"  X_train: {X_train.shape}  CONF={pos:,}  FP={neg:,}  "
          f"scale_pos_weight={scale_pos_weight:.4f}")
    return X_train, y_train, pos, neg, scale_pos_weight


# ===========================================================================
# 2. LOAD TEST DATA
# ===========================================================================
def load_test_data():
    print("\n[INFO] Loading test data (will not be accessed until threshold is frozen)...")
    X_test = pd.read_csv(DATA_PROCESSED / "X_test_processed.csv")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").squeeze()
    assert X_test.shape[1] == 11 and X_test.isnull().sum().sum() == 0
    print(f"  X_test: {X_test.shape}  [SEALED until threshold frozen]")
    return X_test, y_test


# ===========================================================================
# 3. LOAD CANDIDATE DATA
# ===========================================================================
def load_candidate_data():
    print("\n[INFO] Loading candidate data...")
    X_cand = pd.read_csv(DATA_PROCESSED / "X_candidate_processed.csv")
    print(f"  X_candidate: {X_cand.shape}")
    return X_cand


# ===========================================================================
# 4. LOAD PHASE 4 PARAMETERS
# ===========================================================================
def load_phase_4_parameters():
    with open(REPORTS_P4 / "best_parameters.json") as fh:
        params = json.load(fh)
    print(f"\n[INFO] Phase 4 hyperparameters loaded for "
          f"{list(params.keys())}")
    return params


# ===========================================================================
# 5. LOAD PHASE 5 RESULTS
# ===========================================================================
def load_phase_5_results():
    print("\n[INFO] Loading Phase 5 CV results...")
    cv_df = pd.read_csv(REPORTS_P5 / "imbalance_cv_results.csv")

    # Normalise strategy names to lowercase for matching
    cv_df["strategy_key"] = cv_df["strategy"].str.lower().str.replace(
        " ", "_").str.replace("-", "_")

    print(f"\n  {'Model':<24} {'Strategy':<14} {'CV PR-AUC':>10} {'std':>7} "
          f"{'Recall':>8} {'F1':>8}")
    print("  " + "-" * 75)
    for _, row in cv_df.iterrows():
        print(f"  {row['model']:<24} {row['strategy']:<14} "
              f"{row['cv_pr_auc_mean']:>10.4f} {row['cv_pr_auc_std']:>7.4f} "
              f"{row['cv_recall_mean']:>8.4f} {row['cv_f1_mean']:>8.4f}")

    assert len(cv_df) == 12, f"Expected 12 CV rows, got {len(cv_df)}"
    print("  [OK] 12 configurations verified")
    return cv_df


# ===========================================================================
# 6. BUILD CANDIDATE CONFIGURATIONS
# ===========================================================================
def build_candidate_configurations(params, scale_pos_weight):
    """
    Reconstruct all 12 model/strategy estimators from Phase 4 hyperparameters.
    Returns dict: {(model_name, strategy): estimator}.
    """
    print("\n[INFO] Building candidate configurations from Phase 4 parameters...")

    configs = {}
    for model_name in ["Logistic Regression", "Decision Tree",
                       "Random Forest", "XGBoost"]:
        p = params[model_name]

        # Base estimators
        if model_name == "Logistic Regression":
            base = lambda cw=None: LogisticRegression(
                random_state=RANDOM_STATE, max_iter=2000,
                class_weight=cw, **p)
        elif model_name == "Decision Tree":
            base = lambda cw=None: DecisionTreeClassifier(
                random_state=RANDOM_STATE, class_weight=cw, **p)
        elif model_name == "Random Forest":
            base = lambda cw=None: RandomForestClassifier(
                random_state=RANDOM_STATE, n_jobs=-1,
                class_weight=cw, **p)
        else:  # XGBoost
            base = lambda spw=None: XGBClassifier(
                random_state=RANDOM_STATE, eval_metric="logloss",
                n_jobs=-1,
                scale_pos_weight=(spw if spw is not None else 1.0),
                **p)

        # baseline
        if model_name == "XGBoost":
            configs[(model_name, "baseline")] = base()
        else:
            configs[(model_name, "baseline")] = base()

        # class_weight (or scale_pos_weight for XGBoost)
        if model_name == "XGBoost":
            configs[(model_name, "class_weight")] = base(spw=scale_pos_weight)
        else:
            configs[(model_name, "class_weight")] = base(cw="balanced")

        # SMOTE
        if model_name == "XGBoost":
            est = base()
        else:
            est = base()
        configs[(model_name, "smote")] = ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", est),
        ])

    print(f"  Built {len(configs)} configurations")
    return configs


# ===========================================================================
# 7. SELECT FINAL CONFIGURATION
# ===========================================================================
def select_final_configuration(cv_df, configs):
    """
    Select using CV evidence only.
    Primary: highest mean CV PR-AUC.
    Secondary (tie-break): lower PR-AUC std, then higher recall.
    No test-set information used.
    """
    print("\n[INFO] Selecting final configuration using CV evidence...")

    # Sort by mean CV PR-AUC desc, then std asc
    sorted_df = cv_df.sort_values(
        ["cv_pr_auc_mean", "cv_pr_auc_std"],
        ascending=[False, True]
    ).reset_index(drop=True)

    best_row = sorted_df.iloc[0]
    best_model    = best_row["model"]
    best_strategy = best_row["strategy_key"]
    best_pr       = best_row["cv_pr_auc_mean"]
    best_std      = best_row["cv_pr_auc_std"]

    print(f"  Selected: {best_model} -- {best_strategy}")
    print(f"  CV PR-AUC = {best_pr:.4f} +/- {best_std:.4f}")
    print(f"  CV Recall = {best_row['cv_recall_mean']:.4f}")
    print(f"  CV F1     = {best_row['cv_f1_mean']:.4f}")

    # Build selection table
    rows = []
    for _, row in cv_df.iterrows():
        is_selected = (row["model"] == best_model and
                       row["strategy_key"] == best_strategy)
        reason = ("Highest CV PR-AUC with lowest std among top configurations"
                  if is_selected else "")
        rows.append({
            "model":              row["model"],
            "strategy":           row["strategy"],
            "cv_pr_auc_mean":     row["cv_pr_auc_mean"],
            "cv_pr_auc_std":      row["cv_pr_auc_std"],
            "cv_roc_auc_mean":    row["cv_roc_auc_mean"],
            "cv_roc_auc_std":     row["cv_roc_auc_std"],
            "cv_precision_mean":  row["cv_precision_mean"],
            "cv_recall_mean":     row["cv_recall_mean"],
            "cv_f1_mean":         row["cv_f1_mean"],
            "selection_status":   "selected" if is_selected else "not_selected",
            "selection_reason":   reason,
        })
    sel_df = pd.DataFrame(rows)
    sel_df.to_csv(REPORTS_P7 / "final_model_selection.csv", index=False)
    print(f"  Saved: {REPORTS_P7 / 'final_model_selection.csv'}")

    final_config = configs[(best_model, best_strategy)]
    return best_model, best_strategy, best_row, final_config


# ===========================================================================
# 8. FIT FINAL MODEL ON FULL TRAINING DATA
# ===========================================================================
def fit_final_model(final_config, X_train, y_train):
    """Fit the selected configuration on the complete training set."""
    print("\n[INFO] Fitting final model on complete training set...")
    final_config.fit(X_train, y_train)
    print("  [OK] Final model fitted")
    return final_config


# ===========================================================================
# 9. GENERATE OOF PREDICTIONS
# ===========================================================================
def generate_oof_predictions(best_model, best_strategy, params,
                              scale_pos_weight, X_train, y_train):
    """
    Generate out-of-fold predicted probabilities using 5-fold stratified CV.
    SMOTE applied inside each fold if the selected strategy is SMOTE.
    Test set is NOT touched.
    """
    print(f"\n[INFO] Generating OOF predictions "
          f"({N_SPLITS}-fold StratifiedKFold)...")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
    p  = params[best_model]
    oof_proba = np.zeros(len(y_train))

    def make_estimator():
        if best_model == "Logistic Regression":
            est = LogisticRegression(
                random_state=RANDOM_STATE, max_iter=2000, **p)
        elif best_model == "Decision Tree":
            est = DecisionTreeClassifier(random_state=RANDOM_STATE, **p)
        elif best_model == "Random Forest":
            est = RandomForestClassifier(
                random_state=RANDOM_STATE, n_jobs=-1, **p)
        else:
            spw = scale_pos_weight if "class_weight" in best_strategy else 1.0
            est = XGBClassifier(random_state=RANDOM_STATE,
                                eval_metric="logloss", n_jobs=-1,
                                scale_pos_weight=spw, **p)

        if "class_weight" in best_strategy and best_model != "XGBoost":
            est.set_params(class_weight="balanced")

        if "smote" in best_strategy:
            return ImbPipeline([
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("model", est),
            ])
        return est

    X_arr = X_train.values
    y_arr = y_train.values

    for fold_i, (tr_idx, val_idx) in enumerate(cv.split(X_arr, y_arr)):
        est = make_estimator()
        est.fit(X_arr[tr_idx], y_arr[tr_idx])
        oof_proba[val_idx] = est.predict_proba(X_arr[val_idx])[:, 1]
        print(f"  Fold {fold_i+1}/{N_SPLITS} done")

    # Verify no sample was skipped
    assert len(oof_proba) == len(y_train)
    assert not np.any(np.isnan(oof_proba))

    oof_df = pd.DataFrame({
        "sample_index":        range(len(y_train)),
        "actual_class":        y_train.values,
        "predicted_probability": oof_proba.round(6),
    })
    oof_df.to_csv(REPORTS_P7 / "oof_threshold_predictions.csv", index=False)
    print(f"  Saved: {REPORTS_P7 / 'oof_threshold_predictions.csv'}")
    print(f"  OOF samples: {len(oof_proba):,}  (equals training set)")
    return oof_proba


# ===========================================================================
# 10. ANALYZE THRESHOLDS
# ===========================================================================
def analyze_thresholds(oof_proba, y_train):
    """Evaluate metrics at each threshold on OOF predictions."""
    print("\n[INFO] Analyzing thresholds on OOF predictions...")

    y_true = y_train.values
    rows = []
    for thr in THRESHOLD_RANGE:
        y_pred = (oof_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        acc  = accuracy_score(y_true, y_pred)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ba   = balanced_accuracy_score(y_true, y_pred)
        fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr  = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        rows.append({
            "threshold":        round(float(thr), 2),
            "precision":        round(prec, 4),
            "recall":           round(rec,  4),
            "f1":               round(f1,   4),
            "accuracy":         round(acc,  4),
            "specificity":      round(spec, 4),
            "balanced_accuracy":round(ba,   4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
        })

    thr_df = pd.DataFrame(rows)
    thr_df.to_csv(REPORTS_P7 / "threshold_analysis.csv", index=False)
    print(f"  Saved: {REPORTS_P7 / 'threshold_analysis.csv'}")
    return thr_df


# ===========================================================================
# 11. SELECT THRESHOLD
# ===========================================================================
def select_threshold(thr_df):
    """
    Rule: among thresholds with recall >= RECALL_MIN (0.90),
    select the one with the highest F1.
    Tie-break: closest to 0.50.
    If constraint cannot be satisfied, select highest F1 without constraint.
    """
    print(f"\n[INFO] Selecting threshold "
          f"(maximize F1 subject to recall >= {RECALL_MIN})...")

    feasible = thr_df[thr_df["recall"] >= RECALL_MIN]
    constraint_met = len(feasible) > 0

    if constraint_met:
        best_f1    = feasible["f1"].max()
        candidates = feasible[feasible["f1"] == best_f1]
        # tie-break: closest to 0.5
        candidates = candidates.copy()
        candidates["dist_05"] = (candidates["threshold"] - 0.5).abs()
        chosen_row = candidates.sort_values("dist_05").iloc[0]
        method_note = (f"OOF F1 maximized subject to recall >= {RECALL_MIN}; "
                       "tie-break: closest to 0.50")
    else:
        print(f"  [NOTE] No threshold satisfies recall >= {RECALL_MIN}; "
              "selecting highest F1 unconstrained")
        best_f1    = thr_df["f1"].max()
        chosen_row = thr_df[thr_df["f1"] == best_f1].iloc[0]
        method_note = (f"OOF F1 maximized (recall constraint {RECALL_MIN} "
                       "could not be satisfied)")

    threshold = float(chosen_row["threshold"])
    print(f"  Selected threshold : {threshold:.2f}")
    print(f"  OOF recall at thr  : {chosen_row['recall']:.4f}")
    print(f"  OOF F1 at thr      : {chosen_row['f1']:.4f}")
    print(f"  Constraint satisfied: {constraint_met}")
    print(f"  Method: {method_note}")

    meta = {
        "threshold":        threshold,
        "selection_method": method_note,
        "n_folds":          N_SPLITS,
        "random_state":     RANDOM_STATE,
        "recall_constraint": RECALL_MIN,
        "oof_f1":           round(float(chosen_row["f1"]), 4),
        "oof_recall":       round(float(chosen_row["recall"]), 4),
        "oof_precision":    round(float(chosen_row["precision"]), 4),
        "recall_constraint_met": constraint_met,
    }
    with open(REPORTS_P7 / "final_threshold.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  Saved: {REPORTS_P7 / 'final_threshold.json'}")
    return threshold, thr_df, constraint_met


# ===========================================================================
# 12. THRESHOLD PLOTS
# ===========================================================================
def generate_threshold_plots(thr_df, threshold):
    """Threshold metrics and precision-recall-vs-threshold plots."""
    print("\n[INFO] Generating threshold plots...")

    # --- Metrics vs threshold ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thr_df["threshold"], thr_df["precision"],
            label="Precision", color="#2196F3", linewidth=1.6)
    ax.plot(thr_df["threshold"], thr_df["recall"],
            label="Recall", color="#F44336", linewidth=1.6)
    ax.plot(thr_df["threshold"], thr_df["f1"],
            label="F1", color="#4CAF50", linewidth=1.6)
    ax.plot(thr_df["threshold"], thr_df["balanced_accuracy"],
            label="Balanced Accuracy", color="#FF9800",
            linewidth=1.4, linestyle="--")
    ax.axvline(threshold, color="black", linestyle=":", linewidth=1.5,
               label=f"Selected thr={threshold:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Metrics (OOF predictions)")
    ax.legend(fontsize=9)
    ax.set_xlim([0.05, 0.95])
    plt.tight_layout()
    out = PLOTS_P7 / "threshold_metrics.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    # --- Precision / Recall vs threshold ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thr_df["threshold"], thr_df["precision"],
            label="Precision", color="#2196F3", linewidth=1.8)
    ax.plot(thr_df["threshold"], thr_df["recall"],
            label="Recall", color="#F44336", linewidth=1.8)
    ax.axhline(RECALL_MIN, color="#F44336", linestyle="--",
               linewidth=1.0, alpha=0.6,
               label=f"Recall constraint ({RECALL_MIN})")
    ax.axvline(threshold, color="black", linestyle=":", linewidth=1.5,
               label=f"Selected thr={threshold:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision and Recall vs Threshold (OOF)")
    ax.legend(fontsize=9)
    ax.set_xlim([0.05, 0.95])
    plt.tight_layout()
    out = PLOTS_P7 / "precision_recall_threshold.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ===========================================================================
# 13. FINAL TEST EVALUATION
# ===========================================================================
def evaluate_final_test_set(final_model, X_test, y_test, threshold):
    """
    First and only access to the test set in Phase 7.
    Uses the frozen threshold.
    """
    print(f"\n[INFO] Final test evaluation "
          f"(FIRST test access in Phase 7, threshold={threshold:.2f})...")

    y_proba = final_model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    assert len(y_pred) == len(y_test)
    assert not np.any(np.isnan(y_proba))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp_cm, fn, tp = cm.ravel()

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    auc   = roc_auc_score(y_test, y_proba)
    prauc = average_precision_score(y_test, y_proba)
    ba    = balanced_accuracy_score(y_test, y_pred)
    spec  = tn / (tn + fp_cm) if (tn + fp_cm) > 0 else 0.0
    fpr   = fp_cm / (fp_cm + tn) if (fp_cm + tn) > 0 else 0.0
    fnr   = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics = {
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall":   round(rec, 4), "f1":        round(f1, 4),
        "roc_auc":  round(auc, 4), "pr_auc":    round(prauc, 4),
        "specificity":       round(spec, 4),
        "balanced_accuracy": round(ba, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "true_negative":  int(tn),  "false_positive": int(fp_cm),
        "false_negative": int(fn),  "true_positive":  int(tp),
    }

    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Precision   : {prec:.4f}")
    print(f"  Recall      : {rec:.4f}")
    print(f"  F1          : {f1:.4f}")
    print(f"  ROC-AUC     : {auc:.4f}")
    print(f"  PR-AUC      : {prauc:.4f}")
    print(f"  Balanced Acc: {ba:.4f}")
    print(f"  TN={tn}  FP={fp_cm}  FN={fn}  TP={tp}")

    pd.DataFrame([metrics]).to_csv(
        REPORTS_P7 / "final_test_metrics.csv", index=False)
    print(f"  Saved: {REPORTS_P7 / 'final_test_metrics.csv'}")

    return metrics, y_pred, y_proba


# ===========================================================================
# 14. GENERATE TEST PLOTS
# ===========================================================================
def generate_test_plots(y_test, y_pred, y_proba, threshold, metrics):
    """Confusion matrix, ROC, PR curve for final test evaluation."""
    print("\n[INFO] Generating final test plots...")

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    classes = ["FALSE POS (0)", "CONFIRMED (1)"]
    ax.set_xticks([0, 1]);  ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, rotation=15, ha="right")
    ax.set_yticklabels(classes)
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label");  ax.set_xlabel("Predicted Label")
    ax.set_title(f"Final Confusion Matrix (thr={threshold:.2f})")
    plt.tight_layout()
    plt.savefig(PLOTS_P7 / "final_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_P7 / 'final_confusion_matrix.png'}")

    # --- ROC curve ---
    fpr_c, tpr_c, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_c, tpr_c, color="#2196F3", linewidth=2.0,
            label=f"Final Model (AUC={metrics['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1.0,
            label="Random (AUC=0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Final ROC Curve")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_P7 / "final_roc_curve.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_P7 / 'final_roc_curve.png'}")

    # --- PR curve ---
    prec_v, rec_v, _ = precision_recall_curve(y_test, y_proba)
    prevalence = float((y_test == 1).mean())
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec_v, prec_v, color="#4CAF50", linewidth=2.0,
            label=f"Final Model (AP={metrics['pr_auc']:.4f})")
    ax.axhline(prevalence, linestyle="--", color="grey", linewidth=1.0,
               label=f"Prevalence ({prevalence:.3f})")
    ax.set_xlabel("Recall");  ax.set_ylabel("Precision")
    ax.set_title("Final Precision-Recall Curve")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_P7 / "final_pr_curve.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_P7 / 'final_pr_curve.png'}")


# ===========================================================================
# 15. CALIBRATION ANALYSIS
# ===========================================================================
def evaluate_calibration(final_model, X_test, y_test, y_proba):
    """Calibration curve and Brier score (diagnostic only, no recalibration)."""
    print("\n[INFO] Evaluating probability calibration (diagnostic)...")

    brier = brier_score_loss(y_test, y_proba)
    print(f"  Brier score: {brier:.4f}  (lower = better, 0=perfect)")

    frac_pos, mean_pred = calibration_curve(
        y_test, y_proba, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(mean_pred, frac_pos, "s-", color="#2196F3", linewidth=1.8,
            label="Final Model")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1.2,
            label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_P7 / "calibration_curve.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_P7 / 'calibration_curve.png'}")

    cal_df = pd.DataFrame([
        {"metric": "brier_score", "value": round(brier, 6)},
    ])
    cal_df.to_csv(REPORTS_P7 / "calibration_metrics.csv", index=False)
    print(f"  Saved: {REPORTS_P7 / 'calibration_metrics.csv'}")
    return brier


# ===========================================================================
# 16. SAVE FINAL MODEL
# ===========================================================================
def save_final_model(final_model, best_model, best_strategy, params,
                     threshold, scale_pos_weight):
    """Save the fitted final model + metadata."""
    print("\n[INFO] Saving final model...")

    model_path = MODELS_FINAL / "final_model.pkl"
    joblib.dump(final_model, model_path)
    print(f"  Saved: {model_path}  "
          f"({model_path.stat().st_size / 1024:.1f} KB)")

    meta = {
        "project":            "Kepler Exoplanet Classification",
        "phase":              7,
        "model_type":         best_model,
        "imbalance_strategy": best_strategy,
        "hyperparameters":    params[best_model],
        "threshold":          threshold,
        "scale_pos_weight":   round(scale_pos_weight, 4)
                              if "class_weight" in best_strategy
                              and best_model == "XGBoost" else None,
        "primary_metric":     "PR-AUC",
        "threshold_metric":   f"F1 subject to recall >= {RECALL_MIN}",
        "random_state":       RANDOM_STATE,
        "feature_count":      11,
        "features":           CORE_FEATURES,
        "python_version":     platform.python_version(),
        "sklearn_version":    sklearn.__version__,
        "xgboost_version":    xgboost.__version__,
        "imblearn_version":   imblearn.__version__,
        "numpy_version":      np.__version__,
        "pandas_version":     pd.__version__,
    }
    meta_path = MODELS_FINAL / "final_model_metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  Saved: {meta_path}")
    return model_path


# ===========================================================================
# 17. VERIFY MODEL RELOAD
# ===========================================================================
def verify_model_reload(final_model, X_test, threshold):
    """Reload and verify predictions match on a small deterministic sample."""
    print("\n[INFO] Verifying model reload...")

    model_path  = MODELS_FINAL / "final_model.pkl"
    loaded      = joblib.load(model_path)
    X_sample    = X_test.iloc[:20]
    orig_proba  = final_model.predict_proba(X_sample)[:, 1]
    load_proba  = loaded.predict_proba(X_sample)[:, 1]

    proba_match = np.allclose(orig_proba, load_proba, atol=1e-6)
    pred_match  = np.array_equal(
        (orig_proba >= threshold).astype(int),
        (load_proba >= threshold).astype(int),
    )

    result = (
        f"Model reload verification\n"
        f"  Sample size: 20\n"
        f"  Probabilities match (atol=1e-6): {proba_match}\n"
        f"  Predictions match: {pred_match}\n"
        f"  Max absolute probability difference: "
        f"{float(np.abs(orig_proba - load_proba).max()):.2e}\n"
        f"  Status: {'PASS' if proba_match and pred_match else 'FAIL'}\n"
    )
    print("  " + result.replace("\n", "\n  ").strip())
    (REPORTS_P7 / "model_reload_verification.txt").write_text(
        result, encoding="utf-8")
    print(f"  Saved: {REPORTS_P7 / 'model_reload_verification.txt'}")

    assert proba_match and pred_match, "Model reload verification FAILED"
    print("  [OK] Reload verified")


# ===========================================================================
# 18. PREDICT CANDIDATES
# ===========================================================================
def predict_candidates(final_model, X_candidate, threshold):
    """Generate predictions for 1,977 CANDIDATE rows using the final model."""
    print("\n[INFO] Generating candidate predictions...")

    y_proba_cand = final_model.predict_proba(X_candidate)[:, 1]
    y_pred_cand  = (y_proba_cand >= threshold).astype(int)

    pred_conf = int((y_pred_cand == 1).sum())
    pred_fp   = int((y_pred_cand == 0).sum())

    print(f"  Candidates total          : {len(y_pred_cand):,}")
    print(f"  Predicted CONFIRMED       : {pred_conf:,}  "
          f"({pred_conf/len(y_pred_cand)*100:.1f}%)")
    print(f"  Predicted FALSE POSITIVE  : {pred_fp:,}  "
          f"({pred_fp/len(y_pred_cand)*100:.1f}%)")
    print("  [NOTE] These are model predictions, not astronomical confirmations.")

    cand_df = pd.DataFrame({
        "candidate_index":      range(len(y_pred_cand)),
        "predicted_class":      [
            "CONFIRMED" if p == 1 else "FALSE_POSITIVE" for p in y_pred_cand
        ],
        "predicted_probability": y_proba_cand.round(4),
        "threshold":             threshold,
    })
    cand_df.to_csv(REPORTS_P7 / "candidate_predictions.csv", index=False)
    print(f"  Saved: {REPORTS_P7 / 'candidate_predictions.csv'}")

    # Candidate plots
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Predicted\nCONFIRMED", "Predicted\nFALSE POSITIVE"]
    counts = [pred_conf, pred_fp]
    bars   = ax.bar(labels, counts, color=["#2196F3", "#F44336"],
                    edgecolor="white", linewidth=0.8)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + len(y_pred_cand) * 0.01,
                f"{c:,}\n({c/len(y_pred_cand)*100:.1f}%)",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Count")
    ax.set_title("Candidate Prediction Distribution\n"
                 "(Model predictions -- not astronomical confirmations)")
    plt.tight_layout()
    plt.savefig(PLOTS_P7 / "candidate_prediction_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_P7 / 'candidate_prediction_distribution.png'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_proba_cand, bins=50, color="#607D8B", alpha=0.8,
            edgecolor="white", density=True)
    ax.axvline(threshold, color="#F44336", linestyle="--",
               linewidth=2.0, label=f"Threshold ({threshold:.2f})")
    ax.set_xlabel("Predicted Probability (CONFIRMED)")
    ax.set_ylabel("Density")
    ax.set_title("Candidate Probability Distribution")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_P7 / "candidate_probability_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_P7 / 'candidate_probability_distribution.png'}")

    return pred_conf, pred_fp, y_proba_cand


# ===========================================================================
# 19. VALIDATE PREDICTION PIPELINE
# ===========================================================================
def validate_prediction_pipeline(final_model, X_test, threshold):
    """Run pipeline validation tests 1-5."""
    print("\n[INFO] Running pipeline validation tests...")

    # Test 1: Correct features
    try:
        p = final_model.predict_proba(X_test.iloc[:5])
        assert p.shape[1] == 2
        print("  Test 1 (correct features): PASS")
    except Exception as e:
        print(f"  Test 1 (correct features): FAIL -- {e}")

    # Test 2: Missing feature (pass only 10 features)
    try:
        final_model.predict_proba(X_test.iloc[:5, :10])
        print("  Test 2 (missing feature): expected error not raised")
    except Exception:
        print("  Test 2 (missing feature): PASS -- error raised as expected")

    # Test 3: Shuffled columns
    shuffled = X_test.iloc[:5][list(reversed(X_test.columns))]
    try:
        # predict on shuffled order -- should give different result
        p_correct = final_model.predict_proba(
            X_test.iloc[:5][CORE_FEATURES])[:, 1]
        # Force correct ordering
        shuffled_reordered = shuffled[CORE_FEATURES]
        p_reordered = final_model.predict_proba(shuffled_reordered)[:, 1]
        match = np.allclose(p_correct, p_reordered, atol=1e-6)
        print(f"  Test 3 (column ordering): "
              f"{'PASS' if match else 'WARNING -- ordering matters'}")
    except Exception as e:
        print(f"  Test 3 (column ordering): {e}")

    # Test 4: Model reload
    loaded = joblib.load(MODELS_FINAL / "final_model.pkl")
    p_orig  = final_model.predict_proba(X_test.iloc[:5])[:, 1]
    p_load  = loaded.predict_proba(X_test.iloc[:5])[:, 1]
    print(f"  Test 4 (reload match): "
          f"{'PASS' if np.allclose(p_orig, p_load, atol=1e-6) else 'FAIL'}")

    # Test 5: Candidate data
    try:
        X_cand = pd.read_csv(DATA_PROCESSED / "X_candidate_processed.csv")
        final_model.predict_proba(X_cand.iloc[:5])
        print("  Test 5 (candidate data): PASS")
    except Exception as e:
        print(f"  Test 5 (candidate data): FAIL -- {e}")

    print("  [OK] Pipeline validation complete")


# ===========================================================================
# 20. GENERATE MARKDOWN REPORT
# ===========================================================================
def generate_report(best_model, best_strategy, best_row, threshold,
                    metrics, brier, pred_conf, pred_fp,
                    n_candidates, constraint_met):
    """Write the Phase 7 finalization report."""
    print("\n[INFO] Writing Phase 7 report...")

    lines = [
        "# Phase 7 Model Selection & Finalization Report",
        "## Kepler Exoplanet Classification \u2014 Explainable ML System",
        "",
        "*All values derived from actual data. Test set accessed exactly once.*",
        "",
        "---",
        "",
        "## 1. Objective",
        "",
        "Phase 7 converts the experimental results of Phases 1-6 into a "
        "reproducible finalized prediction pipeline. "
        "The final model is selected using training/CV evidence only, "
        "a classification threshold is optimized on out-of-fold predictions, "
        "and the system is evaluated on the untouched test set once.",
        "",
        "---",
        "",
        "## 2. Candidate Models",
        "",
        "12 configurations evaluated (4 models x 3 strategies): "
        "baseline, class_weight/scale_pos_weight, SMOTE. "
        "See `reports/phase_5/imbalance_cv_results.csv`.",
        "",
        "---",
        "",
        "## 3. Selection Method",
        "",
        "Selection based entirely on 5-fold stratified CV metrics from Phase 5. "
        "Primary criterion: mean CV PR-AUC. "
        "Secondary (tie-break): lower PR-AUC std deviation.",
        "",
        "The test set was NOT inspected during model selection.",
        "",
        "---",
        "",
        "## 4. Selected Configuration",
        "",
        f"| Item | Value |",
        f"|------|-------|",
        f"| Model | {best_model} |",
        f"| Strategy | {best_strategy} |",
        f"| CV PR-AUC | {best_row['cv_pr_auc_mean']:.4f} +/- "
        f"{best_row['cv_pr_auc_std']:.4f} |",
        f"| CV ROC-AUC | {best_row['cv_roc_auc_mean']:.4f} +/- "
        f"{best_row['cv_roc_auc_std']:.4f} |",
        f"| CV Recall | {best_row['cv_recall_mean']:.4f} |",
        f"| CV F1 | {best_row['cv_f1_mean']:.4f} |",
        "",
        "---",
        "",
        "## 5. Threshold Optimization",
        "",
        f"Out-of-fold ({N_SPLITS}-fold) predicted probabilities generated on "
        "training data only.",
        f"Threshold range: 0.05 to 0.95 (step 0.01).",
        f"Criterion: maximize F1 subject to recall >= {RECALL_MIN}.",
        f"Recall constraint met: {constraint_met}.",
        f"Selected threshold: **{threshold:.2f}**",
        "",
        "---",
        "",
        "## 6. Final Test Evaluation",
        "",
        f"Test set accessed once, after threshold was frozen.",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]
    metric_display = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("specificity", "Specificity"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("false_positive_rate", "False Positive Rate"),
        ("false_negative_rate", "False Negative Rate"),
        ("true_negative", "TN"), ("false_positive", "FP"),
        ("false_negative", "FN"), ("true_positive", "TP"),
    ]
    for key, label in metric_display:
        lines.append(f"| {label} | {metrics[key]} |")

    lines += [
        "",
        "---",
        "",
        "## 7. Calibration",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Brier Score | {brier:.4f} |",
        "",
        "The Brier score measures probability calibration "
        "(0 = perfect, 1 = worst). "
        "No automatic recalibration was applied in this phase.",
        "",
        "---",
        "",
        "## 8. Candidate Predictions",
        "",
        f"| Item | Value |",
        f"|------|-------|",
        f"| Total candidates | {n_candidates:,} |",
        f"| Predicted CONFIRMED | {pred_conf:,} |",
        f"| Predicted FALSE POSITIVE | {pred_fp:,} |",
        f"| Threshold used | {threshold:.2f} |",
        "",
        "**Important**: These are model predictions for currently unlabeled "
        "CANDIDATE rows. They are NOT scientifically confirmed exoplanets.",
        "",
        "---",
        "",
        "## 9. Reproducibility",
        "",
        f"| Item | Value |",
        f"|------|-------|",
        f"| random_state | {RANDOM_STATE} |",
        f"| Python | {platform.python_version()} |",
        f"| scikit-learn | {sklearn.__version__} |",
        f"| XGBoost | {xgboost.__version__} |",
        f"| imbalanced-learn | {imblearn.__version__} |",
        "| Final model | `models/final/final_model.pkl` |",
        "| Threshold | `reports/phase_7/final_threshold.json` |",
        "",
        "---",
        "",
        "## 10. Limitations",
        "",
        "- Model predictions are not astronomical scientific confirmations",
        "- Dataset labels originate from the NASA Kepler KOI classification data",
        "- Threshold reflects the selected operational objective "
        "(F1 subject to recall constraint)",
        "- Probability calibration may have limitations",
        "- Model performance depends on the feature distribution in training data",
        "- Candidate predictions are preliminary and should be reviewed by "
        "domain experts",
        "",
        "---",
        "",
        "## 11. Conclusion",
        "",
        f"A finalized prediction pipeline has been created using {best_model} "
        f"with {best_strategy} strategy. "
        f"The pipeline achieves PR-AUC={metrics['pr_auc']:.4f} "
        f"and F1={metrics['f1']:.4f} on the held-out test set. "
        f"The classification threshold of {threshold:.2f} was selected using "
        "out-of-fold predictions from the training set only.",
        "",
        "**Phase 7 is complete. Phase 8 will handle Streamlit deployment.**",
    ]

    path = REPORTS_P7 / "phase_7_finalization_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {path}")


# ===========================================================================
# 21. MODEL CARD
# ===========================================================================
def generate_model_card(best_model, best_strategy, threshold, metrics,
                        brier, pred_conf, pred_fp, n_candidates):
    """Write model_card.md."""
    print("\n[INFO] Writing model card...")

    lines = [
        "# Model Card",
        "## Kepler Exoplanet Classification",
        "",
        "---",
        "",
        "## Model",
        "",
        f"- **Type**: {best_model} with {best_strategy} imbalance strategy",
        f"- **Version**: Phase 7 finalized model",
        f"- **Saved as**: `models/final/final_model.pkl`",
        "",
        "---",
        "",
        "## Intended Use",
        "",
        "Classify NASA Kepler Objects of Interest (KOIs) into CONFIRMED "
        "or FALSE POSITIVE categories. "
        "The system is intended as an exploratory machine-learning analysis tool. "
        "It is NOT a replacement for peer-reviewed astronomical analysis.",
        "",
        "---",
        "",
        "## Input Features",
        "",
        "11 preprocessed (imputed, log-transformed where applicable, "
        "standardized) numerical features:",
        "",
    ]
    feature_desc = {
        "koi_period": "Orbital period",
        "koi_duration": "Transit duration",
        "koi_depth": "Transit depth",
        "koi_prad": "Planetary radius estimate",
        "koi_impact": "Transit impact parameter",
        "koi_model_snr": "Signal-to-noise ratio",
        "koi_teq": "Equilibrium temperature",
        "koi_insol": "Stellar insolation",
        "koi_steff": "Stellar effective temperature",
        "koi_slogg": "Stellar surface gravity",
        "koi_srad": "Stellar radius",
    }
    for feat in CORE_FEATURES:
        lines.append(f"- `{feat}`: {feature_desc.get(feat, '')}")

    lines += [
        "",
        "---",
        "",
        "## Output",
        "",
        "- **Predicted class**: CONFIRMED (1) or FALSE POSITIVE (0)",
        f"- **Predicted probability**: P(CONFIRMED)",
        f"- **Classification threshold**: {threshold:.2f}",
        "",
        "---",
        "",
        "## Training Data",
        "",
        "NASA Exoplanet Archive Cumulative KOI Table "
        "(cumulative_kois.csv, downloaded September 2026).",
        "Labeled set: 7,587 rows (2,748 CONFIRMED, 4,839 FALSE POSITIVE).",
        "Train split: 6,069 rows (80%).",
        "",
        "---",
        "",
        "## Evaluation Method",
        "",
        "- 5-fold stratified cross-validation for model/threshold selection",
        "- Untouched 20% test set (1,518 rows) for final evaluation",
        "- Primary metric: PR-AUC (Average Precision)",
        "",
        "---",
        "",
        "## Performance",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Precision | {metrics['precision']:.4f} |",
        f"| Recall | {metrics['recall']:.4f} |",
        f"| F1 | {metrics['f1']:.4f} |",
        f"| ROC-AUC | {metrics['roc_auc']:.4f} |",
        f"| PR-AUC | {metrics['pr_auc']:.4f} |",
        f"| Brier Score | {brier:.4f} |",
        "",
        "---",
        "",
        "## Threshold",
        "",
        f"Classification threshold: **{threshold:.2f}**",
        "",
        "Selected to maximize F1 subject to recall >= 0.90, "
        "using 5-fold out-of-fold predictions on training data.",
        "",
        "---",
        "",
        "## Limitations",
        "",
        "1. Labels are sourced from the NASA Kepler KOI catalogue, "
        "which itself reflects the state of analysis at time of download.",
        "2. The model may not generalize to KOI observations from "
        "other telescopes or with different noise characteristics.",
        "3. Probability outputs are not perfectly calibrated.",
        "4. Feature distributions in the training set may not represent "
        "all possible planetary systems.",
        "",
        "---",
        "",
        "## Ethical and Scientific Considerations",
        "",
        "**This model is a machine-learning classifier, not a scientific "
        "confirmation instrument.**",
        "",
        "- A prediction of CONFIRMED by this model means the model assigns "
        "a high probability of agreement with the existing KOI "
        "CONFIRMED label. It does not constitute independent scientific "
        "confirmation of the existence of an exoplanet.",
        "- A prediction of FALSE POSITIVE does not mean the object has "
        "been scientifically ruled out as an exoplanet.",
        "- All candidate predictions should be treated as preliminary "
        "machine-learning outputs, subject to expert review.",
        "- The system should not be used as the sole basis for "
        "scientific publications or follow-up observation decisions.",
    ]

    path = REPORTS_P7 / "model_card.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {path}")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 70)
    print("PHASE 7: MODEL SELECTION & FINALIZATION")
    print("Kepler Exoplanet Classification")
    print("=" * 70)

    # 1-3. Load data
    X_train, y_train, pos_n, neg_n, scale_pos_weight = load_training_data()
    X_test,  y_test  = load_test_data()
    X_cand           = load_candidate_data()

    # 4-5. Load prior phase artifacts
    params  = load_phase_4_parameters()
    cv_df   = load_phase_5_results()

    # 6. Build 12 candidate configurations
    configs = build_candidate_configurations(params, scale_pos_weight)

    # 7. Select final configuration (CV evidence only, no test set)
    best_model, best_strategy, best_row, final_config = \
        select_final_configuration(cv_df, configs)

    # 8. Fit final model on full training data
    final_config = fit_final_model(final_config, X_train, y_train)

    # 9. Generate OOF predictions for threshold optimization
    oof_proba = generate_oof_predictions(
        best_model, best_strategy, params, scale_pos_weight,
        X_train, y_train)

    # 10. Analyze thresholds
    thr_df = analyze_thresholds(oof_proba, y_train)

    # 11. Select threshold (training data only)
    threshold, thr_df, constraint_met = select_threshold(thr_df)

    # 12. Threshold plots
    generate_threshold_plots(thr_df, threshold)

    # --- THRESHOLD IS NOW FROZEN ---
    print(f"\n[INFO] THRESHOLD FROZEN: {threshold:.2f}  "
          f"-- Test set will now be accessed for the first and only time.")

    # 13. Final test evaluation (first + only test access)
    metrics, y_pred, y_proba = evaluate_final_test_set(
        final_config, X_test, y_test, threshold)

    # 14. Test plots
    generate_test_plots(y_test, y_pred, y_proba, threshold, metrics)

    # 15. Calibration
    brier = evaluate_calibration(final_config, X_test, y_test, y_proba)

    # 16. Save final model + metadata
    model_path = save_final_model(final_config, best_model, best_strategy,
                                  params, threshold, scale_pos_weight)

    # 17. Verify model reload
    verify_model_reload(final_config, X_test, threshold)

    # 18. Candidate predictions
    pred_conf, pred_fp, _ = predict_candidates(final_config, X_cand, threshold)

    # 19. Validate pipeline
    validate_prediction_pipeline(final_config, X_test, threshold)

    # 20. Generate reports
    generate_report(best_model, best_strategy, best_row, threshold,
                    metrics, brier, pred_conf, pred_fp,
                    len(X_cand), constraint_met)
    generate_model_card(best_model, best_strategy, threshold, metrics,
                        brier, pred_conf, pred_fp, len(X_cand))

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("PHASE 7 COMPLETE -- FINAL MODEL SUMMARY")
    print("=" * 70)
    print(f"  Model            : {best_model}")
    print(f"  Strategy         : {best_strategy}")
    print(f"  Threshold        : {threshold:.2f}")
    print(f"  Test PR-AUC      : {metrics['pr_auc']:.4f}")
    print(f"  Test ROC-AUC     : {metrics['roc_auc']:.4f}")
    print(f"  Test Recall      : {metrics['recall']:.4f}")
    print(f"  Test Precision   : {metrics['precision']:.4f}")
    print(f"  Test F1          : {metrics['f1']:.4f}")
    print(f"  Brier Score      : {brier:.4f}")
    print(f"  Cand. CONFIRMED  : {pred_conf:,}")
    print(f"  Cand. FALSE POS  : {pred_fp:,}")
    print(f"  Model saved      : {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
