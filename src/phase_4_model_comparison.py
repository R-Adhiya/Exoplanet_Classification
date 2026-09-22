"""
Phase 4: Model Comparison & Robust Evaluation
Kepler Exoplanet Classification -- Explainable ML System

Performs 5-fold stratified cross-validation and hyperparameter tuning for:
  1. Logistic Regression (GridSearchCV)
  2. Decision Tree (GridSearchCV)
  3. Random Forest (RandomizedSearchCV, n_iter=20)
  4. XGBoost (RandomizedSearchCV, n_iter=20)

Primary tuning metric: average_precision (PR-AUC).
Test set is ONLY used for final evaluation after tuning is complete.
NO SMOTE, NO class weighting, NO threshold optimization, NO SHAP.

Compatibility: Python 3.12, pandas 2.x, scikit-learn 1.9+, XGBoost 3.x,
               Windows PowerShell (ASCII-safe terminal output).
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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Stdout: force UTF-8 on Windows so report writes never fail
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
MODELS_TUNED    = ROOT / "models" / "tuned"
REPORTS_P3      = ROOT / "reports" / "phase_3"
REPORTS_P4      = ROOT / "reports" / "phase_4"
PLOTS_P4        = ROOT / "plots" / "phase_4"

for _d in [MODELS_TUNED, REPORTS_P4, PLOTS_P4]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE   = 42
N_SPLITS       = 5
N_ITER_RF      = 20
N_ITER_XGB     = 20
PRIMARY_METRIC = "average_precision"
np.random.seed(RANDOM_STATE)

MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
]

MODEL_SLUGS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree":       "decision_tree",
    "Random Forest":       "random_forest",
    "XGBoost":             "xgboost",
}

PLOT_COLORS = {
    "Logistic Regression": "#2196F3",
    "Decision Tree":       "#FF9800",
    "Random Forest":       "#4CAF50",
    "XGBoost":             "#E91E63",
}

CV_SCORING = {
    "average_precision": "average_precision",
    "roc_auc":           "roc_auc",
    "f1":                "f1",
    "precision":         "precision",
    "recall":            "recall",
    "accuracy":          "accuracy",
}


# ===========================================================================
# 1. LOAD PHASE 2 DATA
# ===========================================================================
def load_phase_2_data():
    """
    Load the Phase 2 preprocessed train/test splits.
    Returns (X_train, X_test, y_train, y_test, feature_names).
    """
    print("\n[INFO] Loading Phase 2 processed data...")

    files = {
        "X_train":  DATA_PROCESSED / "X_train_processed.csv",
        "X_test":   DATA_PROCESSED / "X_test_processed.csv",
        "y_train":  DATA_PROCESSED / "y_train.csv",
        "y_test":   DATA_PROCESSED / "y_test.csv",
        "features": DATA_PROCESSED / "processed_feature_names.json",
    }
    for name, path in files.items():
        if not path.exists():
            print(f"\n[CRITICAL] Required file not found: {path}")
            print("  Run Phase 2 first.")
            sys.exit(1)

    X_train = pd.read_csv(files["X_train"])
    X_test  = pd.read_csv(files["X_test"])
    y_train = pd.read_csv(files["y_train"]).squeeze()
    y_test  = pd.read_csv(files["y_test"]).squeeze()

    with open(files["features"]) as fh:
        feature_names = json.load(fh)

    # Validate
    assert X_train.shape[1] == 11, f"X_train: {X_train.shape[1]} features (expected 11)"
    assert X_test.shape[1]  == 11, f"X_test: {X_test.shape[1]} features (expected 11)"
    assert X_train.isnull().sum().sum() == 0, "NaN in X_train"
    assert X_test.isnull().sum().sum()  == 0, "NaN in X_test"

    print(f"  X_train : {X_train.shape}  "
          f"(CONF={int((y_train==1).sum()):,}  FP={int((y_train==0).sum()):,})")
    print(f"  X_test  : {X_test.shape}  "
          f"(CONF={int((y_test==1).sum()):,}  FP={int((y_test==0).sum()):,})")
    print(f"  Features: {feature_names}")
    print("  [OK] Phase 2 data loaded")

    return X_train, X_test, y_train, y_test, feature_names


# ===========================================================================
# 2. LOAD PHASE 3 BASELINE RESULTS
# ===========================================================================
def load_phase_3_baseline_results():
    """
    Load Phase 3 baseline comparison CSV.
    Returns a DataFrame indexed by model name.
    """
    print("\n[INFO] Loading Phase 3 baseline results...")
    path = REPORTS_P3 / "baseline_model_comparison.csv"
    if not path.exists():
        print(f"  [WARNING] Phase 3 baseline file not found: {path}")
        print("  Baseline comparison will be skipped.")
        return None

    df = pd.read_csv(path)
    df = df.set_index("model")
    print(f"  Loaded baseline for: {df.index.tolist()}")
    return df


# ===========================================================================
# 3. CREATE CV STRATEGY
# ===========================================================================
def create_cv_strategy():
    """Return the shared StratifiedKFold object used across all searches."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    print(f"\n[INFO] CV strategy: StratifiedKFold(n_splits={N_SPLITS}, "
          f"shuffle=True, random_state={RANDOM_STATE})")
    return cv


# ===========================================================================
# 4. DEFINE SEARCH SPACES
# ===========================================================================
def define_search_spaces():
    """
    Return a dict of {model_name: (estimator, param_grid, search_type)}.
    search_type is "grid" or "random".
    """
    spaces = {
        "Logistic Regression": (
            LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
            {
                "C":       [0.01, 0.1, 1, 10, 100],
                "solver":  ["lbfgs"],
                "penalty": ["l2"],
            },
            "grid",
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "max_depth":        [3, 5, 8, 12, None],
                "min_samples_split":[2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 5, 10],
                "criterion":        ["gini", "entropy"],
            },
            "grid",
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators":     [200, 400],
                "max_depth":        [None, 10, 20],
                "min_samples_split":[2, 5],
                "min_samples_leaf": [1, 2],
                "max_features":     ["sqrt", "log2"],
            },
            "random",
        ),
        "XGBoost": (
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss",
                          n_jobs=-1),
            {
                "n_estimators":     [100, 200, 300, 500],
                "max_depth":        [3, 4, 5, 6, 8],
                "learning_rate":    [0.01, 0.03, 0.05, 0.1, 0.2],
                "subsample":        [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "min_child_weight": [1, 3, 5],
            },
            "random",
        ),
    }
    return spaces


# ===========================================================================
# 5-8. INDIVIDUAL MODEL TUNING FUNCTIONS
# ===========================================================================
def _run_search(name, estimator, param_grid, search_type, X_train, y_train, cv):
    """
    Run GridSearchCV or RandomizedSearchCV.
    Returns the fitted search object.
    """
    common_kw = dict(
        estimator=estimator,
        scoring=PRIMARY_METRIC,
        cv=cv,
        refit=True,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
    )
    if search_type == "grid":
        n_combos = 1
        for v in param_grid.values():
            n_combos *= len(v)
        print(f"  GridSearchCV: {n_combos} combinations x {N_SPLITS} folds "
              f"= {n_combos * N_SPLITS} fits")
        search = GridSearchCV(param_grid=param_grid, **common_kw)
    else:
        n_iter = N_ITER_RF if name == "Random Forest" else N_ITER_XGB
        print(f"  RandomizedSearchCV: n_iter={n_iter} x {N_SPLITS} folds "
              f"= {n_iter * N_SPLITS} fits")
        search = RandomizedSearchCV(
            param_distributions=param_grid,
            n_iter=n_iter,
            random_state=RANDOM_STATE,
            **common_kw,
        )
    search.fit(X_train, y_train)
    return search


def tune_logistic_regression(spaces, X_train, y_train, cv):
    name = "Logistic Regression"
    print(f"\n[INFO] Tuning {name}...")
    est, grid, stype = spaces[name]
    return name, _run_search(name, est, grid, stype, X_train, y_train, cv)


def tune_decision_tree(spaces, X_train, y_train, cv):
    name = "Decision Tree"
    print(f"\n[INFO] Tuning {name}...")
    est, grid, stype = spaces[name]
    return name, _run_search(name, est, grid, stype, X_train, y_train, cv)


def tune_random_forest(spaces, X_train, y_train, cv):
    name = "Random Forest"
    print(f"\n[INFO] Tuning {name}...")
    est, grid, stype = spaces[name]
    return name, _run_search(name, est, grid, stype, X_train, y_train, cv)


def tune_xgboost(spaces, X_train, y_train, cv):
    name = "XGBoost"
    print(f"\n[INFO] Tuning {name}...")
    est, grid, stype = spaces[name]
    return name, _run_search(name, est, grid, stype, X_train, y_train, cv)


# ===========================================================================
# 9. COLLECT CV RESULTS
# ===========================================================================
def collect_cv_results(search_results, X_train, y_train, cv):
    """
    Extract CV statistics from each search object.
    Also runs a multi-metric cross_validate on the best estimator to get
    std values for all metrics.
    Returns a list of result dicts.
    """
    print("\n[INFO] Collecting cross-validation results...")

    cv_rows  = []
    best_params_all = {}

    for name, search in search_results.items():
        best_params = search.best_params_
        best_score  = search.best_score_      # mean CV PR-AUC from search

        # Run multi-metric CV on best estimator to get mean/std for all metrics
        best_est = search.best_estimator_
        cv_scores = cross_validate(
            best_est, X_train, y_train,
            cv=cv,
            scoring=CV_SCORING,
            n_jobs=-1,
        )

        pr_mean  = float(np.mean(cv_scores["test_average_precision"]))
        pr_std   = float(np.std(cv_scores["test_average_precision"]))
        pr_min   = float(np.min(cv_scores["test_average_precision"]))
        pr_max   = float(np.max(cv_scores["test_average_precision"]))

        row = {
            "model":              name,
            "best_cv_pr_auc":     round(best_score, 4),
            "cv_pr_auc_mean":     round(pr_mean, 4),
            "cv_pr_auc_std":      round(pr_std, 4),
            "cv_pr_auc_min":      round(pr_min, 4),
            "cv_pr_auc_max":      round(pr_max, 4),
            "cv_roc_auc_mean":    round(float(np.mean(cv_scores["test_roc_auc"])), 4),
            "cv_roc_auc_std":     round(float(np.std(cv_scores["test_roc_auc"])), 4),
            "cv_f1_mean":         round(float(np.mean(cv_scores["test_f1"])), 4),
            "cv_f1_std":          round(float(np.std(cv_scores["test_f1"])), 4),
            "cv_precision_mean":  round(float(np.mean(cv_scores["test_precision"])), 4),
            "cv_precision_std":   round(float(np.std(cv_scores["test_precision"])), 4),
            "cv_recall_mean":     round(float(np.mean(cv_scores["test_recall"])), 4),
            "cv_recall_std":      round(float(np.std(cv_scores["test_recall"])), 4),
            "cv_accuracy_mean":   round(float(np.mean(cv_scores["test_accuracy"])), 4),
        }
        cv_rows.append(row)
        best_params_all[name] = best_params

        print(f"  {name:<22}  PR-AUC={pr_mean:.4f}+/-{pr_std:.4f}  "
              f"ROC-AUC={row['cv_roc_auc_mean']:.4f}  "
              f"F1={row['cv_f1_mean']:.4f}")
        print(f"    Best params: {best_params}")

    # Save CV summary
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(REPORTS_P4 / "cv_results_summary.csv", index=False)
    print(f"\n  Saved: {REPORTS_P4 / 'cv_results_summary.csv'}")

    # Save best parameters
    with open(REPORTS_P4 / "best_parameters.json", "w") as fh:
        json.dump(best_params_all, fh, indent=2)
    print(f"  Saved: {REPORTS_P4 / 'best_parameters.json'}")

    return cv_rows, best_params_all


# ===========================================================================
# 10. EVALUATE TUNED MODELS ON TEST SET
# ===========================================================================
def evaluate_tuned_models_on_test(search_results, X_test, y_test):
    """
    Refit each best estimator on full X_train is already done by refit=True.
    Predict on X_test (first and only access to test set in Phase 4).
    Returns a list of result dicts.
    """
    print("\n[INFO] Evaluating tuned models on test set "
          "(FIRST access to test set in Phase 4)...")

    results = []
    print(f"  {'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} "
          f"{'F1':>6} {'ROC':>6} {'PR':>6}")
    print("  " + "-" * 62)

    for name in MODEL_NAMES:
        search  = search_results[name]
        y_pred  = search.predict(X_test)
        y_proba = search.predict_proba(X_test)[:, 1]

        # Validation
        assert len(y_pred) == len(y_test), f"{name}: prediction count mismatch"
        assert not np.any(np.isnan(y_proba)), f"{name}: NaN in proba"
        assert float(y_proba.min()) >= 0.0
        assert float(y_proba.max()) <= 1.0

        cm = confusion_matrix(y_test, y_pred)
        tn, fp_cm, fn, tp = cm.ravel()

        acc   = accuracy_score(y_test, y_pred)
        prec  = precision_score(y_test, y_pred, zero_division=0)
        rec   = recall_score(y_test, y_pred, zero_division=0)
        f1    = f1_score(y_test, y_pred, zero_division=0)
        auc   = roc_auc_score(y_test, y_proba)
        prauc = average_precision_score(y_test, y_proba)

        res = {
            "model":          name,
            "accuracy":       round(acc, 4),
            "precision":      round(prec, 4),
            "recall":         round(rec, 4),
            "f1":             round(f1, 4),
            "roc_auc":        round(auc, 4),
            "pr_auc":         round(prauc, 4),
            "true_negative":  int(tn),
            "false_positive": int(fp_cm),
            "false_negative": int(fn),
            "true_positive":  int(tp),
            "y_pred":         y_pred,
            "y_proba":        y_proba,
        }
        results.append(res)
        print(f"  {name:<22} {acc:>6.4f} {prec:>6.4f} {rec:>6.4f} "
              f"{f1:>6.4f} {auc:>6.4f} {prauc:>6.4f}")

    print("  [OK] All tuned models evaluated on test set")
    return results


# ===========================================================================
# 11. CONFUSION MATRICES
# ===========================================================================
def generate_confusion_matrices(results, y_test):
    """Save confusion matrix PNG for each tuned model + CSV summary."""
    print("\n[INFO] Generating tuned confusion matrices...")

    cm_rows = []
    for res in results:
        name   = res["model"]
        y_pred = res["y_pred"]

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap="Greens")
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

        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        ax.set_title(f"Confusion Matrix (Tuned): {name}")
        plt.tight_layout()

        slug = MODEL_SLUGS[name]
        out  = PLOTS_P4 / f"confusion_matrix_tuned_{slug}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved: {out}")

        cm_rows.append({
            "model":          name,
            "true_negative":  res["true_negative"],
            "false_positive": res["false_positive"],
            "false_negative": res["false_negative"],
            "true_positive":  res["true_positive"],
        })

    pd.DataFrame(cm_rows).to_csv(REPORTS_P4 / "confusion_matrices.csv", index=False)
    print(f"  Saved: {REPORTS_P4 / 'confusion_matrices.csv'}")


# ===========================================================================
# 12. ROC CURVES
# ===========================================================================
def generate_roc_curves(results, y_test):
    """Combined ROC plot for all four tuned models."""
    print("\n[INFO] Generating tuned ROC curves...")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey",
            linewidth=1.2, label="Random (AUC=0.50)")

    for res in results:
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr, color=PLOT_COLORS[res["model"]], linewidth=1.8,
                label=f"{res['model']} (AUC={res['roc_auc']:.4f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves -- Tuned Models (Phase 4)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    out = PLOTS_P4 / "tuned_roc_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ===========================================================================
# 13. PR CURVES
# ===========================================================================
def generate_pr_curves(results, y_test):
    """Combined PR-curve plot for all four tuned models."""
    print("\n[INFO] Generating tuned Precision-Recall curves...")

    prevalence = float((y_test == 1).mean())
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(y=prevalence, linestyle="--", color="grey",
               linewidth=1.2, label=f"Prevalence ({prevalence:.3f})")

    for res in results:
        prec_v, rec_v, _ = precision_recall_curve(y_test, res["y_proba"])
        ax.plot(rec_v, prec_v, color=PLOT_COLORS[res["model"]], linewidth=1.8,
                label=f"{res['model']} (AP={res['pr_auc']:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves -- Tuned Models (Phase 4)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    plt.tight_layout()
    out = PLOTS_P4 / "tuned_pr_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ===========================================================================
# 14. OPTIONAL COMPARISON PLOTS
# ===========================================================================
def generate_comparison_plots(baseline_df, tuned_results, cv_rows):
    """Baseline vs tuned and CV variability plots."""
    print("\n[INFO] Generating comparison plots...")

    # --- Baseline vs tuned PR-AUC ---
    if baseline_df is not None:
        fig, ax = plt.subplots(figsize=(9, 5))
        x      = np.arange(len(MODEL_NAMES))
        width  = 0.35

        base_pr = [float(baseline_df.loc[n, "pr_auc"]) if n in baseline_df.index else 0
                   for n in MODEL_NAMES]
        tune_pr = [r["pr_auc"] for r in tuned_results]

        ax.bar(x - width/2, base_pr, width, label="Baseline (Phase 3)",
               color="#90CAF9", edgecolor="white")
        ax.bar(x + width/2, tune_pr, width, label="Tuned (Phase 4)",
               color="#1565C0", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_NAMES, rotation=10, ha="right")
        ax.set_ylim([0, 1.1])
        ax.set_ylabel("PR-AUC")
        ax.set_title("PR-AUC: Baseline vs Tuned")
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_P4 / "baseline_vs_tuned_pr_auc.png", dpi=150)
        plt.close()
        print(f"  Saved: {PLOTS_P4 / 'baseline_vs_tuned_pr_auc.png'}")

        # --- Baseline vs tuned ROC-AUC ---
        base_roc = [float(baseline_df.loc[n, "roc_auc"]) if n in baseline_df.index else 0
                    for n in MODEL_NAMES]
        tune_roc = [r["roc_auc"] for r in tuned_results]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width/2, base_roc, width, label="Baseline (Phase 3)",
               color="#A5D6A7", edgecolor="white")
        ax.bar(x + width/2, tune_roc, width, label="Tuned (Phase 4)",
               color="#2E7D32", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_NAMES, rotation=10, ha="right")
        ax.set_ylim([0, 1.1])
        ax.set_ylabel("ROC-AUC")
        ax.set_title("ROC-AUC: Baseline vs Tuned")
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_P4 / "baseline_vs_tuned_roc_auc.png", dpi=150)
        plt.close()
        print(f"  Saved: {PLOTS_P4 / 'baseline_vs_tuned_roc_auc.png'}")

    # --- CV PR-AUC variability ---
    fig, ax = plt.subplots(figsize=(9, 5))
    means = [r["cv_pr_auc_mean"] for r in cv_rows]
    stds  = [r["cv_pr_auc_std"]  for r in cv_rows]
    names = [r["model"] for r in cv_rows]
    x     = np.arange(len(names))

    bars = ax.bar(x, means, color=[PLOT_COLORS[n] for n in names],
                  alpha=0.85, edgecolor="white")
    ax.errorbar(x, means, yerr=stds, fmt="none", color="black",
                capsize=5, linewidth=1.5)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(stds) * 0.15,
                f"{m:.3f}\n+/-{s:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10, ha="right")
    ax.set_ylim([0, 1.1])
    ax.set_ylabel("CV PR-AUC (mean +/- std)")
    ax.set_title(f"CV PR-AUC Variability ({N_SPLITS}-fold)")
    plt.tight_layout()
    plt.savefig(PLOTS_P4 / "cv_pr_auc_variability.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_P4 / 'cv_pr_auc_variability.png'}")


# ===========================================================================
# 15. COMPARE BASELINE VS TUNED
# ===========================================================================
def compare_baseline_vs_tuned(baseline_df, tuned_results):
    """Build and save baseline_vs_tuned.csv and tuned_model_comparison.csv."""
    print("\n[INFO] Building baseline vs tuned comparison...")

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]

    # Tuned test results
    tuned_rows = []
    for res in tuned_results:
        row = {k: v for k, v in res.items()
               if k not in ("y_pred", "y_proba")}
        tuned_rows.append(row)
    pd.DataFrame(tuned_rows).to_csv(
        REPORTS_P4 / "tuned_model_comparison.csv", index=False
    )
    print(f"  Saved: {REPORTS_P4 / 'tuned_model_comparison.csv'}")

    if baseline_df is None:
        print("  [NOTE] Phase 3 baseline file not found -- skipping delta comparison")
        return

    # Delta comparison
    delta_rows = []
    for res in tuned_results:
        name = res["model"]
        if name not in baseline_df.index:
            continue
        row = {"model": name}
        for m in metrics:
            b = float(baseline_df.loc[name, m])
            t = float(res[m])
            row[f"baseline_{m}"] = round(b, 4)
            row[f"tuned_{m}"]    = round(t, 4)
            row[f"delta_{m}"]    = round(t - b, 4)
        delta_rows.append(row)

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(REPORTS_P4 / "baseline_vs_tuned.csv", index=False)
    print(f"  Saved: {REPORTS_P4 / 'baseline_vs_tuned.csv'}")

    print("\n  Baseline vs Tuned delta (tuned - baseline):")
    print(f"  {'Model':<22} {'dAcc':>7} {'dPrec':>7} {'dRec':>7} "
          f"{'dF1':>7} {'dROC':>7} {'dPR':>7}")
    print("  " + "-" * 65)
    for row in delta_rows:
        print(f"  {row['model']:<22} "
              f"{row['delta_accuracy']:>+7.4f} "
              f"{row['delta_precision']:>+7.4f} "
              f"{row['delta_recall']:>+7.4f} "
              f"{row['delta_f1']:>+7.4f} "
              f"{row['delta_roc_auc']:>+7.4f} "
              f"{row['delta_pr_auc']:>+7.4f}")

    return delta_df


# ===========================================================================
# 16. SAVE TUNED MODELS
# ===========================================================================
def save_tuned_models(search_results):
    """Save each best estimator to models/tuned/."""
    print("\n[INFO] Saving tuned models...")
    paths = {}
    for name, search in search_results.items():
        slug = MODEL_SLUGS[name]
        path = MODELS_TUNED / f"{slug}_tuned.pkl"
        joblib.dump(search.best_estimator_, path)
        paths[name] = path
        print(f"  Saved: {path}")
    return paths


# ===========================================================================
# 17. VALIDATE SAVED MODELS
# ===========================================================================
def validate_saved_models(model_paths, search_results, X_test):
    """Reload each saved model and verify predictions match on 10-sample check."""
    print("\n[INFO] Validating saved tuned models...")
    X_sample = X_test.iloc[:10]
    for name, path in model_paths.items():
        loaded      = joblib.load(path)
        orig_pred   = search_results[name].best_estimator_.predict(X_sample)
        reload_pred = loaded.predict(X_sample)
        assert np.array_equal(orig_pred, reload_pred), \
            f"{name}: reloaded predictions differ"
        print(f"  [OK] {name} reloaded, predictions match")


# ===========================================================================
# 18. SAVE SEARCH CONFIGURATION
# ===========================================================================
def save_search_configuration(spaces, search_results):
    """Record search configuration metadata."""
    config = {}
    for name, search in search_results.items():
        _, grid, stype = spaces[name]
        if stype == "grid":
            n_combos = 1
            for v in grid.values():
                n_combos *= len(v)
            n_fits = n_combos * N_SPLITS
        else:
            n_iter = N_ITER_RF if name == "Random Forest" else N_ITER_XGB
            n_fits = n_iter * N_SPLITS
            n_combos = n_iter

        config[name] = {
            "search_type":         stype,
            "n_combinations_or_iter": n_combos,
            "n_cv_folds":          N_SPLITS,
            "n_total_fits":        n_fits,
            "primary_scoring":     PRIMARY_METRIC,
            "random_state":        RANDOM_STATE,
            "parameter_space":     {k: (v if isinstance(v, list) else str(v))
                                    for k, v in grid.items()},
        }

    with open(REPORTS_P4 / "search_configuration.json", "w") as fh:
        json.dump(config, fh, indent=2, default=str)
    print(f"  Saved: {REPORTS_P4 / 'search_configuration.json'}")


# ===========================================================================
# 19. GENERATE MARKDOWN REPORT
# ===========================================================================
def generate_report(X_train, X_test, y_train, y_test, feature_names,
                    cv_rows, best_params, tuned_results,
                    baseline_df, spaces):
    """Write the Phase 4 Markdown report."""
    print("\n[INFO] Writing Phase 4 report...")

    n_train   = len(y_train)
    n_test    = len(y_test)
    conf_test = int((y_test == 1).sum())
    fp_test   = int((y_test == 0).sum())

    lines = [
        "# Phase 4 Model Comparison & Robust Evaluation Report",
        "## Kepler Exoplanet Classification \u2014 Explainable ML System",
        "",
        "*All metrics computed from actual data. No hard-coded values.*",
        "",
        "---",
        "",
        "## 1. Objective",
        "",
        "Phase 4 performs stratified cross-validation and hyperparameter tuning "
        "on the four baseline models from Phase 3. "
        "The goal is to establish more rigorous, generalisation-aware performance "
        "estimates before applying class-imbalance techniques (Phase 5) "
        "and model interpretation (Phase 6). "
        "No final model selection occurs here.",
        "",
        "---",
        "",
        "## 2. Methodology",
        "",
        f"| Item | Value |",
        f"|------|-------|",
        f"| CV strategy | StratifiedKFold(n_splits={N_SPLITS}, shuffle=True) |",
        f"| random_state | {RANDOM_STATE} |",
        f"| Primary metric | {PRIMARY_METRIC} (PR-AUC) |",
        "| LR search | GridSearchCV |",
        "| DT search | GridSearchCV |",
        f"| RF search | RandomizedSearchCV (n_iter={N_ITER_RF}) |",
        f"| XGB search | RandomizedSearchCV (n_iter={N_ITER_XGB}) |",
        "",
        "**Parameter spaces:**",
        "",
    ]
    for name in MODEL_NAMES:
        _, grid, stype = spaces[name]
        lines.append(f"- **{name}** ({stype}): `{grid}`")

    lines += [
        "",
        "---",
        "",
        "## 3. Cross-Validation Results",
        "",
        "| Model | CV PR-AUC mean | CV PR-AUC std | CV ROC-AUC | CV F1 |",
        "|-------|---------------|---------------|------------|-------|",
    ]
    for row in cv_rows:
        lines.append(
            f"| {row['model']} | {row['cv_pr_auc_mean']:.4f} "
            f"| {row['cv_pr_auc_std']:.4f} "
            f"| {row['cv_roc_auc_mean']:.4f} "
            f"| {row['cv_f1_mean']:.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Best Hyperparameters",
        "",
    ]
    for name, params in best_params.items():
        lines.append(f"**{name}:** `{params}`")
        lines.append("")

    lines += [
        "---",
        "",
        "## 5. Tuned Test-Set Results",
        "",
        f"| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
        f"|-------|----------|-----------|--------|----|---------|--------|",
    ]
    for res in tuned_results:
        lines.append(
            f"| {res['model']} | {res['accuracy']:.4f} | {res['precision']:.4f} "
            f"| {res['recall']:.4f} | {res['f1']:.4f} | "
            f"{res['roc_auc']:.4f} | {res['pr_auc']:.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 6. Baseline vs Tuned",
        "",
    ]
    if baseline_df is not None:
        lines += [
            "| Model | Metric | Baseline | Tuned | Delta |",
            "|-------|--------|----------|-------|-------|",
        ]
        for res in tuned_results:
            name = res["model"]
            if name not in baseline_df.index:
                continue
            for m in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
                b = float(baseline_df.loc[name, m])
                t = float(res[m])
                lines.append(f"| {name} | {m} | {b:.4f} | {t:.4f} | {t-b:+.4f} |")
    else:
        lines.append("Phase 3 baseline file not found; comparison skipped.")

    lines += [
        "",
        "---",
        "",
        "## 7. Stability (CV PR-AUC)",
        "",
        "| Model | Mean | Std | Min Fold | Max Fold |",
        "|-------|------|-----|----------|----------|",
    ]
    for row in cv_rows:
        lines.append(
            f"| {row['model']} | {row['cv_pr_auc_mean']:.4f} "
            f"| {row['cv_pr_auc_std']:.4f} "
            f"| {row['cv_pr_auc_min']:.4f} "
            f"| {row['cv_pr_auc_max']:.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 8. Limitations",
        "",
        "- No SMOTE or oversampling applied",
        "- No class weighting (`class_weight='balanced'`) applied",
        "- No threshold optimization performed",
        "- No SHAP or feature importance analysis performed",
        "- No final model selection made",
        "",
        "These will be addressed in Phases 5, 6, and 7.",
        "",
        "---",
        "",
        "## 9. Environment",
        "",
        f"| Item | Value |",
        f"|------|-------|",
        f"| Python | {platform.python_version()} |",
        f"| scikit-learn | {sklearn.__version__} |",
        f"| XGBoost | {xgboost.__version__} |",
        f"| Train samples | {n_train:,} |",
        f"| Test samples | {n_test:,} |",
        f"| Features | {len(feature_names)} |",
        "",
        "---",
        "",
        "## 10. Conclusion",
        "",
        "Phase 4 has established cross-validated and hyperparameter-tuned "
        "performance estimates for all four classifiers. "
        "The tuned metrics serve as the reference point for "
        "Phase 5 (class-imbalance handling) and Phase 6 (model interpretation). "
        "No final production model has been selected.",
        "",
        "**Phase 4 is complete. Phase 5 will investigate class-imbalance techniques.**",
    ]

    path = REPORTS_P4 / "phase_4_model_comparison_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {path}")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 70)
    print("PHASE 4: MODEL COMPARISON & ROBUST EVALUATION")
    print("Kepler Exoplanet Classification")
    print("=" * 70)

    # 1. Load Phase 2 data
    X_train, X_test, y_train, y_test, feature_names = load_phase_2_data()

    # 2. Load Phase 3 baseline (for comparison; optional)
    baseline_df = load_phase_3_baseline_results()

    # 3. CV strategy (shared across all searches)
    cv = create_cv_strategy()

    # 4. Define search spaces
    spaces = define_search_spaces()

    # 5-8. Tune all four models on X_train only
    search_results = {}
    _, search_results["Logistic Regression"] = tune_logistic_regression(
        spaces, X_train, y_train, cv)
    _, search_results["Decision Tree"]       = tune_decision_tree(
        spaces, X_train, y_train, cv)
    _, search_results["Random Forest"]       = tune_random_forest(
        spaces, X_train, y_train, cv)
    _, search_results["XGBoost"]             = tune_xgboost(
        spaces, X_train, y_train, cv)

    # 9. Collect CV results
    cv_rows, best_params = collect_cv_results(search_results, X_train, y_train, cv)

    # 10. Final evaluation on test set (FIRST test access in Phase 4)
    tuned_results = evaluate_tuned_models_on_test(search_results, X_test, y_test)

    # 11. Confusion matrices
    generate_confusion_matrices(tuned_results, y_test)

    # 12. ROC curves
    generate_roc_curves(tuned_results, y_test)

    # 13. PR curves
    generate_pr_curves(tuned_results, y_test)

    # 14. Optional comparison plots
    generate_comparison_plots(baseline_df, tuned_results, cv_rows)

    # 15. Baseline vs tuned CSVs
    compare_baseline_vs_tuned(baseline_df, tuned_results)

    # 16. Save tuned models
    model_paths = save_tuned_models(search_results)

    # 17. Validate saved models
    validate_saved_models(model_paths, search_results, X_test)

    # 18. Save search configuration
    save_search_configuration(spaces, search_results)
    print(f"  Saved: {REPORTS_P4 / 'search_configuration.json'}")

    # 19. Markdown report
    generate_report(X_train, X_test, y_train, y_test, feature_names,
                    cv_rows, best_params, tuned_results, baseline_df, spaces)

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE -- TUNED MODEL RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} "
          f"{'F1':>6} {'ROC':>6} {'PR':>6}")
    print("  " + "-" * 62)
    for res in tuned_results:
        print(f"  {res['model']:<22} {res['accuracy']:>6.4f} "
              f"{res['precision']:>6.4f} {res['recall']:>6.4f} "
              f"{res['f1']:>6.4f} {res['roc_auc']:>6.4f} {res['pr_auc']:>6.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
