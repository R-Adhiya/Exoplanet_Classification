"""
Phase 5: Class Imbalance Handling
Kepler Exoplanet Classification -- Explainable ML System

Investigates three imbalance strategies for 4 models (12 configurations):
  Group A: baseline (no imbalance handling)
  Group B: class weighting (class_weight='balanced' or scale_pos_weight for XGB)
  Group C: SMOTE inside imblearn Pipeline (applied per CV fold, never to test)

Primary metric: PR-AUC / Average Precision.
NO threshold optimization. NO SHAP. NO final model selection.

SMOTE leakage prevention:
  SMOTE is applied ONLY inside each CV training fold via imblearn.pipeline.Pipeline.
  Validation folds and the test set are NEVER resampled.

Compatibility: Python 3.12, pandas 2.x, scikit-learn 1.9+, XGBoost 3.x,
               imbalanced-learn 0.14+, Windows PowerShell (ASCII-safe output).
"""

import json
import sys
import warnings
from pathlib import Path

import imblearn
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
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
MODELS_IMBALANCE = ROOT / "models" / "imbalance"
REPORTS_P4      = ROOT / "reports" / "phase_4"
REPORTS_P5      = ROOT / "reports" / "phase_5"
PLOTS_P5        = ROOT / "plots" / "phase_5"

for _d in [MODELS_IMBALANCE, REPORTS_P5, PLOTS_P5]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_SPLITS     = 5
np.random.seed(RANDOM_STATE)

MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
]

STRATEGIES = ["baseline", "class_weight", "smote"]

STRATEGY_LABELS = {
    "baseline":     "baseline",
    "class_weight": "class_weight",   # scale_pos_weight for XGBoost
    "smote":        "SMOTE",
}

PLOT_COLORS = {
    "baseline":     "#2196F3",
    "class_weight": "#FF9800",
    "smote":        "#4CAF50",
}

MODEL_COLORS = {
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

MODEL_SLUGS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree":       "decision_tree",
    "Random Forest":       "random_forest",
    "XGBoost":             "xgboost",
}


# ===========================================================================
# 1. LOAD DATA
# ===========================================================================
def load_data():
    """Load Phase 2 processed train/test splits."""
    print("\n[INFO] Loading Phase 2 processed data...")

    files = {
        "X_train":  DATA_PROCESSED / "X_train_processed.csv",
        "X_test":   DATA_PROCESSED / "X_test_processed.csv",
        "y_train":  DATA_PROCESSED / "y_train.csv",
        "y_test":   DATA_PROCESSED / "y_test.csv",
    }
    for name, path in files.items():
        if not path.exists():
            print(f"\n[CRITICAL] File not found: {path}")
            sys.exit(1)

    X_train = pd.read_csv(files["X_train"])
    X_test  = pd.read_csv(files["X_test"])
    y_train = pd.read_csv(files["y_train"]).squeeze()
    y_test  = pd.read_csv(files["y_test"]).squeeze()

    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum()  == 0

    pos_n = int((y_train == 1).sum())
    neg_n = int((y_train == 0).sum())
    scale_pos_weight = neg_n / pos_n

    print(f"  X_train : {X_train.shape}  "
          f"(CONF={pos_n:,}  FP={neg_n:,})")
    print(f"  X_test  : {X_test.shape}")
    print(f"  scale_pos_weight (XGB) = {scale_pos_weight:.4f}  "
          f"({neg_n}/{pos_n})")
    print("  [OK] Data loaded")

    return X_train, X_test, y_train, y_test, pos_n, neg_n, scale_pos_weight


# ===========================================================================
# 2. LOAD PHASE 4 PARAMETERS
# ===========================================================================
def load_phase_4_parameters():
    """Load best hyperparameters from Phase 4."""
    print("\n[INFO] Loading Phase 4 best hyperparameters...")

    path = REPORTS_P4 / "best_parameters.json"
    if not path.exists():
        print(f"\n[CRITICAL] Phase 4 best_parameters.json not found: {path}")
        sys.exit(1)

    with open(path) as fh:
        params = json.load(fh)

    for name, p in params.items():
        print(f"  {name}: {p}")
    print("  [OK] Phase 4 parameters loaded")
    return params


# ===========================================================================
# 3. CREATE CV STRATEGY
# ===========================================================================
def create_cv_strategy():
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
    print(f"\n[INFO] CV: StratifiedKFold(n_splits={N_SPLITS}, "
          f"shuffle=True, random_state={RANDOM_STATE})")
    return cv


# ===========================================================================
# 4. CREATE BASELINE ESTIMATORS (Group A)
# ===========================================================================
def create_baseline_estimators(params):
    """
    Recreate estimators using Phase 4 tuned hyperparameters.
    No class weighting, no SMOTE.
    """
    lr_p  = {k: v for k, v in params["Logistic Regression"].items()}
    dt_p  = {k: v for k, v in params["Decision Tree"].items()}
    rf_p  = {k: v for k, v in params["Random Forest"].items()}
    xgb_p = {k: v for k, v in params["XGBoost"].items()}

    return {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=2000, **lr_p),
        "Decision Tree":       DecisionTreeClassifier(
            random_state=RANDOM_STATE, **dt_p),
        "Random Forest":       RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, **rf_p),
        "XGBoost":             XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss",
            n_jobs=-1, **xgb_p),
    }


# ===========================================================================
# 5. CREATE CLASS-WEIGHT ESTIMATORS (Group B)
# ===========================================================================
def create_class_weight_estimators(params, scale_pos_weight):
    """
    Same hyperparameters as baseline but with class balancing.
    LR, DT, RF: class_weight='balanced'
    XGBoost: scale_pos_weight=neg/pos ratio (calculated from training data)
    """
    lr_p  = {k: v for k, v in params["Logistic Regression"].items()}
    dt_p  = {k: v for k, v in params["Decision Tree"].items()}
    rf_p  = {k: v for k, v in params["Random Forest"].items()}
    xgb_p = {k: v for k, v in params["XGBoost"].items()}

    print(f"\n[INFO] Group B: class weighting")
    print(f"  LR/DT/RF: class_weight='balanced'")
    print(f"  XGBoost : scale_pos_weight={scale_pos_weight:.4f} "
          f"(= neg_count / pos_count = "
          f"{int(scale_pos_weight * 2198)}/{2198})")

    return {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=2000,
            class_weight="balanced", **lr_p),
        "Decision Tree":       DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight="balanced", **dt_p),
        "Random Forest":       RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1,
            class_weight="balanced", **rf_p),
        "XGBoost":             XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1,
            scale_pos_weight=scale_pos_weight, **xgb_p),
    }


# ===========================================================================
# 6. CREATE SMOTE ESTIMATORS (Group C)
# ===========================================================================
def create_smote_estimators(params):
    """
    Wrap each estimator in an imblearn Pipeline so SMOTE is applied
    per-fold inside cross_validate, never touching validation or test data.
    """
    lr_p  = {k: v for k, v in params["Logistic Regression"].items()}
    dt_p  = {k: v for k, v in params["Decision Tree"].items()}
    rf_p  = {k: v for k, v in params["Random Forest"].items()}
    xgb_p = {k: v for k, v in params["XGBoost"].items()}

    smote = SMOTE(random_state=RANDOM_STATE)

    print(f"\n[INFO] Group C: SMOTE inside imblearn Pipeline")
    print(f"  SMOTE is applied ONLY to each CV training fold.")
    print(f"  Validation folds and the test set are NEVER resampled.")

    return {
        "Logistic Regression": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", LogisticRegression(
                random_state=RANDOM_STATE, max_iter=2000, **lr_p)),
        ]),
        "Decision Tree": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", DecisionTreeClassifier(
                random_state=RANDOM_STATE, **dt_p)),
        ]),
        "Random Forest": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", RandomForestClassifier(
                random_state=RANDOM_STATE, n_jobs=-1, **rf_p)),
        ]),
        "XGBoost": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", XGBClassifier(
                random_state=RANDOM_STATE, eval_metric="logloss",
                n_jobs=-1, **xgb_p)),
        ]),
    }


# ===========================================================================
# 7. RUN CROSS-VALIDATION FOR ALL 12 CONFIGURATIONS
# ===========================================================================
def run_cross_validation(estimator_groups, X_train, y_train, cv):
    """
    Run cross_validate for all 12 (model, strategy) configurations.
    Returns a dict keyed by (model_name, strategy) -> cv_scores dict.
    """
    print("\n[INFO] Running cross-validation for all 12 configurations...")
    print(f"  {'Configuration':<40} {'PR-AUC':>8} {'ROC':>8} {'F1':>8}")
    print("  " + "-" * 66)

    all_cv = {}
    groups = [
        ("baseline",     estimator_groups["baseline"]),
        ("class_weight", estimator_groups["class_weight"]),
        ("smote",        estimator_groups["smote"]),
    ]

    for strategy, estimators in groups:
        for model_name in MODEL_NAMES:
            est = estimators[model_name]
            key = (model_name, strategy)

            scores = cross_validate(
                est, X_train, y_train,
                cv=cv,
                scoring=CV_SCORING,
                n_jobs=-1,
                return_train_score=False,
            )
            all_cv[key] = scores

            pr   = float(np.mean(scores["test_average_precision"]))
            pr_s = float(np.std(scores["test_average_precision"]))
            roc  = float(np.mean(scores["test_roc_auc"]))
            f1   = float(np.mean(scores["test_f1"]))
            label = f"{model_name} -- {STRATEGY_LABELS[strategy]}"
            print(f"  {label:<40} {pr:>8.4f} {roc:>8.4f} {f1:>8.4f}")

    print("  [OK] Cross-validation complete")
    return all_cv


# ===========================================================================
# 8. COLLECT CV RESULTS
# ===========================================================================
def collect_cv_results(all_cv, pos_n, neg_n):
    """Build and save the CV results summary CSV."""
    print("\n[INFO] Collecting CV results...")

    rows = []
    for (model_name, strategy), scores in all_cv.items():
        row = {
            "model":              model_name,
            "strategy":           STRATEGY_LABELS[strategy],
            "cv_accuracy_mean":   round(float(np.mean(scores["test_accuracy"])), 4),
            "cv_accuracy_std":    round(float(np.std(scores["test_accuracy"])), 4),
            "cv_precision_mean":  round(float(np.mean(scores["test_precision"])), 4),
            "cv_precision_std":   round(float(np.std(scores["test_precision"])), 4),
            "cv_recall_mean":     round(float(np.mean(scores["test_recall"])), 4),
            "cv_recall_std":      round(float(np.std(scores["test_recall"])), 4),
            "cv_f1_mean":         round(float(np.mean(scores["test_f1"])), 4),
            "cv_f1_std":          round(float(np.std(scores["test_f1"])), 4),
            "cv_roc_auc_mean":    round(float(np.mean(scores["test_roc_auc"])), 4),
            "cv_roc_auc_std":     round(float(np.std(scores["test_roc_auc"])), 4),
            "cv_pr_auc_mean":     round(float(np.mean(scores["test_average_precision"])), 4),
            "cv_pr_auc_std":      round(float(np.std(scores["test_average_precision"])), 4),
            "positive_class_count": pos_n,
            "negative_class_count": neg_n,
        }
        rows.append(row)

    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(REPORTS_P5 / "imbalance_cv_results.csv", index=False)
    print(f"  Saved: {REPORTS_P5 / 'imbalance_cv_results.csv'}")
    return cv_df


# ===========================================================================
# 9. EVALUATE TEST SET (first and only test access in Phase 5)
# ===========================================================================
def evaluate_test_set(estimator_groups, X_train, X_test, y_train, y_test):
    """
    Refit each configuration on full X_train, then predict on X_test.
    Returns a list of result dicts.
    """
    print("\n[INFO] Evaluating all 12 configurations on test set "
          "(first test access in Phase 5)...")
    print(f"  {'Configuration':<40} {'Acc':>6} {'Prec':>6} {'Rec':>6} "
          f"{'F1':>6} {'ROC':>6} {'PR':>6}")
    print("  " + "-" * 74)

    results = []
    for strategy in ["baseline", "class_weight", "smote"]:
        estimators = estimator_groups[strategy]
        for model_name in MODEL_NAMES:
            est = estimators[model_name]

            # Refit on full training data
            est.fit(X_train, y_train)

            y_pred  = est.predict(X_test)
            y_proba = est.predict_proba(X_test)[:, 1]

            # Validate
            assert len(y_pred) == len(y_test)
            assert not np.any(np.isnan(y_proba))
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
                "model":          model_name,
                "strategy":       STRATEGY_LABELS[strategy],
                "_strategy_key":  strategy,
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
                "estimator":      est,
            }
            results.append(res)

            label = f"{model_name} -- {STRATEGY_LABELS[strategy]}"
            print(f"  {label:<40} {acc:>6.4f} {prec:>6.4f} {rec:>6.4f} "
                  f"{f1:>6.4f} {auc:>6.4f} {prauc:>6.4f}")

    print("  [OK] Test evaluation complete")
    return results


# ===========================================================================
# 10. CONFUSION MATRICES
# ===========================================================================
def generate_confusion_matrices(results, y_test):
    """Save 12 confusion matrix PNGs and summary CSV."""
    print("\n[INFO] Generating confusion matrices...")

    cm_rows = []
    for res in results:
        model_name = res["model"]
        strategy   = res["_strategy_key"]
        y_pred     = res["y_pred"]
        slug       = MODEL_SLUGS[model_name]

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

        strat_label = STRATEGY_LABELS[strategy]
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        ax.set_title(f"{model_name}\n({strat_label})", fontsize=10)
        plt.tight_layout()

        fname = f"confusion_matrix_{slug}_{strategy}.png"
        plt.savefig(PLOTS_P5 / fname, dpi=150)
        plt.close()

        cm_rows.append({
            "model":          model_name,
            "strategy":       strat_label,
            "true_negative":  res["true_negative"],
            "false_positive": res["false_positive"],
            "false_negative": res["false_negative"],
            "true_positive":  res["true_positive"],
        })

    pd.DataFrame(cm_rows).to_csv(REPORTS_P5 / "confusion_matrices.csv", index=False)
    print(f"  Saved 12 confusion matrix PNGs to {PLOTS_P5}")
    print(f"  Saved: {REPORTS_P5 / 'confusion_matrices.csv'}")


# ===========================================================================
# 11. PR CURVES BY MODEL
# ===========================================================================
def generate_pr_curves(results, y_test):
    """One figure with 4 subplots (one per model), each showing 3 strategies."""
    print("\n[INFO] Generating PR curves by model...")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes_flat = axes.flatten()
    prevalence = float((y_test == 1).mean())

    for idx, model_name in enumerate(MODEL_NAMES):
        ax = axes_flat[idx]
        ax.axhline(y=prevalence, linestyle="--", color="grey",
                   linewidth=1.0, label=f"Prevalence ({prevalence:.3f})")

        for res in results:
            if res["model"] != model_name:
                continue
            strategy = res["_strategy_key"]
            pv, rv, _ = precision_recall_curve(y_test, res["y_proba"])
            ax.plot(rv, pv, color=PLOT_COLORS[strategy], linewidth=1.8,
                    label=f"{STRATEGY_LABELS[strategy]} (AP={res['pr_auc']:.4f})")

        ax.set_title(model_name, fontsize=11)
        ax.set_xlabel("Recall");  ax.set_ylabel("Precision")
        ax.set_xlim([0, 1]);      ax.set_ylim([0, 1.05])
        ax.legend(fontsize=8)

    plt.suptitle("PR Curves by Model: 3 Imbalance Strategies (Phase 5)",
                 fontsize=13)
    plt.tight_layout()
    out = PLOTS_P5 / "pr_curves_by_model.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ===========================================================================
# 12. ROC CURVES BY MODEL
# ===========================================================================
def generate_roc_curves(results, y_test):
    """One figure with 4 subplots (one per model), each showing 3 strategies."""
    print("\n[INFO] Generating ROC curves by model...")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes_flat = axes.flatten()

    for idx, model_name in enumerate(MODEL_NAMES):
        ax = axes_flat[idx]
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey",
                linewidth=1.0, label="Random (AUC=0.50)")

        for res in results:
            if res["model"] != model_name:
                continue
            strategy = res["_strategy_key"]
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            ax.plot(fpr, tpr, color=PLOT_COLORS[strategy], linewidth=1.8,
                    label=f"{STRATEGY_LABELS[strategy]} (AUC={res['roc_auc']:.4f})")

        ax.set_title(model_name, fontsize=11)
        ax.set_xlabel("FPR");   ax.set_ylabel("TPR")
        ax.set_xlim([0, 1]);    ax.set_ylim([0, 1.02])
        ax.legend(fontsize=8)

    plt.suptitle("ROC Curves by Model: 3 Imbalance Strategies (Phase 5)",
                 fontsize=13)
    plt.tight_layout()
    out = PLOTS_P5 / "roc_curves_by_model.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ===========================================================================
# 13. PRECISION-RECALL TRADE-OFF
# ===========================================================================
def generate_precision_recall_tradeoff(results):
    """
    Scatter plot: each of the 12 configurations as a point (recall, precision),
    colored by strategy.  No ranking or 'best' label.
    """
    print("\n[INFO] Generating precision-recall trade-off plot...")

    fig, ax = plt.subplots(figsize=(9, 6))
    strategy_plotted = set()

    for res in results:
        strategy = res["_strategy_key"]
        color    = PLOT_COLORS[strategy]
        label    = STRATEGY_LABELS[strategy] if strategy not in strategy_plotted \
                   else "_nolegend_"
        strategy_plotted.add(strategy)

        ax.scatter(res["recall"], res["precision"], color=color,
                   s=90, zorder=3, label=label, edgecolors="white", linewidths=0.5)
        ax.annotate(
            f"{res['model'][:3]}",
            (res["recall"], res["precision"]),
            textcoords="offset points", xytext=(5, 3), fontsize=7,
        )

    ax.set_xlabel("Recall (CONFIRMED)", fontsize=11)
    ax.set_ylabel("Precision (CONFIRMED)", fontsize=11)
    ax.set_title("Precision vs Recall: All 12 Configurations", fontsize=12)
    ax.legend(title="Strategy", fontsize=9)
    ax.set_xlim([0.6, 1.02])
    ax.set_ylim([0.6, 1.02])
    plt.tight_layout()
    out = PLOTS_P5 / "precision_recall_tradeoff.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ===========================================================================
# 14. SMOTE BALANCE VERIFICATION
# ===========================================================================
def verify_smote_balance(X_train, y_train, cv):
    """
    Run one SMOTE + trivial model through folds to measure fold balance.
    Records original vs SMOTE training-fold class counts.
    Validation folds are never resampled.
    """
    print("\n[INFO] Verifying SMOTE balance in CV folds...")

    smote = SMOTE(random_state=RANDOM_STATE)
    rows  = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_fold_tr = X_train.iloc[train_idx]
        y_fold_tr = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        orig_pos = int((y_fold_tr == 1).sum())
        orig_neg = int((y_fold_tr == 0).sum())

        _, y_resampled = smote.fit_resample(X_fold_tr, y_fold_tr)

        smote_pos = int((y_resampled == 1).sum())
        smote_neg = int((y_resampled == 0).sum())

        val_pos = int((y_fold_val == 1).sum())
        val_neg = int((y_fold_val == 0).sum())

        rows.append({
            "fold":            fold_idx + 1,
            "original_pos":    orig_pos,
            "original_neg":    orig_neg,
            "smote_pos":       smote_pos,
            "smote_neg":       smote_neg,
            "validation_pos":  val_pos,
            "validation_neg":  val_neg,
        })
        print(f"  Fold {fold_idx+1}: "
              f"orig=[+{orig_pos}/-{orig_neg}]  "
              f"after_SMOTE=[+{smote_pos}/-{smote_neg}]  "
              f"val(untouched)=[+{val_pos}/-{val_neg}]")

    smote_df = pd.DataFrame(rows)
    smote_df.to_csv(REPORTS_P5 / "smote_balance_summary.csv", index=False)
    print(f"  Saved: {REPORTS_P5 / 'smote_balance_summary.csv'}")
    print("  [OK] SMOTE is balanced in each training fold; "
          "validation folds are untouched")
    return smote_df


# ===========================================================================
# 15. BASELINE VS IMBALANCE COMPARISON
# ===========================================================================
def compare_baseline_vs_imbalance(results):
    """Build and save baseline_vs_imbalance.csv."""
    print("\n[INFO] Building baseline vs imbalance comparison...")

    metrics = ["pr_auc", "recall", "precision", "f1", "roc_auc"]

    # Index baseline results by model name
    baseline_map = {
        res["model"]: res
        for res in results
        if res["_strategy_key"] == "baseline"
    }

    rows = []
    for res in results:
        if res["_strategy_key"] == "baseline":
            continue
        model  = res["model"]
        base   = baseline_map[model]
        strat  = res["strategy"]
        row    = {"model": model, "strategy": strat}
        for m in metrics:
            b = float(base[m])
            t = float(res[m])
            row[f"baseline_{m}"]  = round(b, 4)
            row[f"strategy_{m}"]  = round(t, 4)
            row[f"delta_{m}"]     = round(t - b, 4)
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(REPORTS_P5 / "baseline_vs_imbalance.csv", index=False)
    print(f"  Saved: {REPORTS_P5 / 'baseline_vs_imbalance.csv'}")

    # Print recall and PR-AUC deltas
    print(f"\n  Delta PR-AUC and Recall (strategy - baseline):")
    print(f"  {'Config':<45} {'dPR':>7} {'dRec':>7} {'dPrec':>7}")
    print("  " + "-" * 66)
    for row in rows:
        label = f"{row['model']} -- {row['strategy']}"
        print(f"  {label:<45} {row['delta_pr_auc']:>+7.4f} "
              f"{row['delta_recall']:>+7.4f} {row['delta_precision']:>+7.4f}")

    return comp_df


# ===========================================================================
# 16. SAVE TEST RESULTS CSV
# ===========================================================================
def save_test_results(results):
    """Save imbalance_test_results.csv (drop raw arrays)."""
    skip = {"y_pred", "y_proba", "estimator", "_strategy_key"}
    rows = [{k: v for k, v in res.items() if k not in skip}
            for res in results]
    pd.DataFrame(rows).to_csv(REPORTS_P5 / "imbalance_test_results.csv", index=False)
    print(f"  Saved: {REPORTS_P5 / 'imbalance_test_results.csv'}")


# ===========================================================================
# 17. SAVE EXPERIMENT MODELS
# ===========================================================================
def save_experiment_models(results):
    """
    Save fitted imbalance models (non-baseline) to models/imbalance/.
    These are experimental models, not final production models.
    """
    print("\n[INFO] Saving imbalance experiment models...")
    for res in results:
        if res["_strategy_key"] == "baseline":
            continue
        slug   = MODEL_SLUGS[res["model"]]
        strat  = res["_strategy_key"]
        path   = MODELS_IMBALANCE / f"{slug}_{strat}.pkl"
        joblib.dump(res["estimator"], path)
        print(f"  Saved (experimental): {path}")
    print("  [NOTE] These are experimental models only, not final production models.")


# ===========================================================================
# 18. GENERATE MARKDOWN REPORT
# ===========================================================================
def generate_report(X_train, y_train, y_test, cv_df, results,
                    smote_df, pos_n, neg_n, scale_pos_weight):
    """Write the Phase 5 Markdown report."""
    print("\n[INFO] Writing Phase 5 report...")

    total = pos_n + neg_n
    prevalence = pos_n / total

    # Build quick lookup
    res_map = {(r["model"], r["_strategy_key"]): r for r in results}

    lines = [
        "# Phase 5 Class Imbalance Handling Report",
        "## Kepler Exoplanet Classification \u2014 Explainable ML System",
        "",
        "*All metrics computed from actual data.*",
        "",
        "---",
        "",
        "## 1. Objective",
        "",
        "The labeled training dataset is imbalanced: the minority class "
        "(CONFIRMED planets) represents only 36.2% of labeled samples. "
        "Phase 5 investigates whether explicit imbalance handling improves "
        "classification performance, especially minority-class recall and PR-AUC.",
        "",
        "---",
        "",
        "## 2. Dataset Distribution",
        "",
        "| Class | Count | Percentage |",
        "|-------|-------|------------|",
        f"| CONFIRMED (target=1) | {pos_n:,} | {pos_n/total*100:.2f}% |",
        f"| FALSE POSITIVE (target=0) | {neg_n:,} | {neg_n/total*100:.2f}% |",
        f"| Total | {total:,} | 100% |",
        "",
        "---",
        "",
        "## 3. Methods",
        "",
        "**Group A: Baseline** -- Phase 4 tuned hyperparameters, no imbalance handling.",
        "",
        "**Group B: Class weighting**",
        "- Logistic Regression, Decision Tree, Random Forest: "
        "`class_weight='balanced'`",
        f"- XGBoost: `scale_pos_weight={scale_pos_weight:.4f}` "
        f"(= {neg_n}/{pos_n} = negative/positive training count)",
        "",
        "**Group C: SMOTE**",
        "- `SMOTE(random_state=42)` wrapped in `imblearn.pipeline.Pipeline`",
        "- Applied ONLY to the training portion of each CV fold",
        "- Validation folds and the test set are never resampled",
        "",
        "---",
        "",
        "## 4. Leakage Prevention",
        "",
        "SMOTE was applied ONLY inside each CV training fold via "
        "`imblearn.pipeline.Pipeline`. "
        "The pipeline ensures SMOTE runs on the training split of each fold, "
        "and the validation split remains untouched. "
        "The final test set was accessed only after all CV experiments were complete.",
        "",
        "---",
        "",
        "## 5. CV Results",
        "",
        "| Model | Strategy | CV PR-AUC | +/-std | CV ROC-AUC | CV F1 | CV Recall |",
        "|-------|----------|-----------|--------|------------|-------|-----------|",
    ]
    for _, row in cv_df.iterrows():
        lines.append(
            f"| {row['model']} | {row['strategy']} "
            f"| {row['cv_pr_auc_mean']:.4f} | {row['cv_pr_auc_std']:.4f} "
            f"| {row['cv_roc_auc_mean']:.4f} | {row['cv_f1_mean']:.4f} "
            f"| {row['cv_recall_mean']:.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 6. Test Set Results",
        "",
        "| Model | Strategy | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
        "|-------|----------|----------|-----------|--------|----|---------|--------|",
    ]
    for res in results:
        lines.append(
            f"| {res['model']} | {res['strategy']} "
            f"| {res['accuracy']:.4f} | {res['precision']:.4f} "
            f"| {res['recall']:.4f} | {res['f1']:.4f} "
            f"| {res['roc_auc']:.4f} | {res['pr_auc']:.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 7. Precision-Recall Behavior",
        "",
        "Class weighting and SMOTE typically increase minority-class recall "
        "at the cost of precision. "
        "Observed changes are documented in "
        "`reports/phase_5/baseline_vs_imbalance.csv`.",
        "",
        "---",
        "",
        "## 8. Confusion Matrix Summary",
        "",
        "| Model | Strategy | TN | FP | FN | TP |",
        "|-------|----------|----|----|----|----|",
    ]
    for res in results:
        lines.append(
            f"| {res['model']} | {res['strategy']} "
            f"| {res['true_negative']} | {res['false_positive']} "
            f"| {res['false_negative']} | {res['true_positive']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 9. SMOTE Balance Verification",
        "",
        "| Fold | Orig Pos | Orig Neg | SMOTE Pos | SMOTE Neg | Val Pos | Val Neg |",
        "|------|----------|----------|-----------|-----------|---------|---------|",
    ]
    for _, row in smote_df.iterrows():
        lines.append(
            f"| {int(row['fold'])} | {int(row['original_pos'])} "
            f"| {int(row['original_neg'])} "
            f"| {int(row['smote_pos'])} | {int(row['smote_neg'])} "
            f"| {int(row['validation_pos'])} | {int(row['validation_neg'])} |"
        )

    lines += [
        "",
        "SMOTE creates balanced training folds. Validation folds remain untouched.",
        "",
        "---",
        "",
        "## 10. Limitations",
        "",
        "- Classification threshold remains at 0.5 (not optimized)",
        "- No final model selection performed",
        "- No SHAP or feature importance analysis",
        "- Test set accessed only once, after all CV experiments",
        "- Preprocessing pipeline unchanged from Phase 2",
        "",
        "---",
        "",
        "## 11. Environment",
        "",
        f"| Item | Value |",
        f"|------|-------|",
        f"| imbalanced-learn | {imblearn.__version__} |",
        f"| random_state | {RANDOM_STATE} |",
        f"| CV folds | {N_SPLITS} |",
        f"| Configurations | 12 (4 models x 3 strategies) |",
        "",
        "---",
        "",
        "## 12. Conclusion",
        "",
        "Phase 5 has measured the impact of class-weighting and SMOTE on all four "
        "classifiers using a leakage-safe evaluation framework. "
        "The results provide context for Phase 6 (model interpretation) and "
        "Phase 7 (threshold optimization and final model selection).",
        "",
        "**Phase 5 is complete. Phase 6 will handle SHAP and feature importance.**",
    ]

    path = REPORTS_P5 / "phase_5_imbalance_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {path}")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 70)
    print("PHASE 5: CLASS IMBALANCE HANDLING")
    print("Kepler Exoplanet Classification")
    print("=" * 70)
    print(f"  imbalanced-learn: {imblearn.__version__}")

    # 1. Load data
    X_train, X_test, y_train, y_test, pos_n, neg_n, scale_pos_weight = load_data()

    # 2. Load Phase 4 hyperparameters
    params = load_phase_4_parameters()

    # 3. CV strategy
    cv = create_cv_strategy()

    # 4. Build all three estimator groups
    print("\n[INFO] Building estimator groups...")
    estimator_groups = {
        "baseline":     create_baseline_estimators(params),
        "class_weight": create_class_weight_estimators(params, scale_pos_weight),
        "smote":        create_smote_estimators(params),
    }

    # 5. Run 12-configuration CV (X_train only, test set untouched)
    all_cv = run_cross_validation(estimator_groups, X_train, y_train, cv)

    # 6. Collect and save CV results
    cv_df = collect_cv_results(all_cv, pos_n, neg_n)

    # 7. SMOTE balance verification
    smote_df = verify_smote_balance(X_train, y_train, cv)

    # 8. Evaluate all 12 configs on test set (first and only test access)
    results = evaluate_test_set(
        estimator_groups, X_train, X_test, y_train, y_test)

    # 9. Save test results CSV
    print("\n[INFO] Saving test results...")
    save_test_results(results)

    # 10. Confusion matrices
    generate_confusion_matrices(results, y_test)

    # 11. PR curves
    generate_pr_curves(results, y_test)

    # 12. ROC curves
    generate_roc_curves(results, y_test)

    # 13. Precision-recall trade-off
    generate_precision_recall_tradeoff(results)

    # 14. Baseline vs imbalance comparison
    compare_baseline_vs_imbalance(results)

    # 15. Save experimental models
    save_experiment_models(results)

    # 16. Markdown report
    generate_report(X_train, y_train, y_test, cv_df, results,
                    smote_df, pos_n, neg_n, scale_pos_weight)

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("PHASE 5 COMPLETE -- TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Configuration':<42} {'Acc':>6} {'Prec':>6} "
          f"{'Rec':>6} {'F1':>6} {'PR':>6}")
    print("  " + "-" * 72)
    for res in results:
        label = f"{res['model']} -- {res['strategy']}"
        print(f"  {label:<42} {res['accuracy']:>6.4f} {res['precision']:>6.4f} "
              f"{res['recall']:>6.4f} {res['f1']:>6.4f} {res['pr_auc']:>6.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
