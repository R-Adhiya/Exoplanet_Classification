"""
Phase 3: Baseline Modeling
Kepler Exoplanet Classification -- Explainable ML System

Trains four baseline classifiers on Phase 2 preprocessed data:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. XGBoost

Evaluates with Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
NO hyperparameter tuning. NO SMOTE. NO class weighting. NO SHAP.

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
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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
ROOT           = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "models"
MODELS_BASELINE = MODELS_DIR / "baseline"
REPORTS_P3     = ROOT / "reports" / "phase_3"
PLOTS_P3       = ROOT / "plots" / "phase_3"

for _d in [MODELS_BASELINE, REPORTS_P3, PLOTS_P3]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
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


# ===========================================================================
# 1. LOAD PROCESSED DATA
# ===========================================================================
def load_processed_data():
    """
    Load Phase 2 preprocessed train/test splits and feature names.
    Returns (X_train, X_test, y_train, y_test, feature_names).
    Exits with error if any required file is missing.
    """
    print("\n[INFO] Loading Phase 2 processed data...")

    required = {
        "X_train": DATA_PROCESSED / "X_train_processed.csv",
        "X_test":  DATA_PROCESSED / "X_test_processed.csv",
        "y_train": DATA_PROCESSED / "y_train.csv",
        "y_test":  DATA_PROCESSED / "y_test.csv",
        "features": DATA_PROCESSED / "processed_feature_names.json",
    }
    for name, path in required.items():
        if not path.exists():
            print(f"\n[CRITICAL] Required file not found: {path}")
            print("  Run Phase 2 first to generate preprocessed datasets.")
            sys.exit(1)

    X_train = pd.read_csv(required["X_train"])
    X_test  = pd.read_csv(required["X_test"])
    y_train = pd.read_csv(required["y_train"]).squeeze()
    y_test  = pd.read_csv(required["y_test"]).squeeze()

    with open(required["features"]) as fh:
        feature_names = json.load(fh)

    # Validate shapes
    if X_train.shape[1] != 11:
        print(f"\n[CRITICAL] X_train has {X_train.shape[1]} features, expected 11")
        sys.exit(1)
    if X_test.shape[1] != 11:
        print(f"\n[CRITICAL] X_test has {X_test.shape[1]} features, expected 11")
        sys.exit(1)

    # Validate target values
    for name, y in [("y_train", y_train), ("y_test", y_test)]:
        unique = sorted(y.unique().tolist())
        if unique != [0, 1]:
            print(f"\n[CRITICAL] {name} contains unexpected values: {unique}")
            sys.exit(1)

    # Validate no NaN in features
    for name, X in [("X_train", X_train), ("X_test", X_test)]:
        nan_count = X.isnull().sum().sum()
        if nan_count > 0:
            print(f"\n[CRITICAL] {name} contains {nan_count} NaN values")
            sys.exit(1)

    print(f"  X_train shape   : {X_train.shape}")
    print(f"  X_test shape    : {X_test.shape}")
    print(f"  y_train dist    : CONFIRMED={int((y_train==1).sum()):,}  "
          f"FP={int((y_train==0).sum()):,}")
    print(f"  y_test dist     : CONFIRMED={int((y_test==1).sum()):,}  "
          f"FP={int((y_test==0).sum()):,}")
    print(f"  Features ({len(feature_names)}): {feature_names}")
    print("  [OK] All Phase 2 outputs loaded and validated")

    return X_train, X_test, y_train, y_test, feature_names


# ===========================================================================
# 2. INITIALIZE MODELS
# ===========================================================================
def initialize_models():
    """
    Create the four baseline model objects with deterministic random states.
    No hyperparameter tuning. No class weighting. No SMOTE.
    Returns a dict {name: model}.
    """
    print("\n[INFO] Initialising baseline models...")

    models = {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=2000,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }

    for name in models:
        print(f"  [OK] {name} initialised")

    return models


# ===========================================================================
# 3. TRAIN MODELS
# ===========================================================================
def train_models(models, X_train, y_train):
    """
    Fit every model on X_train / y_train only.
    Returns the same dict with fitted models.
    """
    print("\n[INFO] Training baseline models on training data...")
    fitted = {}
    for name, model in models.items():
        print(f"  Training {name}...", end="", flush=True)
        model.fit(X_train, y_train)
        fitted[name] = model
        print(" done")
    print("  [OK] All models trained")
    return fitted


# ===========================================================================
# 4. EVALUATE MODEL
# ===========================================================================
def evaluate_model(name, model, X_test, y_test):
    """
    Generate predictions and calculate all required metrics for one model.
    Returns a dict of metric values plus raw predictions.
    """
    y_pred   = model.predict(X_test)
    y_proba  = model.predict_proba(X_test)[:, 1]

    # Prediction validation
    assert len(y_pred)  == len(y_test), \
        f"{name}: prediction count {len(y_pred)} != test count {len(y_test)}"
    assert not np.any(np.isnan(y_proba)), f"{name}: NaN in probabilities"
    assert not np.any(np.isinf(y_proba)), f"{name}: Inf in probabilities"
    assert float(y_proba.min()) >= 0.0,  f"{name}: probability < 0"
    assert float(y_proba.max()) <= 1.0,  f"{name}: probability > 1"

    cm = confusion_matrix(y_test, y_pred)
    tn, fp_cm, fn, tp = cm.ravel()

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    auc   = roc_auc_score(y_test, y_proba)
    prauc = average_precision_score(y_test, y_proba)

    # Validate no NaN in metrics
    for metric_name, val in [("accuracy", acc), ("precision", prec),
                               ("recall", rec), ("f1", f1),
                               ("roc_auc", auc), ("pr_auc", prauc)]:
        if np.isnan(val) or np.isinf(val):
            print(f"\n[CRITICAL] {name}: {metric_name} is NaN/Inf")
            sys.exit(1)

    return {
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


# ===========================================================================
# 5. EVALUATE ALL MODELS
# ===========================================================================
def evaluate_all_models(fitted_models, X_test, y_test):
    """
    Evaluate every model and print a results table.
    Returns a list of result dicts.
    """
    print("\n[INFO] Evaluating models on test set...")
    print(f"  Test set size: {len(y_test):,}")
    print(f"  {'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} "
          f"{'F1':>6} {'ROC':>6} {'PR':>6}")
    print("  " + "-" * 62)

    results = []
    for name in MODEL_NAMES:
        model = fitted_models[name]
        res   = evaluate_model(name, model, X_test, y_test)
        results.append(res)
        print(f"  {name:<22} {res['accuracy']:>6.4f} {res['precision']:>6.4f} "
              f"{res['recall']:>6.4f} {res['f1']:>6.4f} "
              f"{res['roc_auc']:>6.4f} {res['pr_auc']:>6.4f}")

    print("  [OK] All models evaluated")
    return results


# ===========================================================================
# 6. CONFUSION MATRICES
# ===========================================================================
def generate_confusion_matrices(results, y_test):
    """
    Save a confusion matrix PNG for each model and a summary CSV.
    """
    print("\n[INFO] Generating confusion matrices...")

    cm_rows = []
    for res in results:
        name   = res["model"]
        y_pred = res["y_pred"]

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)

        classes = ["FALSE POS (0)", "CONFIRMED (1)"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(classes, rotation=15, ha="right")
        ax.set_yticklabels(classes)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=13, fontweight="bold")

        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        ax.set_title(f"Confusion Matrix: {name}")
        plt.tight_layout()

        slug = MODEL_SLUGS[name]
        out  = PLOTS_P3 / f"confusion_matrix_{slug}.png"
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

    cm_df = pd.DataFrame(cm_rows)
    cm_df.to_csv(REPORTS_P3 / "confusion_matrices.csv", index=False)
    print(f"  Saved: {REPORTS_P3 / 'confusion_matrices.csv'}")


# ===========================================================================
# 7. ROC CURVES
# ===========================================================================
def generate_roc_curves(results, y_test):
    """
    Combined ROC curve plot for all four models.
    """
    print("\n[INFO] Generating ROC curves...")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey",
            linewidth=1.2, label="Random classifier (AUC=0.50)")

    for res in results:
        name    = res["model"]
        y_proba = res["y_proba"]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr, color=PLOT_COLORS[name], linewidth=1.8,
                label=f"{name} (AUC={res['roc_auc']:.4f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves -- Baseline Models")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()

    out = PLOTS_P3 / "roc_curves_baseline.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ===========================================================================
# 8. PRECISION-RECALL CURVES
# ===========================================================================
def generate_pr_curves(results, y_test):
    """
    Combined Precision-Recall curve plot for all four models.
    Includes positive-class prevalence reference line.
    """
    print("\n[INFO] Generating Precision-Recall curves...")

    prevalence = float((y_test == 1).mean())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(y=prevalence, linestyle="--", color="grey",
               linewidth=1.2, label=f"Prevalence (CONFIRMED={prevalence:.3f})")

    for res in results:
        name    = res["model"]
        y_proba = res["y_proba"]
        prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_proba)
        ax.plot(rec_vals, prec_vals, color=PLOT_COLORS[name], linewidth=1.8,
                label=f"{name} (AP={res['pr_auc']:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves -- Baseline Models")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()

    out = PLOTS_P3 / "pr_curves_baseline.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ===========================================================================
# 9. METRIC COMPARISON BAR CHART
# ===========================================================================
def generate_metric_comparison(results):
    """
    Side-by-side bar chart of all six metrics across four models.
    No ranking or winner indicated.
    """
    print("\n[INFO] Generating metric comparison chart...")

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    x = np.arange(len(metrics))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, res in enumerate(results):
        vals = [res[m] for m in metrics]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=res["model"],
                      color=PLOT_COLORS[res["model"]], alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim([0, 1.12])
    ax.set_ylabel("Score")
    ax.set_title("Baseline Model Metric Comparison")
    ax.legend(fontsize=9)
    ax.axhline(y=1.0, linestyle="--", color="grey", linewidth=0.8, alpha=0.5)
    plt.tight_layout()

    out = PLOTS_P3 / "metric_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ===========================================================================
# 10. SAVE MODELS
# ===========================================================================
def save_models(fitted_models):
    """
    Save each fitted model as a .pkl file using joblib.
    """
    print("\n[INFO] Saving baseline models...")
    paths = {}
    for name, model in fitted_models.items():
        slug = MODEL_SLUGS[name]
        path = MODELS_BASELINE / f"{slug}.pkl"
        joblib.dump(model, path)
        paths[name] = path
        print(f"  Saved: {path}")
    return paths


# ===========================================================================
# 11. VALIDATE SAVED MODELS
# ===========================================================================
def validate_saved_models(model_paths, fitted_models, X_test):
    """
    Reload each saved model and verify that predictions match original.
    Uses first 10 test samples as a smoke test.
    """
    print("\n[INFO] Validating saved models (reload + predict smoke test)...")

    X_sample = X_test.iloc[:10]

    for name, path in model_paths.items():
        loaded     = joblib.load(path)
        orig_pred  = fitted_models[name].predict(X_sample)
        reload_pred = loaded.predict(X_sample)

        assert np.array_equal(orig_pred, reload_pred), \
            f"{name}: reloaded predictions differ from original"
        print(f"  [OK] {name} reloaded and predictions match (10-sample check)")


# ===========================================================================
# 12. SAVE REPORTS
# ===========================================================================
def save_reports(results, feature_names, y_test):
    """
    Save baseline_model_comparison.csv and classification_reports.json.
    y_test is passed explicitly to avoid reliance on the _y_test sentinel key.
    """
    print("\n[INFO] Saving reports...")

    # Comparison CSV -- drop raw prediction arrays and internal keys
    skip_keys = {"y_pred", "y_proba", "_y_test"}
    comp_rows = []
    for res in results:
        row = {k: v for k, v in res.items() if k not in skip_keys}
        comp_rows.append(row)
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(REPORTS_P3 / "baseline_model_comparison.csv", index=False)
    print(f"  Saved: {REPORTS_P3 / 'baseline_model_comparison.csv'}")

    # Classification reports JSON
    cr_data = {}
    for res in results:
        report_dict = classification_report(
            y_test, res["y_pred"],
            target_names=["FALSE_POSITIVE", "CONFIRMED"],
            output_dict=True,
        )
        cr_data[res["model"]] = report_dict
    with open(REPORTS_P3 / "classification_reports.json", "w") as fh:
        json.dump(cr_data, fh, indent=2)
    print(f"  Saved: {REPORTS_P3 / 'classification_reports.json'}")


# ===========================================================================
# 13. GENERATE MARKDOWN REPORT
# ===========================================================================
def generate_markdown_report(results, X_train, X_test, y_train, y_test,
                              feature_names):
    """Write the Phase 3 Markdown baseline report."""
    print("\n[INFO] Writing Phase 3 report...")

    n_train    = len(y_train)
    n_test     = len(y_test)
    conf_test  = int((y_test == 1).sum())
    fp_test    = int((y_test == 0).sum())
    prevalence = conf_test / n_test

    lines = [
        "# Phase 3 Baseline Modeling Report",
        "## Kepler Exoplanet Classification \u2014 Explainable ML System",
        "",
        "*All metrics computed from actual model predictions on the held-out test set.*",
        "",
        "---",
        "",
        "## 1. Objective",
        "",
        "Phase 3 establishes baseline classification performance for four standard "
        "classifiers trained on the Phase 2 preprocessed dataset. "
        "No hyperparameter tuning, class rebalancing, threshold optimization, "
        "or feature selection is performed in this phase. "
        "These baselines serve as the reference point for all subsequent phases.",
        "",
        "---",
        "",
        "## 2. Dataset",
        "",
        "| Item | Value |",
        "|------|-------|",
        f"| Training samples | {n_train:,} |",
        f"| Test samples | {n_test:,} |",
        f"| Features | {len(feature_names)} |",
        f"| CONFIRMED in test (target=1) | {conf_test:,} ({conf_test/n_test*100:.1f}%) |",
        f"| FALSE POSITIVE in test (target=0) | {fp_test:,} ({fp_test/n_test*100:.1f}%) |",
        f"| Positive class prevalence | {prevalence:.4f} |",
        "",
        "Features used:",
        "",
    ]
    for feat in feature_names:
        lines.append(f"- `{feat}`")

    lines += [
        "",
        "---",
        "",
        "## 3. Models",
        "",
        "All models use `random_state=42`. No hyperparameter tuning applied.",
        "",
        "| Model | Key Parameters |",
        "|-------|----------------|",
        "| Logistic Regression | `max_iter=2000`, default regularization |",
        "| Decision Tree | sklearn defaults |",
        "| Random Forest | `n_jobs=-1`, sklearn defaults |",
        "| XGBoost | `eval_metric='logloss'`, `n_jobs=-1`, XGBoost defaults |",
        "",
        "---",
        "",
        "## 4. Evaluation Metrics",
        "",
        "| Metric | Description |",
        "|--------|-------------|",
        "| Accuracy | Proportion of all correct predictions |",
        "| Precision | Of predicted CONFIRMED, fraction truly CONFIRMED |",
        "| Recall | Of true CONFIRMED, fraction correctly identified |",
        "| F1 | Harmonic mean of Precision and Recall |",
        "| ROC-AUC | Area under the ROC curve (threshold-independent) |",
        "| PR-AUC / AP | Area under the Precision-Recall curve (sensitive to imbalance) |",
        "",
        "PR-AUC is emphasized because the dataset is imbalanced "
        f"({conf_test/n_test*100:.1f}% positive class).",
        "",
        "---",
        "",
        "## 5. Baseline Metric Comparison",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
        "|-------|----------|-----------|--------|----|---------|--------|",
    ]
    for res in results:
        lines.append(
            f"| {res['model']} | {res['accuracy']:.4f} | {res['precision']:.4f} "
            f"| {res['recall']:.4f} | {res['f1']:.4f} | "
            f"{res['roc_auc']:.4f} | {res['pr_auc']:.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 6. Confusion Matrices",
        "",
        "| Model | TN | FP | FN | TP |",
        "|-------|----|----|----|----|",
    ]
    for res in results:
        lines.append(
            f"| {res['model']} | {res['true_negative']} | "
            f"{res['false_positive']} | {res['false_negative']} | "
            f"{res['true_positive']} |"
        )

    lines += [
        "",
        "TN = True Negative (FALSE POSITIVE correctly identified)",
        "FP = False Positive (FALSE POSITIVE misclassified as CONFIRMED)",
        "FN = False Negative (CONFIRMED misclassified as FALSE POSITIVE)",
        "TP = True Positive (CONFIRMED correctly identified)",
        "",
        "---",
        "",
        "## 7. ROC Analysis",
        "",
        "| Model | ROC-AUC |",
        "|-------|---------|",
    ]
    for res in results:
        lines.append(f"| {res['model']} | {res['roc_auc']:.4f} |")

    lines += [
        "",
        "ROC-AUC values are reported for comparison across models. "
        "A higher ROC-AUC indicates better discrimination overall, "
        "but ROC-AUC can be optimistic under class imbalance. "
        "No model is declared a winner based on this metric alone.",
        "",
        "---",
        "",
        "## 8. Precision-Recall Analysis",
        "",
        "| Model | PR-AUC / Average Precision |",
        "|-------|---------------------------|",
    ]
    for res in results:
        lines.append(f"| {res['model']} | {res['pr_auc']:.4f} |")

    lines += [
        "",
        f"Positive class prevalence = {prevalence:.4f}. "
        "A random classifier would achieve a PR-AUC approximately equal to "
        "the prevalence. Models substantially above prevalence demonstrate "
        "meaningful discrimination.",
        "",
        "---",
        "",
        "## 9. Baseline Observations",
        "",
    ]

    # Compute factual observations from actual metrics
    recalls   = {r["model"]: r["recall"]    for r in results}
    precisions= {r["model"]: r["precision"] for r in results}
    pr_aucs   = {r["model"]: r["pr_auc"]    for r in results}
    f1s       = {r["model"]: r["f1"]        for r in results}

    max_recall_model = max(recalls, key=recalls.get)
    min_recall_model = min(recalls, key=recalls.get)
    max_prec_model   = max(precisions, key=precisions.get)
    min_prec_model   = min(precisions, key=precisions.get)
    max_prauc_model  = max(pr_aucs, key=pr_aucs.get)
    min_prauc_model  = min(pr_aucs, key=pr_aucs.get)
    max_fn_model     = max(results, key=lambda r: r["false_negative"])["model"]
    max_fp_model     = max(results, key=lambda r: r["false_positive"])["model"]

    lines += [
        f"- **Recall**: {max_recall_model} produced the highest recall "
        f"({recalls[max_recall_model]:.4f}); "
        f"{min_recall_model} produced the lowest ({recalls[min_recall_model]:.4f}).",
        "",
        f"- **Precision**: {max_prec_model} produced the highest precision "
        f"({precisions[max_prec_model]:.4f}); "
        f"{min_prec_model} produced the lowest ({precisions[min_prec_model]:.4f}).",
        "",
        f"- **PR-AUC**: {max_prauc_model} produced the highest PR-AUC "
        f"({pr_aucs[max_prauc_model]:.4f}); "
        f"{min_prauc_model} produced the lowest ({pr_aucs[min_prauc_model]:.4f}).",
        "",
        f"- **False Negatives**: {max_fn_model} produced the most false negatives "
        f"({max(r['false_negative'] for r in results)}), meaning the most "
        "CONFIRMED planets were misclassified as FALSE POSITIVE.",
        "",
        f"- **False Positives**: {max_fp_model} produced the most false positives "
        f"({max(r['false_positive'] for r in results)}), meaning the most "
        "FALSE POSITIVE objects were misclassified as CONFIRMED.",
        "",
        "These are factual observations from baseline models. "
        "No ranking or final selection is made at this stage.",
        "",
        "---",
        "",
        "## 10. Limitations",
        "",
        "The following techniques have NOT been applied in Phase 3:",
        "",
        "- No hyperparameter tuning",
        "- No SMOTE or oversampling",
        "- No class weighting (`class_weight='balanced'`)",
        "- No threshold optimization",
        "- No SHAP or feature importance analysis",
        "- No cross-validation",
        "- No final model selection",
        "",
        "These will be addressed in later phases.",
        "",
        "---",
        "",
        "## 11. Environment",
        "",
        f"| Item | Version |",
        f"|------|---------|",
        f"| Python | {platform.python_version()} |",
        f"| scikit-learn | {sklearn.__version__} |",
        f"| XGBoost | {xgboost.__version__} |",
        f"| random_state | {RANDOM_STATE} |",
        f"| Train samples | {n_train:,} |",
        f"| Test samples | {n_test:,} |",
        f"| Features | {len(feature_names)} |",
        "",
        "---",
        "",
        "## 12. Conclusion",
        "",
        "Phase 3 has established baseline classification metrics for four standard "
        "models on the Kepler KOI dataset. "
        "The metrics above serve as the reference point for Phases 4 and beyond, "
        "where class balancing, hyperparameter tuning, threshold optimization, "
        "and SHAP analysis will be applied.",
        "",
        "**Phase 3 is complete. Phase 4 will handle advanced model comparison "
        "and hyperparameter tuning.**",
    ]

    report_path = REPORTS_P3 / "phase_3_baseline_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {report_path}")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 70)
    print("PHASE 3: BASELINE MODELING")
    print("Kepler Exoplanet Classification")
    print("=" * 70)

    # 1. Load Phase 2 outputs
    X_train, X_test, y_train, y_test, feature_names = load_processed_data()

    # 2. Initialise models
    models = initialize_models()

    # 3. Train on training data only
    fitted_models = train_models(models, X_train, y_train)

    # 4+5. Evaluate all models on test set
    results = evaluate_all_models(fitted_models, X_test, y_test)

    # 6. Confusion matrices
    generate_confusion_matrices(results, y_test)

    # 7. ROC curves
    generate_roc_curves(results, y_test)

    # 8. PR curves
    generate_pr_curves(results, y_test)

    # 9. Metric comparison bar chart
    generate_metric_comparison(results)

    # 10. Save models
    model_paths = save_models(fitted_models)

    # 11. Validate saved models
    validate_saved_models(model_paths, fitted_models, X_test)

    # 12. Save CSV/JSON reports
    save_reports(results, feature_names, y_test)

    # 13. Markdown report
    generate_markdown_report(results, X_train, X_test, y_train, y_test,
                             feature_names)

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE -- BASELINE RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} "
          f"{'F1':>6} {'ROC':>6} {'PR':>6}")
    print("  " + "-" * 62)
    for res in results:
        print(f"  {res['model']:<22} {res['accuracy']:>6.4f} "
              f"{res['precision']:>6.4f} {res['recall']:>6.4f} "
              f"{res['f1']:>6.4f} {res['roc_auc']:>6.4f} {res['pr_auc']:>6.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
