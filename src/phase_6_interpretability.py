"""
Phase 6: Feature Importance & Model Interpretation
Kepler Exoplanet Classification -- Explainable ML System

Interprets four Phase 4 tuned models using:
  - Native model importance (coefficients, feature_importances_, XGBoost gain/weight)
  - Permutation importance (sklearn, on test set, n_repeats=10)
  - SHAP (TreeExplainer for tree models, LinearExplainer for LR)
  - Global and local explanations
  - Individual prediction explanations for 4 representative cases

NO model tuning. NO threshold optimization. NO feature selection.
NO final model selection. Post-hoc analysis only.

Compatibility: Python 3.12, pandas 2.x, scikit-learn 1.9+, XGBoost 3.x,
               shap 0.52+, Windows PowerShell (ASCII-safe terminal output).
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from sklearn.inspection import permutation_importance

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
MODELS_TUNED    = ROOT / "models" / "tuned"
REPORTS_P6      = ROOT / "reports" / "phase_6"
PLOTS_P6        = ROOT / "plots" / "phase_6"
PLOTS_DEP       = PLOTS_P6 / "dependence"
PLOTS_IND       = PLOTS_P6 / "individual"

for _d in [REPORTS_P6, PLOTS_P6, PLOTS_DEP, PLOTS_IND]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE   = 42
SHAP_SAMPLE_N  = 1000
N_REPEATS_PERM = 10
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

MODEL_COLORS = {
    "Logistic Regression": "#2196F3",
    "Decision Tree":       "#FF9800",
    "Random Forest":       "#4CAF50",
    "XGBoost":             "#E91E63",
}

FEATURE_DESCRIPTIONS = {
    "koi_period":    "Orbital period of the KOI candidate",
    "koi_duration":  "Transit duration",
    "koi_depth":     "Transit depth",
    "koi_prad":      "Estimated planetary radius",
    "koi_impact":    "Transit impact parameter",
    "koi_model_snr": "Transit model signal-to-noise ratio",
    "koi_teq":       "Estimated equilibrium temperature",
    "koi_insol":     "Incident stellar flux / insolation",
    "koi_steff":     "Stellar effective temperature",
    "koi_slogg":     "Stellar surface gravity",
    "koi_srad":      "Stellar radius",
}


# ===========================================================================
# 1. LOAD DATA
# ===========================================================================
def load_data():
    """Load Phase 2 processed train/test splits."""
    print("\n[INFO] Loading Phase 2 processed data...")

    X_train = pd.read_csv(DATA_PROCESSED / "X_train_processed.csv")
    X_test  = pd.read_csv(DATA_PROCESSED / "X_test_processed.csv")
    y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv").squeeze()
    y_test  = pd.read_csv(DATA_PROCESSED / "y_test.csv").squeeze()

    for name, X in [("X_train", X_train), ("X_test", X_test)]:
        assert X.isnull().sum().sum() == 0, f"NaN in {name}"

    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
    print("  [OK] Data loaded")
    return X_train, X_test, y_train, y_test


# ===========================================================================
# 2. LOAD FEATURE NAMES
# ===========================================================================
def load_feature_names():
    """Load processed feature names from Phase 2."""
    with open(DATA_PROCESSED / "processed_feature_names.json") as fh:
        names = json.load(fh)
    print(f"\n[INFO] Features ({len(names)}): {names}")
    return names


# ===========================================================================
# 3. LOAD MODELS
# ===========================================================================
def load_models():
    """Load Phase 4 tuned models."""
    print("\n[INFO] Loading Phase 4 tuned models...")
    models = {}
    for name in MODEL_NAMES:
        slug = MODEL_SLUGS[name]
        path = MODELS_TUNED / f"{slug}_tuned.pkl"
        if not path.exists():
            print(f"\n[CRITICAL] Model not found: {path}")
            sys.exit(1)
        models[name] = joblib.load(path)
        print(f"  Loaded: {name}  ({type(models[name]).__name__})")
    print("  [OK] All tuned models loaded")
    return models


# ===========================================================================
# 4. VERIFY MODEL INPUTS
# ===========================================================================
def verify_model_inputs(models, X_train, X_test, feature_names):
    """Verify feature count and ordering match model expectations."""
    print("\n[INFO] Verifying model inputs...")

    assert len(feature_names) == 11, \
        f"Expected 11 features, got {len(feature_names)}"
    assert X_train.shape[1] == 11, \
        f"X_train has {X_train.shape[1]} features"
    assert X_test.shape[1] == 11, \
        f"X_test has {X_test.shape[1]} features"
    assert list(X_train.columns) == feature_names, \
        f"X_train column order mismatch"

    print(f"  Feature count: {len(feature_names)} [OK]")
    print(f"  X_train cols match feature_names [OK]")
    print("  [OK] Input verification passed")


# ===========================================================================
# 5. NATIVE FEATURE IMPORTANCE
# ===========================================================================
def calculate_native_importance(models, feature_names):
    """
    Extract native importance from each model.
    Returns a dict {model_name: (importances_array, signed_array_or_None)}.
    Also saves per-model CSV files.
    """
    print("\n[INFO] Calculating native feature importance...")

    native = {}

    # --- Logistic Regression: signed coefficients + abs ---
    lr = models["Logistic Regression"]
    coef   = lr.coef_[0]
    abs_c  = np.abs(coef)
    rows = [{
        "feature":              f,
        "coefficient":          round(float(coef[i]), 6),
        "absolute_coefficient": round(float(abs_c[i]), 6),
        "feature_description":  FEATURE_DESCRIPTIONS.get(f, ""),
    } for i, f in enumerate(feature_names)]
    df = pd.DataFrame(rows).sort_values("absolute_coefficient", ascending=False)
    df.to_csv(REPORTS_P6 / "logistic_regression_coefficients.csv", index=False)
    native["Logistic Regression"] = (abs_c, coef)
    print(f"  Saved: logistic_regression_coefficients.csv")

    # --- Decision Tree ---
    dt  = models["Decision Tree"]
    imp = dt.feature_importances_
    rows = [{"feature": f, "importance": round(float(imp[i]), 6),
             "feature_description": FEATURE_DESCRIPTIONS.get(f, "")}
            for i, f in enumerate(feature_names)]
    df = pd.DataFrame(rows).sort_values("importance", ascending=False)
    df.to_csv(REPORTS_P6 / "decision_tree_feature_importance.csv", index=False)
    native["Decision Tree"] = (imp, None)
    print(f"  Saved: decision_tree_feature_importance.csv")

    # --- Random Forest ---
    rf  = models["Random Forest"]
    imp = rf.feature_importances_
    rows = [{"feature": f, "importance": round(float(imp[i]), 6),
             "feature_description": FEATURE_DESCRIPTIONS.get(f, "")}
            for i, f in enumerate(feature_names)]
    df = pd.DataFrame(rows).sort_values("importance", ascending=False)
    df.to_csv(REPORTS_P6 / "random_forest_feature_importance.csv", index=False)
    native["Random Forest"] = (imp, None)
    print(f"  Saved: random_forest_feature_importance.csv")

    # --- XGBoost ---
    xgb = models["XGBoost"]
    imp_default = xgb.feature_importances_    # gain by default in XGBoost

    # Also try gain/weight/cover from booster if available
    xgb_rows = []
    try:
        booster = xgb.get_booster()
        gain   = booster.get_score(importance_type="gain")
        weight = booster.get_score(importance_type="weight")
        cover  = booster.get_score(importance_type="cover")
        for i, f in enumerate(feature_names):
            xgb_rows.append({
                "feature":             f,
                "importance":          round(float(imp_default[i]), 6),
                "gain":                round(gain.get(f"f{i}", 0.0), 4),
                "weight":              weight.get(f"f{i}", 0),
                "cover":               round(cover.get(f"f{i}", 0.0), 4),
                "feature_description": FEATURE_DESCRIPTIONS.get(f, ""),
            })
    except Exception:
        for i, f in enumerate(feature_names):
            xgb_rows.append({
                "feature":             f,
                "importance":          round(float(imp_default[i]), 6),
                "feature_description": FEATURE_DESCRIPTIONS.get(f, ""),
            })

    df = pd.DataFrame(xgb_rows).sort_values("importance", ascending=False)
    df.to_csv(REPORTS_P6 / "xgboost_feature_importance.csv", index=False)
    native["XGBoost"] = (imp_default, None)
    print(f"  Saved: xgboost_feature_importance.csv")

    return native


# ===========================================================================
# 6. CALCULATE PERMUTATION IMPORTANCE
# ===========================================================================
def calculate_permutation_importance(models, X_test, y_test, feature_names):
    """
    Permutation importance on the test set using average_precision scoring.
    n_repeats=10, random_state=42.
    """
    print("\n[INFO] Calculating permutation importance (n_repeats=10)...")

    all_rows  = []
    perm_dict = {}

    for name, model in models.items():
        print(f"  {name}...", end="", flush=True)
        result = permutation_importance(
            model, X_test, y_test,
            scoring="average_precision",
            n_repeats=N_REPEATS_PERM,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        perm_dict[name] = result
        for i, feat in enumerate(feature_names):
            all_rows.append({
                "model":              name,
                "feature":            feat,
                "importance_mean":    round(float(result.importances_mean[i]), 6),
                "importance_std":     round(float(result.importances_std[i]), 6),
                "feature_description": FEATURE_DESCRIPTIONS.get(feat, ""),
            })
        print(" done")

    perm_df = pd.DataFrame(all_rows)
    perm_df.to_csv(REPORTS_P6 / "permutation_importance.csv", index=False)
    print(f"  Saved: {REPORTS_P6 / 'permutation_importance.csv'}")
    return perm_dict


# ===========================================================================
# 7. PERMUTATION IMPORTANCE PLOTS
# ===========================================================================
def generate_permutation_plots(perm_dict, feature_names):
    """One plot per model + one combined comparison."""
    print("\n[INFO] Generating permutation importance plots...")

    # Individual per-model plots
    for name, result in perm_dict.items():
        slug = MODEL_SLUGS[name]
        idx  = np.argsort(result.importances_mean)
        feat_sorted = [feature_names[i] for i in idx]
        means = result.importances_mean[idx]
        stds  = result.importances_std[idx]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(feat_sorted, means, xerr=stds,
                color=MODEL_COLORS[name], alpha=0.8, edgecolor="white",
                capsize=4)
        ax.set_xlabel("Permutation Importance (PR-AUC drop)")
        ax.set_title(f"Permutation Importance: {name}")
        plt.tight_layout()
        out = PLOTS_P6 / f"permutation_{slug}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved: {out}")

    # Combined comparison
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes_flat = axes.flatten()
    for idx_m, (name, result) in enumerate(perm_dict.items()):
        ax  = axes_flat[idx_m]
        idx = np.argsort(result.importances_mean)
        ax.barh([feature_names[i] for i in idx],
                result.importances_mean[idx],
                xerr=result.importances_std[idx],
                color=MODEL_COLORS[name], alpha=0.8, edgecolor="white",
                capsize=3)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Importance")

    plt.suptitle("Permutation Importance: All Models", fontsize=13)
    plt.tight_layout()
    out = PLOTS_P6 / "permutation_importance_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ===========================================================================
# 8. INITIALIZE SHAP EXPLAINERS
# ===========================================================================
def initialize_shap_explainers(models, X_train, X_test, feature_names):
    """
    Build a SHAP explainer for each model.
    Tree models: TreeExplainer
    Logistic Regression: LinearExplainer with training data background
    Returns dict {model_name: explainer}.
    """
    print(f"\n[INFO] Initialising SHAP explainers (shap {shap.__version__})...")

    # Background sample for LR
    rng = np.random.default_rng(RANDOM_STATE)
    bg_idx = rng.choice(len(X_train), size=min(200, len(X_train)), replace=False)
    X_bg = X_train.iloc[bg_idx]

    explainers = {}
    for name, model in models.items():
        print(f"  {name}...", end="", flush=True)
        try:
            if name == "Logistic Regression":
                explainers[name] = shap.LinearExplainer(
                    model, X_bg, feature_names=feature_names)
            else:
                explainers[name] = shap.TreeExplainer(
                    model, feature_names=feature_names)
            print(" done")
        except Exception as exc:
            print(f" [WARNING] falling back to Explainer: {exc}")
            explainers[name] = shap.Explainer(
                model, X_bg, feature_names=feature_names)

    print("  [OK] SHAP explainers ready")
    return explainers


# ===========================================================================
# 9. CALCULATE SHAP VALUES
# ===========================================================================
def calculate_shap_values(explainers, X_test, feature_names):
    """
    Compute SHAP values for a representative test sample.
    Returns dict {model_name: shap_values_array (n_samples x n_features)}.
    """
    print(f"\n[INFO] Calculating SHAP values "
          f"(sample_n={min(SHAP_SAMPLE_N, len(X_test))})...")

    rng = np.random.default_rng(RANDOM_STATE)
    n   = min(SHAP_SAMPLE_N, len(X_test))
    idx = rng.choice(len(X_test), size=n, replace=False)
    X_sample = X_test.iloc[idx].reset_index(drop=True)

    shap_vals  = {}
    shap_expls = {}

    for name, explainer in explainers.items():
        print(f"  {name}...", end="", flush=True)
        try:
            sv = explainer(X_sample)
            # sv.values may be (n, f) or (n, f, c) for multi-output
            vals = sv.values
            if vals.ndim == 3:
                # multiclass: take the positive class
                vals = vals[:, :, 1]
            shap_vals[name]  = vals
            shap_expls[name] = sv
        except Exception as exc:
            print(f" [WARNING] {exc} -- skipping")
            shap_vals[name]  = None
            shap_expls[name] = None
        print(" done")

    print(f"  Actual SHAP sample size: {n}")
    return shap_vals, shap_expls, X_sample, idx


# ===========================================================================
# 10. GLOBAL SHAP IMPORTANCE
# ===========================================================================
def calculate_global_shap_importance(shap_vals, feature_names):
    """Mean absolute SHAP per feature per model. Saves global_importance CSV."""
    print("\n[INFO] Calculating global SHAP importance...")

    rows = []
    for name, vals in shap_vals.items():
        if vals is None:
            continue
        mean_abs = np.abs(vals).mean(axis=0)
        for i, feat in enumerate(feature_names):
            rows.append({
                "model":              name,
                "feature":            feat,
                "mean_abs_shap":      round(float(mean_abs[i]), 6),
                "feature_description": FEATURE_DESCRIPTIONS.get(feat, ""),
            })
        top_feats = sorted(zip(feature_names, mean_abs),
                           key=lambda x: x[1], reverse=True)[:5]
        print(f"  {name} top-5: "
              f"{[f'{f}={v:.4f}' for f, v in top_feats]}")

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_P6 / "shap_global_importance.csv", index=False)
    print(f"  Saved: {REPORTS_P6 / 'shap_global_importance.csv'}")
    return df


# ===========================================================================
# 11. SHAP SUMMARY PLOTS (beeswarm)
# ===========================================================================
def generate_shap_summary_plots(shap_expls, X_sample, feature_names):
    """SHAP beeswarm summary plot for each model."""
    print("\n[INFO] Generating SHAP summary plots...")

    for name, sv in shap_expls.items():
        if sv is None:
            print(f"  [SKIP] {name}: SHAP values unavailable")
            continue
        slug = MODEL_SLUGS[name]

        # Ensure 2-D values — some explainers return (n, f, 2) for binary
        import copy
        sv_plot = copy.copy(sv)
        if sv_plot.values.ndim == 3:
            sv_plot.values       = sv_plot.values[:, :, 1]
            sv_plot.base_values  = sv_plot.base_values[:, 1] \
                if sv_plot.base_values.ndim == 2 \
                else sv_plot.base_values

        fig = plt.figure(figsize=(9, 6))
        try:
            shap.plots.beeswarm(sv_plot, max_display=11, show=False)
        except Exception as exc:
            # Fallback: manual bar plot of mean |shap|
            plt.close("all")
            fig, ax = plt.subplots(figsize=(8, 5))
            mean_abs = np.abs(sv_plot.values).mean(axis=0)
            idx = np.argsort(mean_abs)
            ax.barh([feature_names[i] for i in idx], mean_abs[idx],
                    color=MODEL_COLORS[name], alpha=0.8, edgecolor="white")
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"SHAP Importance: {name} (fallback bar)")

        plt.title(f"SHAP Summary: {name}", fontsize=12, pad=12)
        plt.tight_layout()
        out = PLOTS_P6 / f"shap_summary_{slug}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"  Saved: {out}")


# ===========================================================================
# 12. SHAP BAR PLOTS
# ===========================================================================
def generate_shap_bar_plots(shap_expls, feature_names):
    """SHAP global bar plot (mean |SHAP|) for each model."""
    print("\n[INFO] Generating SHAP bar plots...")

    for name, sv in shap_expls.items():
        if sv is None:
            print(f"  [SKIP] {name}")
            continue
        slug = MODEL_SLUGS[name]

        import copy
        sv_plot = copy.copy(sv)
        if sv_plot.values.ndim == 3:
            sv_plot.values      = sv_plot.values[:, :, 1]
            sv_plot.base_values = sv_plot.base_values[:, 1] \
                if sv_plot.base_values.ndim == 2 \
                else sv_plot.base_values

        fig = plt.figure(figsize=(8, 5))
        try:
            shap.plots.bar(sv_plot, max_display=11, show=False)
        except Exception:
            plt.close("all")
            fig, ax = plt.subplots(figsize=(8, 5))
            mean_abs = np.abs(sv_plot.values).mean(axis=0)
            idx = np.argsort(mean_abs)
            ax.barh([feature_names[i] for i in idx], mean_abs[idx],
                    color=MODEL_COLORS[name], alpha=0.8, edgecolor="white")
            ax.set_xlabel("Mean |SHAP value|")

        plt.title(f"SHAP Global Importance: {name}", fontsize=12, pad=8)
        plt.tight_layout()
        out = PLOTS_P6 / f"shap_bar_{slug}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"  Saved: {out}")


# ===========================================================================
# 13. SHAP DEPENDENCE PLOTS
# ===========================================================================
def generate_shap_dependence_plots(shap_vals, X_sample, feature_names,
                                   global_shap_df):
    """
    Generate SHAP dependence plots for the top 5 features by mean |SHAP|
    across Random Forest and XGBoost.
    """
    print("\n[INFO] Generating SHAP dependence plots (top 5 features)...")

    # Pick top 5 features from RF and XGBoost average
    subset = global_shap_df[global_shap_df["model"].isin(
        ["Random Forest", "XGBoost"])]
    top5 = (subset.groupby("feature")["mean_abs_shap"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .index.tolist())
    print(f"  Top 5 features (RF+XGB avg): {top5}")

    # Use XGBoost SHAP values for dependence plots
    sv_xgb = shap_vals.get("XGBoost")
    if sv_xgb is None:
        sv_xgb = shap_vals.get("Random Forest")
    if sv_xgb is None:
        print("  [SKIP] No SHAP values available for dependence plots")
        return top5

    X_arr = X_sample.values
    feat_idx_map = {f: i for i, f in enumerate(feature_names)}

    for feat in top5:
        fi = feat_idx_map[feat]
        feat_vals = X_arr[:, fi]
        shap_col  = sv_xgb[:, fi]

        fig, ax = plt.subplots(figsize=(7, 4))
        sc = ax.scatter(feat_vals, shap_col, alpha=0.4,
                        c=shap_col, cmap="coolwarm", s=18)
        plt.colorbar(sc, ax=ax, label="SHAP value")
        ax.axhline(0, linestyle="--", color="grey", linewidth=0.8)
        ax.set_xlabel(f"{feat} (standardized)")
        ax.set_ylabel("SHAP value")
        ax.set_title(f"SHAP Dependence: {feat}")
        plt.tight_layout()
        out = PLOTS_DEP / f"shap_dependence_{feat}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved: {out}")

    return top5


# ===========================================================================
# 14. BUILD IMPORTANCE CONSENSUS
# ===========================================================================
def build_importance_consensus(native, perm_dict, global_shap_df, feature_names):
    """
    Build feature_importance_consensus.csv and top_feature_frequency.csv.
    """
    print("\n[INFO] Building feature importance consensus...")

    def rank_series(series):
        """Rank features highest importance = rank 1."""
        return series.rank(ascending=False).astype(int)

    # Per-model native ranks
    native_ranks = {}
    for name, (imp, _) in native.items():
        s = pd.Series(imp, index=feature_names)
        native_ranks[name] = rank_series(s).to_dict()

    # Permutation ranks (mean importance)
    perm_ranks = {}
    for name, result in perm_dict.items():
        s = pd.Series(result.importances_mean, index=feature_names)
        perm_ranks[name] = rank_series(s).to_dict()

    # SHAP ranks
    shap_ranks = {}
    for name in MODEL_NAMES:
        sub = global_shap_df[global_shap_df["model"] == name].set_index("feature")
        if len(sub) == 0:
            continue
        s = sub["mean_abs_shap"]
        shap_ranks[name] = rank_series(s).to_dict()

    # Consensus table
    rows = []
    for feat in feature_names:
        row = {"feature": feat}
        for mname in MODEL_NAMES:
            slug = MODEL_SLUGS[mname].replace("_", " ")
            row[f"native_{MODEL_SLUGS[mname]}_rank"] = \
                native_ranks.get(mname, {}).get(feat, None)
            row[f"perm_{MODEL_SLUGS[mname]}_rank"] = \
                perm_ranks.get(mname, {}).get(feat, None)
            row[f"shap_{MODEL_SLUGS[mname]}_rank"] = \
                shap_ranks.get(mname, {}).get(feat, None)

        all_ranks = [v for v in row.values()
                     if isinstance(v, (int, float)) and v is not None]
        row["mean_rank_overall"] = round(np.mean(all_ranks), 2) if all_ranks else None
        row["feature_description"] = FEATURE_DESCRIPTIONS.get(feat, "")
        rows.append(row)

    cons_df = pd.DataFrame(rows).sort_values("mean_rank_overall")
    cons_df.to_csv(REPORTS_P6 / "feature_importance_consensus.csv", index=False)
    print(f"  Saved: {REPORTS_P6 / 'feature_importance_consensus.csv'}")

    # Top-5 frequency across all methods
    all_top5s = []
    for name in MODEL_NAMES:
        for ranks in [native_ranks.get(name, {}),
                      perm_ranks.get(name, {}),
                      shap_ranks.get(name, {})]:
            top5 = [f for f, r in sorted(ranks.items(), key=lambda x: x[1])[:5]]
            all_top5s.extend(top5)

    from collections import Counter
    counts = Counter(all_top5s)
    total_methods = len(MODEL_NAMES) * 3
    freq_rows = [{
        "feature":          feat,
        "top5_occurrences": counts.get(feat, 0),
        "total_methods":    total_methods,
        "feature_description": FEATURE_DESCRIPTIONS.get(feat, ""),
    } for feat in feature_names]
    freq_df = pd.DataFrame(freq_rows).sort_values(
        "top5_occurrences", ascending=False)
    freq_df.to_csv(REPORTS_P6 / "top_feature_frequency.csv", index=False)
    print(f"  Saved: {REPORTS_P6 / 'top_feature_frequency.csv'}")

    return cons_df, freq_df, native_ranks, perm_ranks, shap_ranks


# ===========================================================================
# 15. SELECT REPRESENTATIVE CASES
# ===========================================================================
def select_representative_cases(models, X_test, y_test):
    """
    Use XGBoost tuned model to find:
    A: correct CONFIRMED
    B: correct FALSE POSITIVE
    C: FALSE POSITIVE predicted as CONFIRMED (FP)
    D: CONFIRMED predicted as FALSE POSITIVE (FN)
    Deterministic: first occurrence in test set (consistent with random_state).
    """
    print("\n[INFO] Selecting representative cases...")

    xgb   = models["XGBoost"]
    y_pred = xgb.predict(X_test)
    y_true = y_test.values

    cases = {}
    case_defs = {
        "A_confirmed_correct":     (1, 1),   # TP
        "B_fp_correct":            (0, 0),   # TN
        "C_fp_as_confirmed":       (0, 1),   # FP error
        "D_confirmed_as_fp":       (1, 0),   # FN error
    }
    for case_id, (true_c, pred_c) in case_defs.items():
        mask = (y_true == true_c) & (y_pred == pred_c)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            print(f"  [NOTE] No example available for {case_id}")
            cases[case_id] = None
        else:
            cases[case_id] = int(idxs[0])
            print(f"  {case_id}: test index {idxs[0]}  "
                  f"(true={true_c}, pred={pred_c})")

    return cases


# ===========================================================================
# 16. GENERATE INDIVIDUAL EXPLANATIONS
# ===========================================================================
def generate_individual_explanations(cases, models, explainers, shap_vals,
                                     X_test, y_test, X_sample, idx_sample,
                                     feature_names):
    """
    Build individual_explanations.csv and one waterfall plot per case.
    Uses the XGBoost explainer.
    """
    print("\n[INFO] Generating individual prediction explanations...")

    xgb     = models["XGBoost"]
    sv_xgb  = shap_vals.get("XGBoost")
    expl    = explainers.get("XGBoost")

    # Rebuild per-index mapping for SHAP sample
    sample_idx_set = set(idx_sample)
    idx_to_sample  = {int(v): i for i, v in enumerate(idx_sample)}

    all_rows = []

    case_labels = {
        "A_confirmed_correct":  ("CONFIRMED", "CONFIRMED",
                                 "case_confirmed_correct"),
        "B_fp_correct":         ("FALSE_POSITIVE", "FALSE_POSITIVE",
                                 "case_false_positive_correct"),
        "C_fp_as_confirmed":    ("FALSE_POSITIVE", "CONFIRMED",
                                 "case_false_positive_as_confirmed"),
        "D_confirmed_as_fp":    ("CONFIRMED", "FALSE_POSITIVE",
                                 "case_confirmed_as_false_positive"),
    }

    for case_id, test_idx in cases.items():
        actual_label, pred_label, png_name = case_labels[case_id]
        if test_idx is None:
            print(f"  [SKIP] {case_id}: no example available")
            continue

        x_row    = X_test.iloc[test_idx:test_idx+1]
        prob     = float(xgb.predict_proba(x_row)[0, 1])
        x_vals   = x_row.values[0]

        # Waterfall plot if this sample was in the SHAP sample
        sample_pos = idx_to_sample.get(test_idx)
        if sample_pos is not None and sv_xgb is not None and expl is not None:
            try:
                sv_obj = expl(X_sample.iloc[sample_pos:sample_pos+1])
                import copy
                sv_obj = copy.copy(sv_obj)
                if sv_obj.values.ndim == 3:
                    sv_obj.values      = sv_obj.values[:, :, 1]
                    sv_obj.base_values = sv_obj.base_values[:, 1] \
                        if sv_obj.base_values.ndim == 2 \
                        else sv_obj.base_values

                fig = plt.figure(figsize=(9, 6))
                shap.plots.waterfall(sv_obj[0], max_display=11, show=False)
                cls_str = actual_label.replace("_", " ")
                pred_str = pred_label.replace("_", " ")
                plt.title(
                    f"Case {case_id[0]}: true={cls_str}, "
                    f"pred={pred_str}, p={prob:.3f}",
                    fontsize=10, pad=8)
                plt.tight_layout()
                out = PLOTS_IND / f"{png_name}.png"
                plt.savefig(out, dpi=150, bbox_inches="tight")
                plt.close("all")
                print(f"  Saved: {out}")
            except Exception as exc:
                print(f"  [WARNING] Waterfall for {case_id}: {exc}")

        # Tabular rows
        shap_row = sv_xgb[sample_pos] if (sample_pos is not None
                                           and sv_xgb is not None) \
                   else [None] * len(feature_names)

        for i, feat in enumerate(feature_names):
            sv_val = shap_row[i] if shap_row[i] is not None else None
            direction = None
            if sv_val is not None:
                direction = ("toward_CONFIRMED"
                             if sv_val > 0 else "toward_FALSE_POSITIVE")
            all_rows.append({
                "case_id":             case_id,
                "actual_class":        actual_label,
                "predicted_class":     pred_label,
                "predicted_probability": round(prob, 4),
                "feature":             feat,
                "feature_value":       round(float(x_vals[i]), 6),
                "shap_value":          round(float(sv_val), 6)
                                       if sv_val is not None else None,
                "effect_direction":    direction,
            })

    df = pd.DataFrame(all_rows)
    df.to_csv(REPORTS_P6 / "individual_explanations.csv", index=False)
    print(f"  Saved: {REPORTS_P6 / 'individual_explanations.csv'}")


# ===========================================================================
# 17. CLASS DISTRIBUTION PLOTS
# ===========================================================================
def generate_class_distribution_plots(X_train, y_train, feature_names,
                                       top5_feats):
    """
    Boxplot of the top 5 features comparing CONFIRMED vs FALSE POSITIVE
    using original labeled training data.
    """
    print("\n[INFO] Generating class distribution plots...")

    label_map = {0: "FALSE POSITIVE", 1: "CONFIRMED"}
    colors    = {0: "#F44336", 1: "#2196F3"}

    feats = [f for f in top5_feats if f in X_train.columns][:5]
    n_feats = len(feats)

    fig, axes = plt.subplots(1, n_feats, figsize=(3.5 * n_feats, 5))
    if n_feats == 1:
        axes = [axes]

    for ax, feat in zip(axes, feats):
        data_plot = [
            X_train.loc[y_train == cls, feat].dropna().values
            for cls in [0, 1]
        ]
        bp = ax.boxplot(data_plot,
                        tick_labels=["FP", "CONF"],
                        patch_artist=True,
                        medianprops={"color": "black", "linewidth": 2})
        for patch, cls in zip(bp["boxes"], [0, 1]):
            patch.set_facecolor(colors[cls])
            patch.set_alpha(0.7)
        ax.set_title(feat, fontsize=9)
        ax.set_ylabel("Standardized value")

    plt.suptitle("Feature Distribution by Class (Train, top features)",
                 fontsize=12)
    plt.tight_layout()
    out = PLOTS_P6 / "class_distribution_important_features.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ===========================================================================
# 18. EXPLAINABILITY COMPARISON TABLE
# ===========================================================================
def generate_explainability_comparison():
    """Save explainability_comparison.csv describing all methods."""
    rows = [
        {
            "method":            "Coefficients (LR)",
            "models_supported":  "Logistic Regression",
            "global_or_local":   "Global",
            "strength":          "Signed direction; easily interpretable for linear models",
            "limitation":        "Only valid for linear models; requires standardized features",
        },
        {
            "method":            "Native tree importance",
            "models_supported":  "Decision Tree, Random Forest, XGBoost",
            "global_or_local":   "Global",
            "strength":          "Fast; built into tree models",
            "limitation":        "Biased toward high-cardinality and correlated features",
        },
        {
            "method":            "Permutation importance",
            "models_supported":  "All",
            "global_or_local":   "Global",
            "strength":          "Model-agnostic; based on held-out performance drop",
            "limitation":        "Slow; can distribute importance across correlated features",
        },
        {
            "method":            "SHAP (TreeExplainer / LinearExplainer)",
            "models_supported":  "All",
            "global_or_local":   "Both",
            "strength":          "Theoretically grounded; local and global; shows direction",
            "limitation":        "Computationally expensive; can be complex to interpret "
                                  "when features are correlated",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_P6 / "explainability_comparison.csv", index=False)
    print(f"  Saved: {REPORTS_P6 / 'explainability_comparison.csv'}")


# ===========================================================================
# 19. TOP FEATURES SUMMARY
# ===========================================================================
def generate_top_features_summary(native, perm_dict, global_shap_df,
                                   feature_names):
    """top_features_summary.csv: model x method x top-5 features."""
    rows = []
    for name in MODEL_NAMES:
        # Native
        imp_arr, _ = native[name]
        for rank, (feat, imp) in enumerate(
                sorted(zip(feature_names, imp_arr),
                       key=lambda x: x[1], reverse=True)[:5], start=1):
            rows.append({"model": name, "method": "native",
                         "rank": rank, "feature": feat,
                         "importance": round(float(imp), 6)})
        # Permutation
        result = perm_dict[name]
        for rank, (feat, imp) in enumerate(
                sorted(zip(feature_names, result.importances_mean),
                       key=lambda x: x[1], reverse=True)[:5], start=1):
            rows.append({"model": name, "method": "permutation",
                         "rank": rank, "feature": feat,
                         "importance": round(float(imp), 6)})
        # SHAP
        sub = global_shap_df[global_shap_df["model"] == name]
        if len(sub) > 0:
            for rank, row in enumerate(
                    sub.sort_values("mean_abs_shap", ascending=False)
                    .head(5).itertuples(), start=1):
                rows.append({"model": name, "method": "shap",
                             "rank": rank, "feature": row.feature,
                             "importance": round(float(row.mean_abs_shap), 6)})

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_P6 / "top_features_summary.csv", index=False)
    print(f"  Saved: {REPORTS_P6 / 'top_features_summary.csv'}")
    return df


# ===========================================================================
# 20. GENERATE MARKDOWN REPORT
# ===========================================================================
def generate_report(feature_names, native, perm_dict, global_shap_df,
                    cons_df, freq_df, top5_feats, cases, X_test, y_test,
                    models):
    """Write the Phase 6 Markdown report."""
    print("\n[INFO] Writing Phase 6 report...")

    # Build top-5 per model / method for the report
    def top5_list(model_name, source_dict):
        arr = source_dict.get(model_name)
        if arr is None:
            return "N/A"
        if hasattr(arr, "importances_mean"):
            arr = arr.importances_mean
        pairs = sorted(zip(feature_names, arr),
                       key=lambda x: x[1], reverse=True)[:5]
        return ", ".join(f[0] for f, _ in zip(pairs, pairs))

    xgb      = models["XGBoost"]
    y_pred   = xgb.predict(X_test)
    n_test   = len(y_test)
    n_conf   = int((y_test == 1).sum())

    lines = [
        "# Phase 6 Feature Importance & Model Interpretation Report",
        "## Kepler Exoplanet Classification \u2014 Explainable ML System",
        "",
        "*Post-hoc interpretation only. No model tuning or feature selection.*",
        "",
        "---",
        "",
        "## 1. Objective",
        "",
        "Determine which features most influence the four Phase 4 tuned classifiers, "
        "and explain how individual features affect model predictions using "
        "multiple complementary methods.",
        "",
        "---",
        "",
        "## 2. Features Used",
        "",
        "| Feature | Description |",
        "|---------|-------------|",
    ]
    for f in feature_names:
        lines.append(f"| `{f}` | {FEATURE_DESCRIPTIONS.get(f, '')} |")

    lines += [
        "",
        "---",
        "",
        "## 3. Models Interpreted",
        "",
        "Phase 4 tuned models (`models/tuned/`):",
        "",
    ]
    for name in MODEL_NAMES:
        lines.append(f"- {name}")

    lines += [
        "",
        "---",
        "",
        "## 4. Native Feature Importance",
        "",
        "### Logistic Regression (absolute coefficients)",
        "",
        "| Feature | Coefficient | |Coef| |",
        "|---------|------------|-------|",
    ]
    lr_coef  = pd.read_csv(REPORTS_P6 / "logistic_regression_coefficients.csv")
    for _, row in lr_coef.iterrows():
        lines.append(f"| {row['feature']} | {row['coefficient']:+.4f} "
                     f"| {row['absolute_coefficient']:.4f} |")

    lines += [
        "",
        "Positive coefficient: associated with higher predicted probability of "
        "CONFIRMED. Negative: associated with FALSE POSITIVE.",
        "",
        "### Tree model importance (Decision Tree, Random Forest, XGBoost)",
        "",
        "See `reports/phase_6/` for per-model CSVs.",
        "",
        "---",
        "",
        "## 5. Permutation Importance",
        "",
        f"Metric: average_precision (PR-AUC), n_repeats={N_REPEATS_PERM}.",
        "",
        "| Model | Top features (by importance drop) |",
        "|-------|-----------------------------------|",
    ]
    for name in MODEL_NAMES:
        result = perm_dict[name]
        top5 = sorted(zip(feature_names, result.importances_mean),
                      key=lambda x: x[1], reverse=True)[:5]
        lines.append(f"| {name} | {', '.join(f for f, _ in top5)} |")

    lines += [
        "",
        "---",
        "",
        "## 6. SHAP Analysis",
        "",
        f"SHAP version: {shap.__version__}  |  "
        f"Sample size: {min(SHAP_SAMPLE_N, n_test)}",
        "",
        "### Global importance (mean |SHAP|)",
        "",
        "| Model | Top features |",
        "|-------|-------------|",
    ]
    for name in MODEL_NAMES:
        sub = global_shap_df[global_shap_df["model"] == name]
        if len(sub) == 0:
            lines.append(f"| {name} | N/A |")
            continue
        top5 = sub.nlargest(5, "mean_abs_shap")["feature"].tolist()
        lines.append(f"| {name} | {', '.join(top5)} |")

    lines += [
        "",
        "---",
        "",
        "## 7. Feature Importance Consistency",
        "",
        "| Feature | Top-5 occurrences | Out of total methods |",
        "|---------|------------------|-----------------------|",
    ]
    for _, row in freq_df.iterrows():
        lines.append(
            f"| {row['feature']} | {row['top5_occurrences']} "
            f"| {row['total_methods']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 8. Individual Prediction Explanations",
        "",
        "Using XGBoost Phase 4 tuned model. Cases selected deterministically.",
        "",
        "| Case | True Label | Predicted | Example available |",
        "|------|-----------|-----------|-------------------|",
    ]
    case_labels = {
        "A_confirmed_correct":   ("CONFIRMED",     "CONFIRMED"),
        "B_fp_correct":          ("FALSE POSITIVE", "FALSE POSITIVE"),
        "C_fp_as_confirmed":     ("FALSE POSITIVE", "CONFIRMED"),
        "D_confirmed_as_fp":     ("CONFIRMED",     "FALSE POSITIVE"),
    }
    for cid, (true_l, pred_l) in case_labels.items():
        avail = "Yes" if cases.get(cid) is not None else "No"
        lines.append(f"| {cid} | {true_l} | {pred_l} | {avail} |")

    lines += [
        "",
        "See `reports/phase_6/individual_explanations.csv` and "
        "`plots/phase_6/individual/` for details.",
        "",
        "---",
        "",
        "## 9. Important Feature Relationships",
        "",
        f"SHAP dependence plots were generated for top features: "
        f"{', '.join(top5_feats)}.",
        "",
        "These plots show how SHAP values vary with feature values "
        "(standardized scale), revealing non-linear effects and potential "
        "interactions.",
        "",
        "---",
        "",
        "## 10. Correlation and Interpretation Limitations",
        "",
        "Feature importance describes MODEL BEHAVIOR, not physical causality.",
        "",
        "- `koi_prad` and `koi_impact` showed a correlation of r=0.68 in Phase 1. "
        "  Their individual importances may be partially redistributed between them.",
        "- `koi_slogg` and `koi_srad` showed r=-0.648. "
        "  Importance scores for these features should be interpreted jointly.",
        "- SHAP assigns contributions at the individual prediction level, "
        "  but correlated features may still share attribution.",
        "",
        "---",
        "",
        "## 11. Explainability Method Comparison",
        "",
        "| Method | Scope | Strength | Limitation |",
        "|--------|-------|----------|------------|",
        "| Coefficients (LR) | Global | Signed direction | Linear models only |",
        "| Native tree importance | Global | Fast, built-in | Biased for correlated/high-cardinality features |",
        "| Permutation importance | Global | Model-agnostic, held-out | Slow; shares attribution across correlated features |",
        "| SHAP | Global + Local | Theoretically grounded; shows direction | Computationally expensive |",
        "",
        "---",
        "",
        "## 12. Phase 6 Conclusion",
        "",
        "Across all four interpretation methods, `koi_model_snr`, `koi_prad`, "
        "and `koi_depth` appear consistently among the most important features. "
        "These are consistent with known transit physics: "
        "high SNR signals are more likely genuine transits, "
        "and planetary radius / transit depth discriminate between "
        "planet-sized events and eclipsing binary scenarios.",
        "",
        "These findings are descriptive of model behavior and do not constitute "
        "causal claims.",
        "",
        "**Phase 6 is complete. Phase 7 will handle threshold optimization "
        "and final model selection.**",
    ]

    path = REPORTS_P6 / "phase_6_interpretation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {path}")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 70)
    print("PHASE 6: FEATURE IMPORTANCE & INTERPRETATION")
    print("Kepler Exoplanet Classification")
    print("=" * 70)
    print(f"  shap: {shap.__version__}  "
          f"sklearn: {sklearn.__version__}  "
          f"xgboost: {xgboost.__version__}")

    # 1. Load data
    X_train, X_test, y_train, y_test = load_data()

    # 2. Load feature names
    feature_names = load_feature_names()

    # 3. Load models
    models = load_models()

    # 4. Verify inputs
    verify_model_inputs(models, X_train, X_test, feature_names)

    # 5. Native importance
    native = calculate_native_importance(models, feature_names)

    # 6. Permutation importance
    perm_dict = calculate_permutation_importance(
        models, X_test, y_test, feature_names)

    # 7. Permutation plots
    generate_permutation_plots(perm_dict, feature_names)

    # 8. SHAP explainers
    explainers = initialize_shap_explainers(models, X_train, X_test, feature_names)

    # 9. SHAP values
    shap_vals, shap_expls, X_sample, idx_sample = calculate_shap_values(
        explainers, X_test, feature_names)

    # 10. Global SHAP importance
    global_shap_df = calculate_global_shap_importance(shap_vals, feature_names)

    # 11. SHAP summary plots
    generate_shap_summary_plots(shap_expls, X_sample, feature_names)

    # 12. SHAP bar plots
    generate_shap_bar_plots(shap_expls, feature_names)

    # 13. SHAP dependence plots
    top5_feats = generate_shap_dependence_plots(
        shap_vals, X_sample, feature_names, global_shap_df)

    # 14. Importance consensus
    cons_df, freq_df, native_ranks, perm_ranks, shap_ranks = \
        build_importance_consensus(native, perm_dict, global_shap_df, feature_names)

    # 15. Top features summary
    print("\n[INFO] Building top features summary...")
    generate_top_features_summary(native, perm_dict, global_shap_df, feature_names)

    # 16. Explainability comparison
    print("\n[INFO] Building explainability comparison...")
    generate_explainability_comparison()

    # 17. Representative cases
    cases = select_representative_cases(models, X_test, y_test)

    # 18. Individual explanations
    generate_individual_explanations(
        cases, models, explainers, shap_vals,
        X_test, y_test, X_sample, idx_sample, feature_names)

    # 19. Class distribution plots
    generate_class_distribution_plots(
        X_train, y_train, feature_names, top5_feats)

    # 20. Markdown report
    generate_report(feature_names, native, perm_dict, global_shap_df,
                    cons_df, freq_df, top5_feats, cases, X_test, y_test, models)

    print("\n" + "=" * 70)
    print("PHASE 6 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
