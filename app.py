"""
app.py -- Kepler Exoplanet Classification
Explainable Machine Learning for KOI Candidate Verification

Streamlit web application that uses the Phase 7 finalized XGBoost model.
All inference is read-only -- no model training occurs here.
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths -- ALL relative so the app is portable
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent
MODEL_PATH  = BASE_DIR / "models" / "final" / "final_model.pkl"
META_PATH   = BASE_DIR / "models" / "final" / "final_model_metadata.json"
CAND_CSV    = BASE_DIR / "data" / "processed" / "X_candidate_processed.csv"
CAND_PRED   = BASE_DIR / "reports" / "phase_7" / "candidate_predictions.csv"
TEST_METRICS= BASE_DIR / "reports" / "phase_7" / "final_test_metrics.csv"
THR_JSON    = BASE_DIR / "reports" / "phase_7" / "final_threshold.json"
PLOTS_P7    = BASE_DIR / "plots" / "phase_7"
PLOTS_P6    = BASE_DIR / "plots" / "phase_6"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Kepler Exoplanet Classifier",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Cards */
.koi-card {
    background: #1A1E2E;
    border: 1px solid #2E3250;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 16px;
}
/* Prediction result boxes */
.pred-confirmed {
    background: linear-gradient(135deg, #0D3B2E, #1B5E20);
    border: 2px solid #43A047;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.pred-fp {
    background: linear-gradient(135deg, #3B0D0D, #5E1B1B);
    border: 2px solid #E53935;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
/* Disclaimer */
.disclaimer {
    background: #1A2030;
    border-left: 4px solid #FFA726;
    padding: 12px 16px;
    border-radius: 4px;
    font-size: 0.88em;
    color: #FFCC80;
    margin-bottom: 20px;
}
/* Metric label */
.metric-box {
    background: #1A1E2E;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
h1 { color: #4FC3F7; }
h2 { color: #81D4FA; border-bottom: 1px solid #2E3250; padding-bottom: 6px; }
h3 { color: #B3E5FC; }
.stSelectbox label { color: #B0BEC5; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REQUIRED_FEATURES = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_impact", "koi_model_snr", "koi_teq", "koi_insol",
    "koi_steff", "koi_slogg", "koi_srad",
]

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

TRANSIT_FEATURES  = ["koi_period", "koi_duration", "koi_depth",
                     "koi_impact", "koi_model_snr"]
PLANET_FEATURES   = ["koi_prad", "koi_teq", "koi_insol"]
STELLAR_FEATURES  = ["koi_steff", "koi_slogg", "koi_srad"]

# Reasonable defaults (median-ish values from labeled training set)
FEATURE_DEFAULTS = {
    "koi_period":    9.49,
    "koi_duration":  3.45,
    "koi_depth":     506.0,
    "koi_prad":      2.20,
    "koi_impact":    0.38,
    "koi_model_snr": 22.0,
    "koi_teq":       865.0,
    "koi_insol":     67.0,
    "koi_steff":     5440.0,
    "koi_slogg":     4.45,
    "koi_srad":      0.93,
}

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metadata():
    with open(META_PATH) as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_test_metrics():
    return pd.read_csv(TEST_METRICS)


@st.cache_data(show_spinner=False)
def load_candidate_data():
    X_cand = pd.read_csv(CAND_CSV)
    preds  = pd.read_csv(CAND_PRED)
    return X_cand, preds


@st.cache_data(show_spinner=False)
def load_threshold_info():
    with open(THR_JSON) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def predict_single(model, X_df: pd.DataFrame, threshold: float):
    """Return (label, probability) for a single-row DataFrame."""
    X_ordered = X_df[REQUIRED_FEATURES]
    proba = float(model.predict_proba(X_ordered)[0, 1])
    label = "CONFIRMED" if proba >= threshold else "FALSE POSITIVE"
    return label, proba


def predict_batch(model, X_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return predictions DataFrame preserving extra columns."""
    X_feat = X_df[REQUIRED_FEATURES]
    probas = model.predict_proba(X_feat)[:, 1]
    labels = ["CONFIRMED" if p >= threshold else "FALSE POSITIVE" for p in probas]
    result = X_df.copy()
    result["predicted_class"]       = labels
    result["predicted_probability"] = probas.round(4)
    result["threshold"]             = threshold
    return result


# ---------------------------------------------------------------------------
# Shared disclaimer
# ---------------------------------------------------------------------------
def show_disclaimer():
    st.markdown("""
<div class="disclaimer">
<strong>Important:</strong> This application provides machine-learning predictions
based on the trained Kepler KOI dataset. A prediction labeled <strong>CONFIRMED</strong>
does not constitute independent astronomical confirmation of an exoplanet.
All outputs are model predictions and should be treated as such.
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Probability bar plot
# ---------------------------------------------------------------------------
def plot_probability_bar(proba: float, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 1.4))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    ax.barh([0], [threshold], color="#E53935", alpha=0.7, height=0.5, label="FALSE POSITIVE zone")
    ax.barh([0], [1 - threshold], left=[threshold], color="#43A047",
            alpha=0.7, height=0.5, label="CONFIRMED zone")

    # Probability marker
    ax.axvline(proba, color="#FFFFFF", linewidth=2.5)
    ax.axvline(threshold, color="#FFA726", linewidth=1.8, linestyle="--")

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, threshold, proba, 1])
    ax.set_xticklabels(
        ["0%", f"Threshold\n{threshold*100:.0f}%",
         f"P={proba*100:.1f}%", "100%"],
        color="#B0BEC5", fontsize=8,
    )
    ax.tick_params(axis="x", colors="#B0BEC5")
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(threshold / 2, 0, "FALSE POSITIVE", ha="center", va="center",
            color="white", fontsize=9, fontweight="bold")
    ax.text((threshold + 1) / 2, 0, "CONFIRMED", ha="center", va="center",
            color="white", fontsize=9, fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ---------------------------------------------------------------------------
# PAGE: Single Prediction
# ---------------------------------------------------------------------------
def page_single_prediction(model, metadata):
    threshold = float(metadata["threshold"])

    st.title("🔭 Single KOI Prediction")
    show_disclaimer()
    st.markdown("Enter the preprocessed (scaled) feature values for a single "
                "Kepler Object of Interest to obtain a model prediction.")

    with st.form("single_pred_form"):
        col_t, col_p, col_s = st.columns(3)

        with col_t:
            st.subheader("Transit Properties")
            vals = {}
            for feat in TRANSIT_FEATURES:
                vals[feat] = st.number_input(
                    label=f"{feat}",
                    value=float(FEATURE_DEFAULTS[feat]),
                    format="%.4f",
                    help=FEATURE_DESCRIPTIONS[feat],
                    key=f"inp_{feat}",
                )

        with col_p:
            st.subheader("Planet Properties")
            for feat in PLANET_FEATURES:
                vals[feat] = st.number_input(
                    label=f"{feat}",
                    value=float(FEATURE_DEFAULTS[feat]),
                    format="%.4f",
                    help=FEATURE_DESCRIPTIONS[feat],
                    key=f"inp_{feat}",
                )

        with col_s:
            st.subheader("Stellar Properties")
            for feat in STELLAR_FEATURES:
                vals[feat] = st.number_input(
                    label=f"{feat}",
                    value=float(FEATURE_DEFAULTS[feat]),
                    format="%.4f",
                    help=FEATURE_DESCRIPTIONS[feat],
                    key=f"inp_{feat}",
                )

        submitted = st.form_submit_button("🚀 Predict KOI", use_container_width=True)

    if submitted:
        # Validate
        errors = []
        for feat in REQUIRED_FEATURES:
            v = vals[feat]
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                errors.append(f"{feat} is NaN or infinite")
        if errors:
            for e in errors:
                st.error(e)
            return

        X_input = pd.DataFrame([vals])[REQUIRED_FEATURES]
        label, proba = predict_single(model, X_input, threshold)

        st.markdown("---")
        st.subheader("Model Prediction")

        col_res, col_info = st.columns([1, 2])

        with col_res:
            if label == "CONFIRMED":
                st.markdown(f"""
<div class="pred-confirmed">
<h2 style="color:#69F0AE; margin:0">CONFIRMED</h2>
<p style="color:#B2DFDB; margin:4px 0 0 0">Model prediction</p>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="pred-fp">
<h2 style="color:#FF5252; margin:0">FALSE POSITIVE</h2>
<p style="color:#FFCDD2; margin:4px 0 0 0">Model prediction</p>
</div>""", unsafe_allow_html=True)

            st.metric("Predicted probability", f"{proba*100:.1f}%")
            st.metric("Classification threshold", f"{threshold*100:.0f}%")

        with col_info:
            st.markdown("**Probability bar**")
            plot_probability_bar(proba, threshold)

            if label == "CONFIRMED":
                st.info("The model assigns this KOI a probability **above** "
                        "the classification threshold for the CONFIRMED class.")
            else:
                st.info("The model assigns this KOI a probability **below** "
                        "the classification threshold for the CONFIRMED class.")

        st.markdown("**Submitted feature values**")
        summary_df = pd.DataFrame({
            "Feature":     REQUIRED_FEATURES,
            "Value":       [round(vals[f], 4) for f in REQUIRED_FEATURES],
            "Description": [FEATURE_DESCRIPTIONS[f] for f in REQUIRED_FEATURES],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Store in session state
        st.session_state["last_prediction"] = {
            "label": label, "proba": proba, "values": vals
        }


# ---------------------------------------------------------------------------
# PAGE: Batch Prediction
# ---------------------------------------------------------------------------
def page_batch_prediction(model, metadata):
    threshold = float(metadata["threshold"])

    st.title("📋 Batch KOI Prediction")
    show_disclaimer()
    st.markdown(
        "Upload a CSV file containing the 11 required features. "
        "Identifier columns (e.g. `kepid`, `kepoi_name`) are allowed and preserved."
    )

    st.info(f"Required columns: `{'`, `'.join(REQUIRED_FEATURES)}`")

    uploaded = st.file_uploader("Upload KOI CSV", type=["csv"])

    if uploaded is None:
        st.markdown("**Sample format:**")
        sample = pd.DataFrame(
            [[FEATURE_DEFAULTS[f] for f in REQUIRED_FEATURES]],
            columns=REQUIRED_FEATURES,
        )
        st.dataframe(sample, use_container_width=True, hide_index=True)
        return

    # Load
    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    if len(df) == 0:
        st.error("Uploaded file has no rows.")
        return

    # Validate
    missing_cols = [f for f in REQUIRED_FEATURES if f not in df.columns]
    if missing_cols:
        st.error(f"Missing required features: {missing_cols}")
        return

    # Check numerics / NaN / Inf
    problems = []
    for feat in REQUIRED_FEATURES:
        col = pd.to_numeric(df[feat], errors="coerce")
        n_nan = col.isnull().sum()
        n_inf = np.isinf(col.dropna()).sum()
        if n_nan > 0:
            problems.append(f"`{feat}`: {n_nan} non-numeric/NaN values")
        if n_inf > 0:
            problems.append(f"`{feat}`: {n_inf} infinite values")

    if problems:
        for p in problems:
            st.warning(p)
        st.error("Fix the above issues before predicting.")
        return

    # Predict
    with st.spinner("Running predictions..."):
        try:
            result_df = predict_batch(model, df, threshold)
        except Exception as exc:
            st.error(f"Prediction error: {exc}")
            return

    n_total = len(result_df)
    n_conf  = (result_df["predicted_class"] == "CONFIRMED").sum()
    n_fp    = (result_df["predicted_class"] == "FALSE POSITIVE").sum()

    st.success(f"Predictions complete for {n_total:,} rows.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total rows", f"{n_total:,}")
    c2.metric("Predicted CONFIRMED", f"{n_conf:,}")
    c3.metric("Predicted FALSE POSITIVE", f"{n_fp:,}")

    # Distribution chart
    col_chart, col_hist = st.columns(2)
    with col_chart:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor("#1A1E2E")
        ax.set_facecolor("#1A1E2E")
        ax.bar(["CONFIRMED", "FALSE POSITIVE"], [n_conf, n_fp],
               color=["#43A047", "#E53935"], edgecolor="#333", alpha=0.85)
        ax.set_ylabel("Count", color="#B0BEC5")
        ax.tick_params(colors="#B0BEC5")
        for spine in ax.spines.values():
            spine.set_color("#2E3250")
        ax.set_title("Prediction Distribution", color="#B3E5FC")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_hist:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor("#1A1E2E")
        ax.set_facecolor("#1A1E2E")
        probas = result_df["predicted_probability"].values
        ax.hist(probas, bins=30, color="#4FC3F7", edgecolor="#333", alpha=0.8)
        ax.axvline(threshold, color="#FFA726", linewidth=2, linestyle="--",
                   label=f"Threshold {threshold:.2f}")
        ax.set_xlabel("Predicted Probability", color="#B0BEC5")
        ax.set_ylabel("Count", color="#B0BEC5")
        ax.tick_params(colors="#B0BEC5")
        ax.legend(fontsize=8, facecolor="#1A1E2E", labelcolor="#B0BEC5")
        for spine in ax.spines.values():
            spine.set_color("#2E3250")
        ax.set_title("Probability Distribution", color="#B3E5FC")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Results table
    st.markdown("**Prediction results (first 100 rows)**")
    display_cols = (
        [c for c in df.columns if c not in REQUIRED_FEATURES] +
        ["predicted_class", "predicted_probability", "threshold"]
    )
    display_cols = [c for c in display_cols if c in result_df.columns]
    st.dataframe(result_df[display_cols].head(100),
                 use_container_width=True, hide_index=True)

    # Download
    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download full predictions CSV",
        data=csv_bytes,
        file_name="kepler_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# PAGE: Candidate Explorer
# ---------------------------------------------------------------------------
def page_candidate_explorer():
    st.title("🌌 Candidate Explorer")
    show_disclaimer()
    st.markdown(
        "These are the **1,977 CANDIDATE KOI rows** that were withheld from training. "
        "The predictions below come from the Phase 7 final model. "
        "**These are model predictions — not scientifically confirmed exoplanets.**"
    )

    try:
        X_cand, preds = load_candidate_data()
    except Exception as exc:
        st.error(f"Could not load candidate data: {exc}")
        return

    threshold = float(preds["threshold"].iloc[0]) if "threshold" in preds.columns else 0.61
    n_total   = len(preds)
    n_conf    = (preds["predicted_class"] == "CONFIRMED").sum()
    n_fp      = (preds["predicted_class"] == "FALSE_POSITIVE").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total candidates", f"{n_total:,}")
    c2.metric("Model-predicted CONFIRMED", f"{n_conf:,}")
    c3.metric("Model-predicted FALSE POSITIVE", f"{n_fp:,}")

    # Probability filter
    st.markdown("---")
    st.subheader("Filter by minimum predicted probability")
    min_prob = st.slider(
        "Minimum probability threshold",
        min_value=0.0, max_value=1.0, value=float(threshold),
        step=0.01, format="%.2f",
    )
    st.caption(
        f"Production threshold is **{threshold:.2f}**. "
        "This slider is for exploration only and does not change the model threshold."
    )

    filtered = preds[preds["predicted_probability"] >= min_prob].reset_index(drop=True)
    st.markdown(f"**{len(filtered):,} candidates** with probability ≥ {min_prob:.2f}")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    # Probability distribution
    st.markdown("---")
    st.subheader("Candidate probability distribution")

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#1A1E2E")
    ax.set_facecolor("#1A1E2E")
    ax.hist(preds["predicted_probability"], bins=50,
            color="#4FC3F7", edgecolor="#333", alpha=0.8, density=True)
    ax.axvline(threshold, color="#FFA726", linewidth=2.2, linestyle="--",
               label=f"Production threshold ({threshold:.2f})")
    ax.set_xlabel("Predicted Probability (CONFIRMED)", color="#B0BEC5")
    ax.set_ylabel("Density", color="#B0BEC5")
    ax.tick_params(colors="#B0BEC5")
    ax.legend(fontsize=9, facecolor="#1A1E2E", labelcolor="#B0BEC5")
    for spine in ax.spines.values():
        spine.set_color("#2E3250")
    ax.set_title("Probability Distribution: 1,977 Candidate KOIs",
                 color="#B3E5FC")
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ---------------------------------------------------------------------------
# PAGE: Model Information
# ---------------------------------------------------------------------------
def page_model_information(metadata):
    threshold = float(metadata["threshold"])

    st.title("📊 Model Information")
    show_disclaimer()

    # --- Final model summary ---
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Final Model Configuration")
        st.markdown(f"""
| Item | Value |
|------|-------|
| Model type | {metadata.get('model_type', 'XGBoost')} |
| Imbalance strategy | {metadata.get('imbalance_strategy', 'class_weight')} |
| scale_pos_weight | {metadata.get('scale_pos_weight', 1.7611)} |
| Classification threshold | **{threshold:.2f}** |
| Feature count | {metadata.get('feature_count', 11)} |
| Primary metric | {metadata.get('primary_metric', 'PR-AUC')} |
""")

        st.markdown("**Hyperparameters (Phase 4 tuned)**")
        hparams = metadata.get("hyperparameters", {})
        hp_df = pd.DataFrame(
            list(hparams.items()), columns=["Parameter", "Value"])
        st.dataframe(hp_df, use_container_width=True, hide_index=True)

    with col_b:
        st.subheader("Threshold Information")
        thr_info = load_threshold_info()
        st.markdown(f"""
| Item | Value |
|------|-------|
| Frozen threshold | **{thr_info.get('threshold', threshold):.2f}** |
| Selection method | {thr_info.get('selection_method', '')} |
| CV folds | {thr_info.get('n_folds', 5)} |
| random_state | {thr_info.get('random_state', 42)} |
| Recall constraint | {thr_info.get('recall_constraint', 0.90)} |
| OOF F1 | {thr_info.get('oof_f1', '')} |
| OOF Recall | {thr_info.get('oof_recall', '')} |
| Constraint met | {thr_info.get('recall_constraint_met', True)} |
""")
        st.info(
            "The threshold was selected using 5-fold out-of-fold training predictions, "
            "maximizing F1 subject to recall ≥ 0.90. "
            "It is fixed and cannot be changed here."
        )

    # --- Test metrics ---
    st.markdown("---")
    st.subheader("Final Test-Set Performance")
    st.caption(
        "These metrics were computed on the 20% held-out test set, accessed exactly once "
        "after the threshold was frozen."
    )

    try:
        metrics_df = load_test_metrics()
        row = metrics_df.iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",  f"{float(row.get('accuracy', 0))*100:.2f}%")
        m2.metric("Precision", f"{float(row.get('precision', 0))*100:.2f}%")
        m3.metric("Recall",    f"{float(row.get('recall', 0))*100:.2f}%")
        m4.metric("F1",        f"{float(row.get('f1', 0))*100:.2f}%")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("ROC-AUC",        f"{float(row.get('roc_auc', 0)):.4f}")
        m6.metric("PR-AUC",         f"{float(row.get('pr_auc', 0)):.4f}")
        m7.metric("Balanced Acc.",  f"{float(row.get('balanced_accuracy', 0))*100:.2f}%")
        m8.metric("Brier Score",    f"{float(row.get('brier_score', row.get('brier', 0))):.4f}"
                  if 'brier_score' in metrics_df.columns or 'brier' in metrics_df.columns
                  else "0.0461")

        # Confusion matrix numbers
        st.markdown("**Confusion matrix**")
        tn = int(row.get("true_negative", 920))
        fp = int(row.get("false_positive", 48))
        fn = int(row.get("false_negative", 50))
        tp = int(row.get("true_positive", 500))
        cm_df = pd.DataFrame(
            {"Predicted FP": [tn, fn], "Predicted CONF": [fp, tp]},
            index=["Actual FP", "Actual CONF"],
        )
        st.table(cm_df)
    except Exception as exc:
        st.warning(f"Could not load test metrics: {exc}")

    # --- Plots ---
    st.markdown("---")
    col_cm, col_roc, col_pr = st.columns(3)

    cm_png = PLOTS_P7 / "final_confusion_matrix.png"
    roc_png = PLOTS_P7 / "final_roc_curve.png"
    pr_png  = PLOTS_P7 / "final_pr_curve.png"

    with col_cm:
        st.subheader("Confusion Matrix")
        if cm_png.exists():
            st.image(str(cm_png), use_container_width=True)
    with col_roc:
        st.subheader("ROC Curve")
        if roc_png.exists():
            st.image(str(roc_png), use_container_width=True)
        st.caption("ROC-AUC measures discrimination across all thresholds.")
    with col_pr:
        st.subheader("Precision-Recall Curve")
        if pr_png.exists():
            st.image(str(pr_png), use_container_width=True)
        st.caption("PR-AUC is the primary metric (sensitive to class imbalance).")

    # --- Calibration ---
    st.markdown("---")
    st.subheader("Probability Calibration")
    cal_png = PLOTS_P7 / "calibration_curve.png"
    if cal_png.exists():
        col_cal, col_cal_txt = st.columns([1, 1])
        with col_cal:
            st.image(str(cal_png), use_container_width=True)
        with col_cal_txt:
            st.markdown("""
**Brier Score: 0.0461**

The calibration curve shows how well the predicted probabilities
correspond to observed outcomes. A perfectly calibrated model would
fall on the diagonal. Slight under/over-confidence is normal and
has not been corrected in this pipeline.
""")


# ---------------------------------------------------------------------------
# PAGE: Explainability
# ---------------------------------------------------------------------------
def page_explainability():
    st.title("🧠 Explainability & Feature Importance")
    show_disclaimer()

    st.markdown("""
The models were interpreted in **Phase 6** using three complementary methods:
native feature importance, permutation importance, and SHAP.

**Key finding:** `koi_prad` (estimated planetary radius) was the top feature
across all four models and all three methods. `koi_model_snr`, `koi_period`,
`koi_duration`, and `koi_insol` were also consistently important.

These describe **model behavior** — not physical causation.
The model relied on these features; this does not mean they cause a planet to exist.
""")

    # Feature importance table
    st.subheader("Features and Their Role")
    imp_data = {
        "koi_prad":      ("Top feature across all methods",       "Very high"),
        "koi_model_snr": ("High SNR signals are more likely genuine", "High"),
        "koi_period":    ("Orbital period distribution differs between classes", "High"),
        "koi_duration":  ("Transit duration reflects orbital geometry", "High"),
        "koi_insol":     ("Stellar flux differs between confirmed and FP", "High"),
        "koi_depth":     ("Deeper transits can indicate stellar eclipses", "Medium"),
        "koi_impact":    ("High impact parameters linked to FP scenarios", "Medium"),
        "koi_teq":       ("Derived from period and stellar properties", "Medium"),
        "koi_slogg":     ("Stellar surface gravity affects transit interpretation", "Medium"),
        "koi_srad":      ("Stellar radius propagates to planetary radius estimate", "Medium"),
        "koi_steff":     ("Stellar temperature affects insolation estimates", "Lower"),
    }
    feat_df = pd.DataFrame(
        [(f, FEATURE_DESCRIPTIONS[f], imp_data[f][0], imp_data[f][1])
         for f in REQUIRED_FEATURES],
        columns=["Feature", "Description", "Phase 6 Note", "Relative Importance"],
    )
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # SHAP plots
    st.markdown("---")
    st.subheader("SHAP Analysis")

    model_sel = st.selectbox(
        "Select model for SHAP visualization",
        ["XGBoost", "Random Forest", "Decision Tree", "Logistic Regression"],
    )
    slug_map = {
        "XGBoost":             "xgboost",
        "Random Forest":       "random_forest",
        "Decision Tree":       "decision_tree",
        "Logistic Regression": "logistic_regression",
    }
    slug = slug_map[model_sel]

    col_sum, col_bar = st.columns(2)
    with col_sum:
        st.markdown(f"**SHAP Summary ({model_sel})**")
        summ = PLOTS_P6 / f"shap_summary_{slug}.png"
        if summ.exists():
            st.image(str(summ), use_container_width=True)
        else:
            st.info("SHAP summary plot not found.")
    with col_bar:
        st.markdown(f"**SHAP Global Importance ({model_sel})**")
        bar = PLOTS_P6 / f"shap_bar_{slug}.png"
        if bar.exists():
            st.image(str(bar), use_container_width=True)
        else:
            st.info("SHAP bar plot not found.")

    # Dependence plots
    st.markdown("---")
    st.subheader("SHAP Dependence Plots (top features)")
    dep_feats = ["koi_prad", "koi_model_snr", "koi_period", "koi_duration", "koi_insol"]
    dep_sel   = st.selectbox("Select feature", dep_feats)
    dep_png   = PLOTS_P6 / "dependence" / f"shap_dependence_{dep_sel}.png"
    if dep_png.exists():
        st.image(str(dep_png), use_container_width=True)
        st.caption(
            f"Each point is a test sample. The SHAP value shows how "
            f"`{dep_sel}` pushed the model toward CONFIRMED (positive) "
            f"or FALSE POSITIVE (negative) for that sample."
        )
    else:
        st.info(f"Dependence plot not found for {dep_sel}.")


# ---------------------------------------------------------------------------
# PAGE: About
# ---------------------------------------------------------------------------
def page_about():
    st.title("ℹ️ About This Project")
    show_disclaimer()

    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.subheader("Project Overview")
        st.markdown("""
The **Kepler Exoplanet Classification** project uses machine learning to classify
NASA Kepler Objects of Interest (KOIs) into **CONFIRMED** and **FALSE POSITIVE**
categories. It is an 8-phase data science pipeline built for educational and
research purposes.

**Classification target:**
- `CONFIRMED` → target = 1
- `FALSE POSITIVE` → target = 0
- `CANDIDATE` → no label (predictions only)
""")

        st.subheader("ML Pipeline Phases")
        phases = [
            ("Phase 1", "Data Acquisition & EDA"),
            ("Phase 2", "Preprocessing & Feature Engineering"),
            ("Phase 3", "Baseline Modeling (LR, DT, RF, XGBoost)"),
            ("Phase 4", "Hyperparameter Tuning (CV, PR-AUC primary)"),
            ("Phase 5", "Class Imbalance Handling (class weight, SMOTE)"),
            ("Phase 6", "Feature Importance & SHAP Interpretation"),
            ("Phase 7", "Model Selection & Threshold Optimization"),
            ("Phase 8", "Streamlit Deployment"),
        ]
        for p, d in phases:
            st.markdown(f"**{p}:** {d}")

    with col_b:
        st.subheader("Dataset Statistics")
        st.markdown("""
| Item | Value |
|------|-------|
| Source | NASA Exoplanet Archive |
| Total KOIs | 9,564 |
| Labeled rows | 7,587 |
| CONFIRMED | 2,748 (36.2%) |
| FALSE POSITIVE | 4,839 (63.8%) |
| CANDIDATE (held out) | 1,977 |
| Training set | 6,069 (80%) |
| Test set | 1,518 (20%) |
| Features used | 11 |
""")

        st.subheader("Final Model")
        st.markdown("""
| Item | Value |
|------|-------|
| Model | XGBoost |
| Imbalance | scale_pos_weight = 1.7611 |
| Threshold | 0.61 |
| Test PR-AUC | 0.9661 |
| Test ROC-AUC | 0.9824 |
| Test F1 | 0.9107 |
| Test Recall | 0.9091 |
""")

    st.markdown("---")
    st.subheader("Limitations")
    st.markdown("""
1. Predictions are **not** independent astronomical confirmation.
2. The model uses only 11 selected KOI features.
3. Performance on future or differently distributed data may differ.
4. Probability outputs may not be perfectly calibrated.
5. CANDIDATE predictions are preliminary and require expert review.
6. The classification threshold (0.61) reflects the training objective; 
   different scientific applications may require different thresholds.
""")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    # Load model and metadata at startup (cached)
    model    = load_model()
    metadata = load_metadata()

    # Sidebar navigation
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/"
        "Kepler_Space_Telescope.jpg/220px-Kepler_Space_Telescope.jpg",
        use_container_width=True,
    )
    st.sidebar.markdown("## 🔭 Kepler Classifier")
    st.sidebar.markdown(
        "Explainable ML for KOI Candidate Verification"
    )
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["Single Prediction", "Batch Prediction",
         "Candidate Explorer", "Model Information",
         "Explainability", "About"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Threshold:** `{metadata.get('threshold', 0.61):.2f}`\n\n"
        f"**Model:** {metadata.get('model_type', 'XGBoost')}\n\n"
        f"**Strategy:** {metadata.get('imbalance_strategy', 'class_weight')}"
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "This application provides ML predictions only. "
        "Not a scientific confirmation instrument."
    )

    # Route
    if page == "Single Prediction":
        page_single_prediction(model, metadata)
    elif page == "Batch Prediction":
        page_batch_prediction(model, metadata)
    elif page == "Candidate Explorer":
        page_candidate_explorer()
    elif page == "Model Information":
        page_model_information(metadata)
    elif page == "Explainability":
        page_explainability()
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()
