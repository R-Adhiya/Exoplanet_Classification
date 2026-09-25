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
BASE_DIR         = Path(__file__).resolve().parent
MODEL_PATH       = BASE_DIR / "models" / "final" / "final_model.pkl"
META_PATH        = BASE_DIR / "models" / "final" / "final_model_metadata.json"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"
CAND_CSV         = BASE_DIR / "data" / "processed" / "X_candidate_processed.csv"
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


@st.cache_resource(show_spinner=False)
def load_preprocessor():
    """Load the Phase 2 fitted preprocessing pipeline (imputer + log + scaler)."""
    if not PREPROCESSOR_PATH.exists():
        st.error(f"Preprocessor not found: {PREPROCESSOR_PATH}")
        st.stop()
    return joblib.load(PREPROCESSOR_PATH)


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
def _get_booster_feature_order(model) -> list:
    """Return the exact feature name list the XGBoost booster expects."""
    return model.get_booster().feature_names


def predict_single(model, preprocessor, X_raw: pd.DataFrame, threshold: float):
    """
    Preprocess raw UI inputs then predict.

    Steps:
      1. Run X_raw through the Phase 2 preprocessor (imputation + log + scaling).
      2. Reconstruct a DataFrame with the preprocessor's output column names.
      3. Reorder columns to match model.get_booster().feature_names exactly.
      4. Call model.predict_proba on the correctly ordered array.

    X_raw must contain exactly the 11 raw-scale feature columns.
    """
    booster_cols = _get_booster_feature_order(model)

    # Step 1 -- preprocess (returns a numpy array)
    X_proc_arr = preprocessor.transform(X_raw[REQUIRED_FEATURES])

    # Step 2 -- rebuild as DataFrame with preprocessor output column names
    # The ColumnTransformer outputs log-features first, then plain-features
    # (matching booster_cols order). We name them using booster_cols directly.
    X_proc = pd.DataFrame(X_proc_arr, columns=booster_cols)

    # Step 3 -- explicit reorder guard (no-op if already correct, safety net otherwise)
    X_final = X_proc[booster_cols]

    # Step 4 -- predict
    proba = float(model.predict_proba(X_final)[0, 1])
    label = "CONFIRMED" if proba >= threshold else "FALSE POSITIVE"
    return label, proba


def predict_batch(model, preprocessor, X_raw: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Preprocess batch CSV inputs then predict.
    Preserves any extra identifier columns from the original DataFrame.
    """
    booster_cols = _get_booster_feature_order(model)

    # Preprocess feature columns only
    X_proc_arr = preprocessor.transform(X_raw[REQUIRED_FEATURES])
    X_proc     = pd.DataFrame(X_proc_arr, columns=booster_cols,
                               index=X_raw.index)
    X_final    = X_proc[booster_cols]

    probas = model.predict_proba(X_final)[:, 1]
    labels = ["CONFIRMED" if p >= threshold else "FALSE POSITIVE" for p in probas]

    result = X_raw.copy()
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
    preprocessor = load_preprocessor()

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
        label, proba = predict_single(model, preprocessor, X_input, threshold)

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
    preprocessor = load_preprocessor()

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
            result_df = predict_batch(model, preprocessor, df, threshold)
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
    # ── Session-state page routing ──────────────────────────────────────────
    if "app_page" not in st.session_state:
        st.session_state.app_page = "landing"

    # Landing page -- full-bleed, no sidebar
    if st.session_state.app_page == "landing":
        _render_landing()
        return

    # ── Classifier pages -- restore sidebar ─────────────────────────────────
    model    = load_model()
    metadata = load_metadata()

    # Back-to-home button at very top
    st.markdown("""
<style>
.back-btn button {
    background: transparent !important;
    border: 1px solid rgba(103,232,249,0.3) !important;
    color: #67e8f9 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    padding: 6px 16px !important;
}
.back-btn button:hover {
    border-color: #67e8f9 !important;
    background: rgba(103,232,249,0.08) !important;
}
</style>""", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("← Back to Home"):
            st.session_state.app_page = "landing"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.sidebar.markdown("## 🔭 Kepler Classifier")
    st.sidebar.markdown("Explainable ML for KOI Candidate Verification")
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


# ---------------------------------------------------------------------------
# LANDING PAGE RENDERER
# ---------------------------------------------------------------------------
def _render_landing():
    """
    Full-screen cinematic space-themed landing page.
    Navigation is handled entirely by Streamlit session state --
    no fragile JS required. CTA buttons call st.rerun() after
    setting st.session_state.app_page = 'classifier'.
    """
    # Hide Streamlit chrome for full-bleed effect
    st.markdown("""
<style>
[data-testid="stAppViewContainer"] > .main > div { padding: 0 !important; max-width: 100% !important; }
[data-testid="stAppViewContainer"] { padding: 0 !important; }
section[data-testid="stSidebar"]   { display: none !important; }
header[data-testid="stHeader"]     { display: none !important; }
#MainMenu, footer                  { display: none !important; }
/* Remove top padding Streamlit adds before stMarkdownContainer */
.block-container { padding: 0 !important; max-width: 100% !important; }
</style>""", unsafe_allow_html=True)

    # ── FULL HTML/CSS LANDING ──────────────────────────────────────────────
    st.markdown("""
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
/* ===== RESET ===== */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
/* ===== BASE ===== */
.lp{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;
    background:#030712;color:#e2e8f0;overflow-x:hidden;line-height:1.6}
/* ===== STARFIELD ===== */
.lp-stars{position:fixed;top:0;left:0;width:100%;height:100%;
  background:#030712;overflow:hidden;pointer-events:none;z-index:0}
.lp-stars::before,.lp-stars::after{content:'';position:absolute;
  width:100%;height:100%;background-image:
    radial-gradient(1px 1px at 10% 15%,rgba(255,255,255,.7) 0%,transparent 100%),
    radial-gradient(1px 1px at 25% 35%,rgba(255,255,255,.5) 0%,transparent 100%),
    radial-gradient(1.5px 1.5px at 40% 8%,rgba(165,243,252,.8) 0%,transparent 100%),
    radial-gradient(1px 1px at 55% 45%,rgba(255,255,255,.6) 0%,transparent 100%),
    radial-gradient(1px 1px at 70% 20%,rgba(255,255,255,.4) 0%,transparent 100%),
    radial-gradient(2px 2px at 82% 60%,rgba(165,243,252,.9) 0%,transparent 100%),
    radial-gradient(1px 1px at 15% 70%,rgba(255,255,255,.5) 0%,transparent 100%),
    radial-gradient(1px 1px at 90% 40%,rgba(255,255,255,.6) 0%,transparent 100%),
    radial-gradient(1.5px 1.5px at 5% 90%,rgba(255,255,255,.5) 0%,transparent 100%),
    radial-gradient(1px 1px at 65% 75%,rgba(255,255,255,.4) 0%,transparent 100%),
    radial-gradient(1px 1px at 48% 55%,rgba(255,255,255,.6) 0%,transparent 100%),
    radial-gradient(1px 1px at 32% 80%,rgba(165,243,252,.5) 0%,transparent 100%),
    radial-gradient(1px 1px at 78% 85%,rgba(255,255,255,.4) 0%,transparent 100%),
    radial-gradient(2px 2px at 20% 50%,rgba(165,243,252,.7) 0%,transparent 100%),
    radial-gradient(1px 1px at 95% 12%,rgba(255,255,255,.5) 0%,transparent 100%),
    radial-gradient(1px 1px at 60% 92%,rgba(255,255,255,.4) 0%,transparent 100%),
    radial-gradient(1px 1px at 38% 25%,rgba(255,255,255,.6) 0%,transparent 100%),
    radial-gradient(1.5px 1.5px at 88% 75%,rgba(165,243,252,.8) 0%,transparent 100%);
  animation:twinkle 4s ease-in-out infinite alternate}
.lp-stars::after{animation-delay:2s;opacity:.6}
@keyframes twinkle{from{opacity:.6}to{opacity:1}}
/* nebula blobs */
.lp-nebula{position:fixed;top:0;left:0;width:100%;height:100%;
  pointer-events:none;z-index:0}
.lp-nebula::before{content:'';position:absolute;
  width:70%;height:70%;top:-10%;left:-20%;
  background:radial-gradient(ellipse,rgba(99,102,241,.06) 0%,rgba(59,130,246,.04) 40%,transparent 70%);
  animation:nebula-drift 20s ease-in-out infinite alternate}
.lp-nebula::after{content:'';position:absolute;
  width:60%;height:60%;bottom:-5%;right:-10%;
  background:radial-gradient(ellipse,rgba(139,92,246,.06) 0%,rgba(103,232,249,.03) 40%,transparent 70%);
  animation:nebula-drift 25s ease-in-out infinite alternate-reverse}
@keyframes nebula-drift{from{transform:translate(0,0) scale(1)}
  to{transform:translate(20px,10px) scale(1.05)}}
/* ===== WRAPPER ===== */
.lp-wrap{position:relative;z-index:1}
/* ===== NAVBAR ===== */
.lp-nav{position:sticky;top:0;display:flex;align-items:center;
  justify-content:space-between;padding:18px 5vw;
  background:rgba(3,7,18,.65);backdrop-filter:blur(18px);
  border-bottom:1px solid rgba(103,232,249,.07);z-index:50}
.lp-brand{display:flex;align-items:center;gap:10px;
  font-size:.95rem;font-weight:700;letter-spacing:.14em;color:#f0f9ff;
  text-transform:uppercase}
.lp-orbit{width:26px;height:26px;border:2px solid #67e8f9;border-radius:50%;
  position:relative;box-shadow:0 0 10px rgba(103,232,249,.4)}
.lp-orbit::before{content:'';position:absolute;width:7px;height:7px;
  background:#67e8f9;border-radius:50%;top:50%;left:50%;
  transform:translateX(-50%) translateY(-50%);
  box-shadow:0 0 6px #67e8f9;
  animation:orbit-spin 3s linear infinite;transform-origin:3px -9px}
@keyframes orbit-spin{from{transform:rotate(0deg) translate(0,-12px)}
  to{transform:rotate(360deg) translate(0,-12px)}}
.lp-nav-links{display:flex;gap:28px;list-style:none}
.lp-nav-links a{text-decoration:none;color:#64748b;font-size:.72rem;
  font-weight:500;letter-spacing:.1em;text-transform:uppercase;
  transition:color .25s}
.lp-nav-links a:hover{color:#67e8f9}
/* ===== HERO ===== */
.lp-hero{min-height:100vh;display:flex;align-items:center;
  padding:80px 5vw 60px;overflow:hidden}
.lp-hero-grid{display:grid;grid-template-columns:1fr 1fr;
  gap:40px;align-items:center;width:100%;max-width:1400px;margin:0 auto}
.lp-eyebrow{font-family:'Courier New',monospace;font-size:.67rem;
  letter-spacing:.25em;color:#67e8f9;text-transform:uppercase;margin-bottom:18px}
.lp-eyebrow::before{content:'// ';opacity:.4}
.lp-h1{font-size:clamp(3rem,6vw,5.8rem);font-weight:800;line-height:1;
  letter-spacing:-.02em;color:#f0f9ff;margin-bottom:18px}
.lp-h1 .cy{display:block;color:#67e8f9;
  text-shadow:0 0 30px rgba(103,232,249,.45),0 0 60px rgba(103,232,249,.18)}
.lp-sub{font-size:.88rem;font-weight:500;letter-spacing:.08em;
  text-transform:uppercase;color:#475569;margin-bottom:16px}
.lp-desc{font-size:.97rem;line-height:1.8;color:#64748b;
  max-width:480px;margin-bottom:32px}
.lp-btns{display:flex;align-items:center;gap:18px;flex-wrap:wrap;
  margin-bottom:44px}
/* Note: actual CTA buttons are rendered as st.button() below */
.lp-stats{display:flex;gap:0;flex-wrap:wrap}
.lp-stat{padding:12px 24px;border-right:1px solid rgba(103,232,249,.07)}
.lp-stat:first-child{padding-left:0}
.lp-stat:last-child{border-right:none}
.lp-stat-n{font-size:1.5rem;font-weight:800;color:#e2e8f0;letter-spacing:-.01em}
.lp-stat-l{font-family:'Courier New',monospace;font-size:.6rem;
  letter-spacing:.18em;color:#334155;text-transform:uppercase}
/* ===== PLANET VISUAL ===== */
.lp-planet-scene{position:relative;display:flex;align-items:center;
  justify-content:center;height:520px}
.lp-aura{position:absolute;width:430px;height:430px;border-radius:50%;
  background:radial-gradient(ellipse,rgba(103,232,249,.04) 0%,
    rgba(99,102,241,.05) 40%,transparent 70%);
  animation:aura 5s ease-in-out infinite}
@keyframes aura{0%,100%{transform:scale(1);opacity:.7}50%{transform:scale(1.06);opacity:1}}
.lp-ring{position:absolute;border-radius:50%;border:1px solid rgba(103,232,249,.1)}
.lp-ring-1{width:390px;height:390px;animation:ring-r 28s linear infinite;
  transform:rotateX(72deg)}
.lp-ring-2{width:320px;height:320px;border-style:dashed;
  border-color:rgba(139,92,246,.08);animation:ring-r 18s linear infinite reverse;
  transform:rotateX(72deg)}
@keyframes ring-r{from{transform:rotateX(72deg) rotateZ(0)}to{transform:rotateX(72deg) rotateZ(360deg)}}
.lp-planet{position:relative;width:240px;height:240px;border-radius:50%;
  background:radial-gradient(circle at 33% 28%,#1a3558 0%,#0a1728 35%,#060b16 65%,#030508 100%);
  box-shadow:-20px -15px 35px rgba(103,232,249,.07),0 0 0 1.5px rgba(103,232,249,.05),
    0 0 55px rgba(99,102,241,.12),inset 10px 8px 35px rgba(103,232,249,.04);
  animation:pfloat 7s ease-in-out infinite;z-index:2;overflow:hidden}
@keyframes pfloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-13px)}}
.lp-atm{position:absolute;inset:-3px;border-radius:50%;
  background:radial-gradient(circle at 28% 24%,rgba(103,232,249,.15) 0%,transparent 55%);
  pointer-events:none}
.lp-scan{position:absolute;inset:0;border-radius:50%;overflow:hidden;pointer-events:none;z-index:3}
.lp-scan-l{position:absolute;width:100%;height:1.5px;
  background:linear-gradient(90deg,transparent,rgba(103,232,249,.3),transparent);
  animation:scan 4s linear infinite;top:0}
@keyframes scan{from{top:0;opacity:1}to{top:100%;opacity:0}}
.lp-moon{position:absolute;width:32px;height:32px;border-radius:50%;
  background:radial-gradient(circle at 33% 28%,#243350,#0e1622);
  box-shadow:-3px -2px 7px rgba(103,232,249,.1);
  animation:moon-o 13s linear infinite;top:50%;left:50%}
@keyframes moon-o{from{transform:rotate(0deg) translate(130px) rotate(0deg)}
  to{transform:rotate(360deg) translate(130px) rotate(-360deg)}}
/* HUD cards */
.lp-hud{position:absolute;z-index:5;font-family:'Courier New',monospace;
  font-size:.58rem;letter-spacing:.12em;text-transform:uppercase;
  background:rgba(3,7,18,.72);backdrop-filter:blur(8px);
  border:1px solid rgba(103,232,249,.12);border-radius:5px;
  padding:7px 11px;color:#67e8f9;white-space:nowrap;
  animation:hfloat 5s ease-in-out infinite}
@keyframes hfloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
.lp-hud::before,.lp-hud::after{content:'';position:absolute;
  width:6px;height:6px;border-color:rgba(103,232,249,.4);border-style:solid}
.lp-hud::before{top:-1px;left:-1px;border-width:1px 0 0 1px}
.lp-hud::after{bottom:-1px;right:-1px;border-width:0 1px 1px 0}
.lp-dot{display:inline-block;width:5px;height:5px;border-radius:50%;
  background:#67e8f9;box-shadow:0 0 5px #67e8f9;margin-right:5px;
  animation:blink 2s step-start infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.15}}
.hud-lbl{color:#334155;display:block;font-size:.52rem}
.hud-val{font-weight:700}
.h1{top:12%;left:0%;animation-delay:0s}
.h2{top:38%;left:-2%;animation-delay:1.4s}
.h3{bottom:16%;left:3%;animation-delay:.7s}
.h4{top:16%;right:0%;animation-delay:2s}
.h5{bottom:18%;right:2%;animation-delay:1.1s}
/* ===== SECTIONS ===== */
.lp-sect{position:relative;z-index:1;padding:100px 5vw;border-top:1px solid rgba(103,232,249,.05)}
.lp-sect-inner{max-width:1180px;margin:0 auto}
.lp-lbl{font-family:'Courier New',monospace;font-size:.62rem;
  letter-spacing:.28em;color:#67e8f9;text-transform:uppercase;margin-bottom:18px}
.lp-lbl::before{content:'// ';opacity:.4}
.lp-h2{font-size:clamp(2.2rem,4vw,3.6rem);font-weight:800;
  line-height:1.05;letter-spacing:-.02em;color:#f0f9ff;margin-bottom:24px}
.lp-txt{font-size:.95rem;line-height:1.8;color:#475569;max-width:520px}
/* Mission grid */
.lp-mission-g{display:grid;grid-template-columns:1fr 1fr;gap:72px;align-items:start}
.lp-tl{display:flex;flex-direction:column;gap:0;margin-top:8px}
.lp-tl-step{display:flex;gap:18px;align-items:flex-start}
.lp-tl-l{display:flex;flex-direction:column;align-items:center;min-width:30px}
.lp-tld{width:9px;height:9px;border-radius:50%;background:#67e8f9;
  box-shadow:0 0 8px #67e8f9;flex-shrink:0;margin-top:4px}
.lp-tll{width:1px;flex:1;min-height:30px;
  background:linear-gradient(to bottom,rgba(103,232,249,.25),rgba(103,232,249,.03))}
.lp-tc{padding-bottom:26px}
.lp-tt{font-family:'Courier New',monospace;font-size:.7rem;
  letter-spacing:.16em;color:#67e8f9;text-transform:uppercase;font-weight:700}
.lp-ts{font-size:.8rem;color:#334155;margin-top:3px}
/* Problem cards */
.lp-prob-g{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:48px}
.lp-pcard{background:rgba(5,11,24,.8);border:1px solid rgba(103,232,249,.07);
  border-radius:14px;padding:36px;position:relative;overflow:hidden;
  transition:border-color .3s,transform .3s}
.lp-pcard:hover{border-color:rgba(103,232,249,.18);transform:translateY(-4px)}
.lp-pcard::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(103,232,249,.15),transparent)}
.lp-ctype{font-family:'Courier New',monospace;font-size:.58rem;
  letter-spacing:.22em;text-transform:uppercase;margin-bottom:10px}
.cconf{color:#34d399}.cfp{color:#f87171}
.lp-ctitle{font-size:1.4rem;font-weight:700;color:#e2e8f0;margin-bottom:10px}
.lp-cdesc{font-size:.84rem;color:#334155;line-height:1.7}
.lp-cbadge{display:inline-block;margin-top:18px;padding:4px 10px;
  border-radius:3px;font-family:'Courier New',monospace;font-size:.58rem;
  letter-spacing:.13em;text-transform:uppercase}
.bdgc{background:rgba(52,211,153,.07);border:1px solid rgba(52,211,153,.18);color:#34d399}
.bdgf{background:rgba(248,113,113,.07);border:1px solid rgba(248,113,113,.18);color:#f87171}
/* Process steps */
.lp-proc-g{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin-top:48px}
.lp-pstep{background:rgba(5,11,24,.7);border:1px solid rgba(103,232,249,.07);
  border-radius:10px;padding:28px 22px;transition:border-color .3s,transform .3s}
.lp-pstep:hover{border-color:rgba(103,232,249,.16);transform:translateY(-3px)}
.lp-pnum{font-family:'Courier New',monospace;font-size:2.2rem;font-weight:800;
  color:rgba(103,232,249,.09);line-height:1;margin-bottom:14px}
.lp-ptitle{font-size:.76rem;font-weight:700;letter-spacing:.11em;
  text-transform:uppercase;color:#67e8f9;margin-bottom:8px}
.lp-pdesc{font-size:.8rem;color:#334155;line-height:1.65}
/* Feature cards */
.lp-feat-g{display:grid;grid-template-columns:repeat(2,1fr);gap:18px;margin-top:48px}
.lp-fcard{background:rgba(5,11,24,.6);border:1px solid rgba(103,232,249,.07);
  border-radius:10px;padding:28px;display:flex;gap:20px;
  transition:border-color .3s,background .3s}
.lp-fcard:hover{border-color:rgba(103,232,249,.15);background:rgba(5,11,24,.85)}
.lp-ficon{font-size:1.7rem;flex-shrink:0;margin-top:2px}
.lp-fnum{font-family:'Courier New',monospace;font-size:.58rem;
  letter-spacing:.18em;color:rgba(103,232,249,.35);margin-bottom:5px}
.lp-ftitle{font-size:.85rem;font-weight:700;letter-spacing:.08em;
  text-transform:uppercase;color:#cbd5e1;margin-bottom:7px}
.lp-fdesc{font-size:.8rem;color:#334155;line-height:1.65}
/* CTA section */
.lp-cta-sect{position:relative;z-index:1;padding:130px 5vw;
  text-align:center;border-top:1px solid rgba(103,232,249,.05)}
.lp-cta-sect::before{content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse at 50% 50%,rgba(99,102,241,.07) 0%,transparent 65%);
  pointer-events:none}
.lp-cta-inner{position:relative;max-width:650px;margin:0 auto}
.lp-cta-h{font-size:clamp(2.5rem,5vw,4.2rem);font-weight:800;
  line-height:1.06;letter-spacing:-.02em;color:#f0f9ff;margin-bottom:20px}
.lp-cta-d{font-size:.95rem;color:#475569;line-height:1.8;margin-bottom:36px}
/* Footer */
.lp-foot{position:relative;z-index:1;border-top:1px solid rgba(103,232,249,.05);
  padding:44px 5vw}
.lp-foot-g{max-width:1180px;margin:0 auto;
  display:grid;grid-template-columns:1fr 1fr;gap:40px;align-items:start}
.lp-fbrand{display:flex;align-items:center;gap:9px;
  font-size:.85rem;font-weight:700;letter-spacing:.12em;
  color:#e2e8f0;text-transform:uppercase;margin-bottom:10px}
.lp-ftag{font-size:.75rem;color:#1e293b;line-height:1.6}
.lp-flinks{display:flex;gap:24px;flex-wrap:wrap;justify-content:flex-end}
.lp-flinks a{text-decoration:none;font-size:.72rem;letter-spacing:.08em;
  text-transform:uppercase;color:#1e293b;transition:color .25s;cursor:pointer}
.lp-flinks a:hover{color:#334155}
.lp-fbot{max-width:1180px;margin:28px auto 0;padding-top:22px;
  border-top:1px solid rgba(255,255,255,.03);
  font-family:'Courier New',monospace;font-size:.6rem;
  letter-spacing:.12em;color:#0f172a;text-align:center;text-transform:uppercase}
/* ===== RESPONSIVE ===== */
@media(max-width:860px){
  .lp-hero-grid,.lp-mission-g,.lp-prob-g,.lp-feat-g,.lp-foot-g{grid-template-columns:1fr}
  .lp-planet-scene{height:320px;order:-1}
  .lp-planet{width:180px;height:180px}
  .lp-ring-1{width:290px;height:290px}
  .lp-ring-2{width:240px;height:240px}
  .lp-proc-g{grid-template-columns:repeat(2,1fr)}
  .lp-nav-links{display:none}
  .lp-flinks{justify-content:flex-start}
}
@media(max-width:520px){
  .lp-proc-g{grid-template-columns:1fr}
  .lp-stats{gap:0}
  .lp-stat{padding:10px 16px}
}
</style>

<div class="lp">
<div class="lp-stars"></div>
<div class="lp-nebula"></div>
<div class="lp-wrap">

<!-- NAV -->
<nav class="lp-nav">
  <div class="lp-brand">
    <div class="lp-orbit"></div>
    EXOPLANET AI
  </div>
  <ul class="lp-nav-links">
    <li><a href="#mission">Mission</a></li>
    <li><a href="#problem">Problem</a></li>
    <li><a href="#works">How It Works</a></li>
    <li><a href="#features">Features</a></li>
  </ul>
</nav>

<!-- HERO -->
<section class="lp-hero" id="home">
  <div class="lp-hero-grid">
    <div>
      <p class="lp-eyebrow">NASA Kepler &mdash; Artificial Intelligence</p>
      <h1 class="lp-h1">DISCOVER<br><span class="cy">WORLDS</span>BEYOND.</h1>
      <p class="lp-sub">AI-powered classification of Kepler Objects of Interest</p>
      <p class="lp-desc">Analyze stellar and transit properties to distinguish confirmed exoplanets from false positive signals detected by the Kepler telescope.</p>
      <!-- CTA placeholder -- actual Streamlit buttons injected below -->
      <div id="lp-cta-anchor" style="height:1px"></div>
    </div>
    <!-- PLANET -->
    <div class="lp-planet-scene">
      <div class="lp-aura"></div>
      <div class="lp-ring lp-ring-1"></div>
      <div class="lp-ring lp-ring-2"></div>
      <div class="lp-planet">
        <div class="lp-atm"></div>
        <div class="lp-scan"><div class="lp-scan-l"></div></div>
      </div>
      <div class="lp-moon"></div>
      <div class="lp-hud h1"><span class="lp-dot"></span><span class="hud-lbl">KOI Classification</span><span class="hud-val">ACTIVE</span></div>
      <div class="lp-hud h2"><span class="lp-dot"></span><span class="hud-lbl">Transit Signal</span><span class="hud-val">DETECTED</span></div>
      <div class="lp-hud h3"><span class="lp-dot"></span><span class="hud-lbl">Stellar Data</span><span class="hud-val">SYNCED</span></div>
      <div class="lp-hud h4"><span class="lp-dot"></span><span class="hud-lbl">AI Analysis</span><span class="hud-val">RUNNING</span></div>
      <div class="lp-hud h5"><span class="lp-dot"></span><span class="hud-lbl">XGBoost Model</span><span class="hud-val">LOADED</span></div>
    </div>
  </div>
  <!-- Stats -->
  <div style="max-width:1400px;margin:48px auto 0;padding:0 0">
    <div class="lp-stats">
      <div class="lp-stat"><div class="lp-stat-n">150K+</div><div class="lp-stat-l">Stars Observed</div></div>
      <div class="lp-stat"><div class="lp-stat-n">9K+</div><div class="lp-stat-l">KOIs Analyzed</div></div>
      <div class="lp-stat"><div class="lp-stat-n">ML</div><div class="lp-stat-l">Classification</div></div>
      <div class="lp-stat"><div class="lp-stat-n">SHAP</div><div class="lp-stat-l">Explainability</div></div>
    </div>
  </div>
</section>

<!-- MISSION -->
<section class="lp-sect" id="mission">
  <div class="lp-sect-inner">
    <p class="lp-lbl">The Mission</p>
    <div class="lp-mission-g">
      <div>
        <h2 class="lp-h2">Finding a planet<br>in a drop of light.</h2>
        <p class="lp-txt">NASA&rsquo;s Kepler telescope monitored over 150,000 stars continuously. When a planet passes in front of its host star, the star&rsquo;s brightness drops by a tiny fraction &mdash; creating a &ldquo;transit signal.&rdquo; Our ML system learns to distinguish genuine planetary transits from all other astrophysical and instrumental effects.</p>
      </div>
      <div class="lp-tl">
        <div class="lp-tl-step"><div class="lp-tl-l"><div class="lp-tld"></div><div class="lp-tll"></div></div><div class="lp-tc"><div class="lp-tt">Star</div><div class="lp-ts">Kepler monitors stellar brightness</div></div></div>
        <div class="lp-tl-step"><div class="lp-tl-l"><div class="lp-tld"></div><div class="lp-tll"></div></div><div class="lp-tc"><div class="lp-tt">Transit</div><div class="lp-ts">Periodic brightness dip is detected</div></div></div>
        <div class="lp-tl-step"><div class="lp-tl-l"><div class="lp-tld"></div><div class="lp-tll"></div></div><div class="lp-tc"><div class="lp-tt">Light Curve</div><div class="lp-ts">Transit shape and depth are measured</div></div></div>
        <div class="lp-tl-step"><div class="lp-tl-l"><div class="lp-tld"></div><div class="lp-tll"></div></div><div class="lp-tc"><div class="lp-tt">KOI</div><div class="lp-ts">Candidate catalogued for investigation</div></div></div>
        <div class="lp-tl-step"><div class="lp-tl-l"><div class="lp-tld" style="background:#a78bfa;box-shadow:0 0 8px #a78bfa"></div></div><div class="lp-tc"><div class="lp-tt" style="color:#a78bfa">AI Classification</div><div class="lp-ts">Model predicts: Confirmed or False Positive</div></div></div>
      </div>
    </div>
  </div>
</section>

<!-- PROBLEM -->
<section class="lp-sect" id="problem">
  <div class="lp-sect-inner">
    <p class="lp-lbl">The Problem</p>
    <h2 class="lp-h2">Not every signal<br>is a planet.</h2>
    <div class="lp-prob-g">
      <div class="lp-pcard">
        <div class="lp-ctype cconf">// Confirmed Exoplanet</div>
        <div class="lp-ctitle">🪐 Real World</div>
        <div class="lp-cdesc">A genuine planet transiting its host star. Transit depth, duration, and shape are consistent with a planetary body. Signal-to-noise ratio is high and repeatable across multiple observations.</div>
        <span class="lp-cbadge bdgc">Target&nbsp;=&nbsp;1</span>
      </div>
      <div class="lp-pcard">
        <div class="lp-ctype cfp">// False Positive</div>
        <div class="lp-ctitle">⭐ Stellar Mimic</div>
        <div class="lp-cdesc">A brightness dip caused by a background eclipsing binary, a grazing stellar eclipse, or instrumental artifacts. These can closely resemble planetary transits but require different physical explanations.</div>
        <span class="lp-cbadge bdgf">Target&nbsp;=&nbsp;0</span>
      </div>
    </div>
  </div>
</section>

<!-- HOW IT WORKS -->
<section class="lp-sect" id="works">
  <div class="lp-sect-inner">
    <p class="lp-lbl">How It Works</p>
    <h2 class="lp-h2">From starlight<br>to insight.</h2>
    <div class="lp-proc-g">
      <div class="lp-pstep"><div class="lp-pnum">01</div><div class="lp-ptitle">Input</div><div class="lp-pdesc">Kepler candidate measurements: orbital period, transit depth, duration, planetary radius, stellar properties.</div></div>
      <div class="lp-pstep"><div class="lp-pnum">02</div><div class="lp-ptitle">Analyze</div><div class="lp-pdesc">XGBoost processes 11 standardized features. Trained on 6,069 labeled KOIs with class-imbalance handling.</div></div>
      <div class="lp-pstep"><div class="lp-pnum">03</div><div class="lp-ptitle">Classify</div><div class="lp-pdesc">Model outputs a probability. Threshold 0.61 applied to label the candidate as Confirmed or False Positive.</div></div>
      <div class="lp-pstep"><div class="lp-pnum">04</div><div class="lp-ptitle">Explain</div><div class="lp-pdesc">SHAP values reveal which features drove the prediction. Planetary radius and SNR are consistently most influential.</div></div>
    </div>
  </div>
</section>

<!-- FEATURES -->
<section class="lp-sect" id="features">
  <div class="lp-sect-inner">
    <p class="lp-lbl">Capabilities</p>
    <h2 class="lp-h2">Built for<br>exoplanet analysis.</h2>
    <div class="lp-feat-g">
      <div class="lp-fcard"><div class="lp-ficon">🤖</div><div><div class="lp-fnum">// 01</div><div class="lp-ftitle">Multi-Model AI</div><div class="lp-fdesc">Logistic Regression, Decision Tree, Random Forest, and XGBoost compared with 5-fold CV and PR-AUC as primary metric.</div></div></div>
      <div class="lp-fcard"><div class="lp-ficon">⚖️</div><div><div class="lp-fnum">// 02</div><div class="lp-ftitle">Imbalance Handling</div><div class="lp-fdesc">Class weighting and SMOTE (inside CV folds) tested for the 36/64 class split. XGBoost scale_pos_weight = 1.7611.</div></div></div>
      <div class="lp-fcard"><div class="lp-ficon">🔍</div><div><div class="lp-fnum">// 03</div><div class="lp-ftitle">Explainable AI</div><div class="lp-fdesc">SHAP TreeExplainer for global and local explanations. Permutation importance cross-validates feature rankings.</div></div></div>
      <div class="lp-fcard"><div class="lp-ficon">🚀</div><div><div class="lp-fnum">// 04</div><div class="lp-ftitle">Interactive Analysis</div><div class="lp-fdesc">Single-KOI prediction form, batch CSV upload, and candidate explorer for 1,977 unlabeled Kepler objects.</div></div></div>
    </div>
  </div>
</section>

<!-- FINAL CTA placeholder (Streamlit button injected below) -->
<section class="lp-cta-sect" id="launch">
  <div style="position:absolute;inset:0;background:radial-gradient(ellipse at 50% 50%,rgba(99,102,241,.07) 0%,transparent 65%);pointer-events:none"></div>
  <div class="lp-cta-inner">
    <p class="lp-lbl" style="text-align:center">// Mission Control</p>
    <h2 class="lp-cta-h">Ready to explore<br><span class="cy">a new world?</span></h2>
    <p class="lp-cta-d">Enter a Kepler Object of Interest and let AI analyze the evidence hidden in its stellar signal.</p>
    <div id="lp-final-cta-anchor" style="height:1px"></div>
  </div>
</section>

<!-- FOOTER -->
<footer class="lp-foot">
  <div class="lp-foot-g">
    <div>
      <div class="lp-fbrand"><div class="lp-orbit" style="width:20px;height:20px"></div>EXOPLANET AI</div>
      <div class="lp-ftag">AI-powered exploration of Kepler Objects of Interest.<br>Not a scientific confirmation instrument.</div>
    </div>
    <nav class="lp-flinks">
      <a href="#mission">Mission</a>
      <a href="#works">How It Works</a>
      <a href="#features">Features</a>
    </nav>
  </div>
  <div class="lp-fbot">Built with Machine Learning &bull; Explainable AI &bull; NASA Kepler Data</div>
</footer>

</div></div>
""", unsafe_allow_html=True)

    # ── Streamlit CTA buttons (session-state navigation -- reliable) ─────────
    # Injected into the page flow directly after the HTML block.
    # They appear below the hero section in Streamlit's layout.
    st.markdown("---")
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.markdown(
            "<p style='text-align:center;font-family:Courier New,monospace;"
            "font-size:.7rem;letter-spacing:.2em;color:#475569;text-transform:uppercase;"
            "margin-bottom:12px'>// Launch the classifier to begin</p>",
            unsafe_allow_html=True,
        )
        if st.button("🚀  START EXPLORING  →", use_container_width=True, type="primary"):
            st.session_state.app_page = "classifier"
            st.rerun()
        st.markdown(
            "<p style='text-align:center;font-size:.75rem;color:#1e293b;"
            "margin-top:10px;font-family:Courier New,monospace;letter-spacing:.1em'>"
            "Predictions are ML outputs — not scientific confirmation</p>",
            unsafe_allow_html=True,
        )




if __name__ == "__main__":
    main()
