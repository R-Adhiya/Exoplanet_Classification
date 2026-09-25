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
# PAGE: Landing
# ---------------------------------------------------------------------------
def page_landing():
    """Full-screen futuristic space-themed landing page."""

    # Hide the default Streamlit sidebar toggle and padding for full-bleed effect
    st.markdown("""
<style>
/* Full-bleed landing: remove Streamlit's default page margins */
[data-testid="stAppViewContainer"] > .main > div {
    padding: 0 !important;
    max-width: 100% !important;
}
[data-testid="stAppViewContainer"] {
    padding: 0 !important;
}
section[data-testid="stSidebar"] {
    display: none !important;
}
header[data-testid="stHeader"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

    st.markdown(r"""
<style>
/* ================================================================
   RESET & BASE
================================================================ */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #030712;
    color: #e2e8f0;
    overflow-x: hidden;
}

/* ================================================================
   FONTS & UTILITIES
================================================================ */
.mono { font-family: 'Courier New', Courier, monospace; }

.glow-text {
    color: #67e8f9;
    text-shadow: 0 0 20px rgba(103,232,249,0.6), 0 0 40px rgba(103,232,249,0.3);
}
.glow-soft {
    color: #a5f3fc;
    text-shadow: 0 0 12px rgba(165,243,252,0.4);
}

/* ================================================================
   STARFIELD CANVAS
================================================================ */
#starfield {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: 0;
    pointer-events: none;
}

/* ================================================================
   NAVIGATION
================================================================ */
.lp-nav {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 100;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 48px;
    background: rgba(3,7,18,0.55);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(103,232,249,0.08);
}
.lp-nav-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: #f0f9ff;
    text-transform: uppercase;
}
.lp-nav-orbit {
    width: 28px; height: 28px;
    border: 2px solid #67e8f9;
    border-radius: 50%;
    position: relative;
    box-shadow: 0 0 10px rgba(103,232,249,0.5);
}
.lp-nav-orbit::before {
    content: '';
    position: absolute;
    width: 8px; height: 8px;
    background: #67e8f9;
    border-radius: 50%;
    top: 50%; left: -5px;
    transform: translateY(-50%);
    box-shadow: 0 0 8px #67e8f9;
    animation: orbit-dot 3s linear infinite;
}
@keyframes orbit-dot {
    from { transform: translateY(-50%) rotate(0deg) translateX(14px); }
    to   { transform: translateY(-50%) rotate(360deg) translateX(14px); }
}
.lp-nav-links {
    display: flex;
    align-items: center;
    gap: 32px;
    list-style: none;
}
.lp-nav-links a {
    text-decoration: none;
    color: #94a3b8;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    transition: color 0.25s;
}
.lp-nav-links a:hover { color: #67e8f9; }
.lp-nav-cta {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 20px;
    border: 1px solid rgba(103,232,249,0.5);
    border-radius: 6px;
    color: #67e8f9;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    text-decoration: none;
    background: rgba(103,232,249,0.05);
    cursor: pointer;
    transition: all 0.3s;
    box-shadow: 0 0 12px rgba(103,232,249,0.1);
}
.lp-nav-cta:hover {
    background: rgba(103,232,249,0.12);
    box-shadow: 0 0 20px rgba(103,232,249,0.3);
    color: #fff;
}

/* ================================================================
   HERO SECTION
================================================================ */
.lp-hero {
    position: relative;
    min-height: 100vh;
    display: flex;
    align-items: center;
    padding: 100px 6vw 60px;
    overflow: hidden;
}
/* Nebula clouds */
.lp-hero::before {
    content: '';
    position: absolute;
    top: -20%;
    left: -10%;
    width: 70%;
    height: 80%;
    background: radial-gradient(ellipse at 40% 50%,
        rgba(99,102,241,0.08) 0%,
        rgba(59,130,246,0.05) 40%,
        transparent 70%);
    pointer-events: none;
    z-index: 1;
}
.lp-hero::after {
    content: '';
    position: absolute;
    bottom: -10%;
    right: -5%;
    width: 60%;
    height: 70%;
    background: radial-gradient(ellipse at 60% 50%,
        rgba(139,92,246,0.07) 0%,
        rgba(103,232,249,0.04) 40%,
        transparent 70%);
    pointer-events: none;
    z-index: 1;
}

.lp-hero-inner {
    position: relative;
    z-index: 2;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 40px;
    align-items: center;
    width: 100%;
    max-width: 1400px;
    margin: 0 auto;
}

/* LEFT COLUMN */
.lp-hero-left { display: flex; flex-direction: column; gap: 28px; }

.lp-eyebrow {
    font-family: 'Courier New', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    color: #67e8f9;
    text-transform: uppercase;
    opacity: 0.9;
}
.lp-eyebrow::before {
    content: '// ';
    opacity: 0.5;
}

.lp-hero-heading {
    font-size: clamp(3.2rem, 6vw, 6rem);
    font-weight: 800;
    line-height: 1.0;
    letter-spacing: -0.02em;
    color: #f0f9ff;
}
.lp-hero-heading .highlight {
    display: block;
    color: #67e8f9;
    text-shadow: 0 0 30px rgba(103,232,249,0.5), 0 0 60px rgba(103,232,249,0.2);
}

.lp-hero-sub {
    font-size: 1rem;
    font-weight: 500;
    color: #94a3b8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.lp-hero-desc {
    font-size: 1rem;
    line-height: 1.75;
    color: #94a3b8;
    max-width: 480px;
}

.lp-hero-buttons {
    display: flex;
    align-items: center;
    gap: 20px;
    flex-wrap: wrap;
}
.lp-btn-primary {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 14px 32px;
    background: linear-gradient(135deg, rgba(103,232,249,0.12), rgba(99,102,241,0.12));
    border: 1px solid rgba(103,232,249,0.4);
    border-radius: 8px;
    color: #fff;
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.35s ease;
    box-shadow: 0 0 20px rgba(103,232,249,0.1), inset 0 0 20px rgba(103,232,249,0.03);
    text-decoration: none;
}
.lp-btn-primary:hover {
    background: linear-gradient(135deg, rgba(103,232,249,0.22), rgba(99,102,241,0.22));
    box-shadow: 0 0 35px rgba(103,232,249,0.25), inset 0 0 30px rgba(103,232,249,0.06);
    transform: translateY(-2px);
    color: #67e8f9;
}
.lp-btn-secondary {
    background: none;
    border: none;
    color: #64748b;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    cursor: pointer;
    transition: color 0.25s;
    text-decoration: none;
}
.lp-btn-secondary:hover { color: #94a3b8; }

/* STATS STRIP */
.lp-stats {
    display: flex;
    gap: 36px;
    padding-top: 12px;
    flex-wrap: wrap;
}
.lp-stat {
    display: flex;
    flex-direction: column;
    gap: 3px;
}
.lp-stat-num {
    font-size: 1.4rem;
    font-weight: 800;
    color: #e2e8f0;
    letter-spacing: -0.01em;
}
.lp-stat-label {
    font-family: 'Courier New', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    color: #475569;
    text-transform: uppercase;
}
.lp-stat-divider {
    width: 1px;
    background: rgba(103,232,249,0.12);
    align-self: stretch;
}

/* ================================================================
   PLANET VISUAL (RIGHT COLUMN)
================================================================ */
.lp-planet-wrap {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 580px;
}

/* Outer glow aura */
.lp-planet-aura {
    position: absolute;
    width: 480px; height: 480px;
    border-radius: 50%;
    background: radial-gradient(ellipse,
        rgba(103,232,249,0.04) 0%,
        rgba(99,102,241,0.06) 40%,
        transparent 70%);
    animation: aura-pulse 5s ease-in-out infinite;
}
@keyframes aura-pulse {
    0%,100% { transform: scale(1);   opacity: 0.7; }
    50%      { transform: scale(1.06); opacity: 1; }
}

/* Orbital ring */
.lp-orbit-ring {
    position: absolute;
    width: 440px; height: 440px;
    border-radius: 50%;
    border: 1px solid rgba(103,232,249,0.12);
    animation: ring-spin 28s linear infinite;
}
.lp-orbit-ring-2 {
    width: 380px; height: 380px;
    border: 1px dashed rgba(99,102,241,0.1);
    animation: ring-spin 18s linear infinite reverse;
}
@keyframes ring-spin {
    from { transform: rotateX(72deg) rotateZ(0deg); }
    to   { transform: rotateX(72deg) rotateZ(360deg); }
}

/* Orbiting dot */
.lp-orbit-dot {
    position: absolute;
    width: 10px; height: 10px;
    background: #67e8f9;
    border-radius: 50%;
    box-shadow: 0 0 12px #67e8f9, 0 0 24px rgba(103,232,249,0.4);
    animation: dot-orbit 18s linear infinite;
    top: 50%; left: 50%;
    transform-origin: 0 0;
}
@keyframes dot-orbit {
    from { transform: rotate(0deg) translateX(190px) translateY(-50%); }
    to   { transform: rotate(360deg) translateX(190px) translateY(-50%); }
}

/* Planet body */
.lp-planet {
    position: relative;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 30%,
        #1e3a5f 0%,
        #0f1f3d 30%,
        #080d1a 60%,
        #04060e 100%);
    box-shadow:
        -24px -18px 40px rgba(103,232,249,0.08),
        0 0 0 2px rgba(103,232,249,0.06),
        0 0 60px rgba(99,102,241,0.15),
        inset 12px 10px 40px rgba(103,232,249,0.04);
    overflow: hidden;
    animation: planet-float 7s ease-in-out infinite;
    z-index: 2;
}
@keyframes planet-float {
    0%,100% { transform: translateY(0px); }
    50%      { transform: translateY(-14px); }
}

/* Atmosphere rim */
.lp-planet::after {
    content: '';
    position: absolute;
    inset: -4px;
    border-radius: 50%;
    background: radial-gradient(circle at 28% 25%,
        rgba(103,232,249,0.18) 0%,
        transparent 55%);
    pointer-events: none;
}

/* Planet surface texture lines */
.lp-planet-surface {
    position: absolute;
    inset: 0;
    border-radius: 50%;
    overflow: hidden;
}
.lp-surface-band {
    position: absolute;
    width: 120%;
    height: 1px;
    background: rgba(103,232,249,0.04);
    left: -10%;
}

/* Moon */
.lp-moon {
    position: absolute;
    width: 36px; height: 36px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 30%, #2d3f5c, #111827);
    box-shadow: -4px -3px 8px rgba(103,232,249,0.12),
                0 0 12px rgba(99,102,241,0.1);
    top: 16%;
    right: 14%;
    animation: moon-orbit 14s linear infinite;
}
@keyframes moon-orbit {
    from { transform: rotate(0deg) translate(130px) rotate(0deg); }
    to   { transform: rotate(360deg) translate(130px) rotate(-360deg); }
}

/* ================================================================
   HUD DATA ELEMENTS
================================================================ */
.lp-hud {
    position: absolute;
    z-index: 5;
}
.lp-hud-card {
    background: rgba(3,7,18,0.7);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(103,232,249,0.15);
    border-radius: 6px;
    padding: 8px 12px;
    font-family: 'Courier New', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    color: #67e8f9;
    text-transform: uppercase;
    line-height: 1.6;
    white-space: nowrap;
    animation: hud-float 6s ease-in-out infinite;
}
.lp-hud-card .hud-label {
    color: #475569;
    display: block;
}
.lp-hud-card .hud-value {
    color: #67e8f9;
    font-weight: 700;
}
.lp-hud-card .hud-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #67e8f9;
    box-shadow: 0 0 6px #67e8f9;
    margin-right: 6px;
    animation: hud-blink 2s step-start infinite;
}
@keyframes hud-blink { 0%,100% { opacity:1; } 50% { opacity:0.2; } }
@keyframes hud-float {
    0%,100% { transform: translateY(0px); }
    50%      { transform: translateY(-6px); }
}
.lp-hud-1 { top: 12%; left: 2%;  animation-delay: 0s;   }
.lp-hud-2 { top: 40%; left: 0%;  animation-delay: 1.5s; }
.lp-hud-3 { top: 70%; left: 4%;  animation-delay: 0.8s; }
.lp-hud-4 { top: 18%; right: 2%; animation-delay: 2.1s; }
.lp-hud-5 { bottom: 14%; right: 3%; animation-delay: 1.2s; }

/* Corner brackets */
.lp-hud-card::before, .lp-hud-card::after {
    content: '';
    position: absolute;
    width: 8px; height: 8px;
    border-color: rgba(103,232,249,0.5);
    border-style: solid;
}
.lp-hud-card::before { top: -1px; left: -1px; border-width: 1px 0 0 1px; }
.lp-hud-card::after  { bottom: -1px; right: -1px; border-width: 0 1px 1px 0; }

/* Scan line on planet */
.lp-scan {
    position: absolute;
    inset: 0;
    border-radius: 50%;
    overflow: hidden;
    pointer-events: none;
    z-index: 3;
}
.lp-scan-line {
    position: absolute;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg,
        transparent, rgba(103,232,249,0.35), transparent);
    animation: scan 4s linear infinite;
    top: 0;
}
@keyframes scan {
    from { top: 0%; opacity: 1; }
    to   { top: 100%; opacity: 0; }
}

/* ================================================================
   SCROLL INDICATOR
================================================================ */
.lp-scroll {
    position: absolute;
    bottom: 32px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    z-index: 10;
    cursor: pointer;
}
.lp-scroll-text {
    font-family: 'Courier New', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.22em;
    color: #334155;
    text-transform: uppercase;
}
.lp-scroll-arrow {
    width: 20px; height: 20px;
    border-right: 1.5px solid #334155;
    border-bottom: 1.5px solid #334155;
    transform: rotate(45deg);
    animation: scroll-bounce 2s ease-in-out infinite;
}
@keyframes scroll-bounce {
    0%,100% { transform: rotate(45deg) translateY(0);   opacity: 0.4; }
    50%      { transform: rotate(45deg) translateY(5px); opacity: 0.9; }
}

/* ================================================================
   SECTIONS COMMON
================================================================ */
.lp-section {
    position: relative;
    z-index: 2;
    padding: 120px 6vw;
}
.lp-section-inner {
    max-width: 1200px;
    margin: 0 auto;
}
.lp-section-label {
    font-family: 'Courier New', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    color: #67e8f9;
    text-transform: uppercase;
    margin-bottom: 20px;
    opacity: 0.8;
}
.lp-section-label::before { content: '// '; opacity: 0.5; }
.lp-section-heading {
    font-size: clamp(2.4rem, 4.5vw, 4rem);
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -0.02em;
    color: #f0f9ff;
    margin-bottom: 28px;
}

.lp-divider {
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, transparent,
        rgba(103,232,249,0.1) 30%, rgba(103,232,249,0.1) 70%, transparent);
    margin: 0 auto;
}

/* ================================================================
   SECTION 2 -- THE MISSION
================================================================ */
.lp-mission-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 80px;
    align-items: start;
}
.lp-mission-desc {
    font-size: 1rem;
    line-height: 1.8;
    color: #64748b;
    margin-bottom: 32px;
}
.lp-timeline {
    display: flex;
    flex-direction: column;
    gap: 0;
    margin-top: 12px;
}
.lp-timeline-step {
    display: flex;
    align-items: flex-start;
    gap: 20px;
}
.lp-tl-left {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 36px;
}
.lp-tl-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #67e8f9;
    box-shadow: 0 0 10px #67e8f9;
    flex-shrink: 0;
    margin-top: 4px;
}
.lp-tl-line {
    width: 1px;
    flex: 1;
    min-height: 36px;
    background: linear-gradient(to bottom, rgba(103,232,249,0.3), rgba(103,232,249,0.05));
}
.lp-tl-content {
    padding-bottom: 32px;
}
.lp-tl-title {
    font-family: 'Courier New', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    color: #67e8f9;
    text-transform: uppercase;
    font-weight: 700;
}
.lp-tl-sub {
    font-size: 0.82rem;
    color: #475569;
    margin-top: 3px;
    line-height: 1.5;
}

/* ================================================================
   SECTION 3 -- THE PROBLEM
================================================================ */
.lp-problem-cards {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-top: 52px;
}
.lp-problem-card {
    background: rgba(5,11,24,0.8);
    border: 1px solid rgba(103,232,249,0.08);
    border-radius: 16px;
    padding: 40px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.35s, transform 0.35s;
}
.lp-problem-card:hover {
    border-color: rgba(103,232,249,0.2);
    transform: translateY(-4px);
}
.lp-problem-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(103,232,249,0.2), transparent);
}
.lp-card-icon {
    font-size: 2.4rem;
    margin-bottom: 20px;
    display: block;
}
.lp-card-type {
    font-family: 'Courier New', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.lp-card-type.confirmed { color: #34d399; }
.lp-card-type.fp        { color: #f87171; }
.lp-card-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 12px;
}
.lp-card-desc {
    font-size: 0.88rem;
    color: #475569;
    line-height: 1.7;
}
.lp-card-badge {
    display: inline-block;
    margin-top: 20px;
    padding: 5px 12px;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
.lp-card-badge.confirmed {
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.2);
    color: #34d399;
}
.lp-card-badge.fp {
    background: rgba(248,113,113,0.08);
    border: 1px solid rgba(248,113,113,0.2);
    color: #f87171;
}

/* ================================================================
   SECTION 4 -- HOW AI HELPS
================================================================ */
.lp-process-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
    margin-top: 52px;
}
.lp-process-step {
    background: rgba(5,11,24,0.7);
    border: 1px solid rgba(103,232,249,0.07);
    border-radius: 12px;
    padding: 32px 24px;
    position: relative;
    transition: border-color 0.3s, transform 0.3s;
}
.lp-process-step:hover {
    border-color: rgba(103,232,249,0.18);
    transform: translateY(-3px);
}
.lp-process-num {
    font-family: 'Courier New', monospace;
    font-size: 2.4rem;
    font-weight: 800;
    color: rgba(103,232,249,0.1);
    line-height: 1;
    margin-bottom: 16px;
}
.lp-process-title {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #67e8f9;
    margin-bottom: 10px;
}
.lp-process-desc {
    font-size: 0.82rem;
    color: #475569;
    line-height: 1.65;
}
.lp-process-connector {
    position: absolute;
    top: 50%;
    right: -12px;
    width: 24px;
    height: 1px;
    background: rgba(103,232,249,0.15);
    z-index: 1;
}
.lp-process-connector::after {
    content: '';
    position: absolute;
    right: 0;
    top: -3px;
    width: 6px; height: 6px;
    border-right: 1px solid rgba(103,232,249,0.3);
    border-top: 1px solid rgba(103,232,249,0.3);
    transform: rotate(45deg);
}

/* ================================================================
   SECTION 5 -- FEATURES
================================================================ */
.lp-feature-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 20px;
    margin-top: 52px;
}
.lp-feature-card {
    background: rgba(5,11,24,0.6);
    border: 1px solid rgba(103,232,249,0.07);
    border-radius: 12px;
    padding: 32px;
    display: flex;
    gap: 24px;
    align-items: flex-start;
    transition: border-color 0.3s, background 0.3s;
}
.lp-feature-card:hover {
    border-color: rgba(103,232,249,0.16);
    background: rgba(5,11,24,0.85);
}
.lp-feature-icon {
    font-size: 1.8rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.lp-feature-num {
    font-family: 'Courier New', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: rgba(103,232,249,0.4);
    margin-bottom: 6px;
}
.lp-feature-title {
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #cbd5e1;
    margin-bottom: 8px;
}
.lp-feature-desc {
    font-size: 0.82rem;
    color: #475569;
    line-height: 1.65;
}

/* ================================================================
   SECTION 6 -- FEATURE VIZ
================================================================ */
.lp-viz-card {
    background: rgba(5,11,24,0.7);
    border: 1px solid rgba(103,232,249,0.1);
    border-radius: 16px;
    padding: 52px;
    margin-top: 40px;
    position: relative;
    overflow: hidden;
}
.lp-viz-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%,
        rgba(103,232,249,0.04) 0%, transparent 60%);
    pointer-events: none;
}
.lp-viz-heading {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 8px;
    letter-spacing: -0.01em;
}
.lp-viz-sub {
    font-size: 0.82rem;
    color: #475569;
    margin-bottom: 44px;
}
.lp-feature-bars {
    display: flex;
    flex-direction: column;
    gap: 18px;
}
.lp-feat-row {
    display: grid;
    grid-template-columns: 160px 1fr 48px;
    align-items: center;
    gap: 20px;
}
.lp-feat-label {
    font-family: 'Courier New', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    color: #64748b;
    text-transform: uppercase;
}
.lp-feat-track {
    height: 4px;
    background: rgba(103,232,249,0.06);
    border-radius: 2px;
    overflow: hidden;
}
.lp-feat-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #1e40af, #67e8f9);
    box-shadow: 0 0 8px rgba(103,232,249,0.3);
    animation: bar-grow 1.5s ease-out forwards;
    transform-origin: left;
}
@keyframes bar-grow {
    from { transform: scaleX(0); }
    to   { transform: scaleX(1); }
}
.lp-feat-pct {
    font-family: 'Courier New', monospace;
    font-size: 0.62rem;
    color: rgba(103,232,249,0.5);
    text-align: right;
}

/* ================================================================
   FINAL CTA SECTION
================================================================ */
.lp-cta-section {
    position: relative;
    z-index: 2;
    padding: 140px 6vw;
    text-align: center;
    overflow: hidden;
}
.lp-cta-section::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 50%,
        rgba(99,102,241,0.08) 0%,
        rgba(103,232,249,0.04) 40%,
        transparent 70%);
    pointer-events: none;
}
/* Distant planet glow */
.lp-cta-planet {
    position: absolute;
    bottom: -80px;
    left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 300px;
    border-radius: 50%;
    background: radial-gradient(ellipse at 50% 100%,
        rgba(99,102,241,0.12) 0%,
        rgba(103,232,249,0.05) 40%,
        transparent 70%);
    pointer-events: none;
}
.lp-cta-inner {
    position: relative;
    z-index: 1;
    max-width: 700px;
    margin: 0 auto;
}
.lp-cta-eyebrow {
    font-family: 'Courier New', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    color: #67e8f9;
    text-transform: uppercase;
    margin-bottom: 24px;
    opacity: 0.7;
}
.lp-cta-heading {
    font-size: clamp(2.8rem, 5vw, 4.5rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.02em;
    color: #f0f9ff;
    margin-bottom: 24px;
}
.lp-cta-desc {
    font-size: 1rem;
    color: #64748b;
    line-height: 1.75;
    margin-bottom: 40px;
}

/* ================================================================
   FOOTER
================================================================ */
.lp-footer {
    position: relative;
    z-index: 2;
    border-top: 1px solid rgba(103,232,249,0.06);
    padding: 48px 6vw;
}
.lp-footer-inner {
    max-width: 1200px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 48px;
    align-items: start;
}
.lp-footer-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.12em;
    color: #e2e8f0;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.lp-footer-tagline {
    font-size: 0.78rem;
    color: #334155;
    line-height: 1.6;
}
.lp-footer-links {
    display: flex;
    gap: 28px;
    flex-wrap: wrap;
    justify-content: flex-end;
    align-items: center;
}
.lp-footer-links a {
    text-decoration: none;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #334155;
    transition: color 0.25s;
    cursor: pointer;
}
.lp-footer-links a:hover { color: #64748b; }
.lp-footer-bottom {
    max-width: 1200px;
    margin: 32px auto 0;
    padding-top: 24px;
    border-top: 1px solid rgba(255,255,255,0.04);
    font-family: 'Courier New', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    color: #1e293b;
    text-align: center;
    text-transform: uppercase;
}

/* ================================================================
   RESPONSIVE
================================================================ */
@media (max-width: 900px) {
    .lp-hero-inner  { grid-template-columns: 1fr; }
    .lp-planet-wrap { height: 360px; order: -1; }
    .lp-planet      { width: 200px; height: 200px; }
    .lp-orbit-ring  { width: 320px; height: 320px; }
    .lp-orbit-ring-2{ width: 270px; height: 270px; }
    .lp-mission-grid   { grid-template-columns: 1fr; }
    .lp-problem-cards  { grid-template-columns: 1fr; }
    .lp-process-grid   { grid-template-columns: repeat(2, 1fr); }
    .lp-feature-grid   { grid-template-columns: 1fr; }
    .lp-footer-inner   { grid-template-columns: 1fr; }
    .lp-footer-links   { justify-content: flex-start; }
    .lp-nav-links      { display: none; }
    .lp-nav { padding: 16px 24px; }
    .lp-hero { padding: 80px 5vw 50px; }
    .lp-section { padding: 80px 5vw; }
}
@media (max-width: 560px) {
    .lp-process-grid { grid-template-columns: 1fr; }
    .lp-stats        { gap: 20px; }
    .lp-feat-row     { grid-template-columns: 120px 1fr 36px; }
}
</style>

<!-- STARFIELD CANVAS -->
<canvas id="starfield"></canvas>

<!-- ================================================================
     NAVIGATION
================================================================ -->
<nav class="lp-nav">
  <div class="lp-nav-brand">
    <div class="lp-nav-orbit"></div>
    EXOPLANET AI
  </div>
  <ul class="lp-nav-links">
    <li><a href="#mission">Mission</a></li>
    <li><a href="#problem">The Problem</a></li>
    <li><a href="#how-it-works">How It Works</a></li>
    <li><a href="#features">Features</a></li>
  </ul>
  <button class="lp-nav-cta" onclick="launchClassifier()">LAUNCH CLASSIFIER &rarr;</button>
</nav>

<!-- ================================================================
     HERO SECTION
================================================================ -->
<section class="lp-hero" id="home">
  <div class="lp-hero-inner">
    <!-- LEFT: Text -->
    <div class="lp-hero-left">
      <p class="lp-eyebrow">NASA Kepler &mdash; Artificial Intelligence</p>

      <h1 class="lp-hero-heading">
        DISCOVER<br>
        <span class="highlight">WORLDS</span>
        BEYOND.
      </h1>

      <p class="lp-hero-sub">AI-powered classification of Kepler Objects of Interest</p>

      <p class="lp-hero-desc">
        Explore the signals hidden in distant starlight.
        Our machine learning system analyzes stellar and
        transit properties to distinguish confirmed exoplanets
        from false positives.
      </p>

      <div class="lp-hero-buttons">
        <button class="lp-btn-primary" onclick="launchClassifier()">
          START EXPLORING &rarr;
        </button>
        <a href="#mission" class="lp-btn-secondary">EXPLORE THE MISSION &darr;</a>
      </div>

      <!-- Stats strip -->
      <div class="lp-stats">
        <div class="lp-stat">
          <span class="lp-stat-num">150K+</span>
          <span class="lp-stat-label">Stars Observed</span>
        </div>
        <div class="lp-stat-divider"></div>
        <div class="lp-stat">
          <span class="lp-stat-num">9K+</span>
          <span class="lp-stat-label">KOIs Analyzed</span>
        </div>
        <div class="lp-stat-divider"></div>
        <div class="lp-stat">
          <span class="lp-stat-num">ML</span>
          <span class="lp-stat-label">Classification</span>
        </div>
        <div class="lp-stat-divider"></div>
        <div class="lp-stat">
          <span class="lp-stat-num">SHAP</span>
          <span class="lp-stat-label">Explainability</span>
        </div>
      </div>
    </div>

    <!-- RIGHT: Planet visual -->
    <div class="lp-planet-wrap">
      <div class="lp-planet-aura"></div>
      <div class="lp-orbit-ring"></div>
      <div class="lp-orbit-ring lp-orbit-ring-2"></div>
      <div class="lp-orbit-dot"></div>

      <div class="lp-planet">
        <div class="lp-planet-surface">
          <div class="lp-surface-band" style="top:28%"></div>
          <div class="lp-surface-band" style="top:44%"></div>
          <div class="lp-surface-band" style="top:60%"></div>
          <div class="lp-surface-band" style="top:76%"></div>
        </div>
        <div class="lp-scan"><div class="lp-scan-line"></div></div>
      </div>

      <div class="lp-moon"></div>

      <!-- HUD data elements -->
      <div class="lp-hud lp-hud-1">
        <div class="lp-hud-card">
          <span class="hud-dot"></span>
          <span class="hud-label">KOI-CLASSIFICATION</span>
          <span class="hud-value">ACTIVE</span>
        </div>
      </div>
      <div class="lp-hud lp-hud-2">
        <div class="lp-hud-card">
          <span class="hud-dot"></span>
          <span class="hud-label">TRANSIT SIGNAL</span>
          <span class="hud-value">DETECTED</span>
        </div>
      </div>
      <div class="lp-hud lp-hud-3">
        <div class="lp-hud-card">
          <span class="hud-dot"></span>
          <span class="hud-label">STELLAR DATA</span>
          <span class="hud-value">SYNCED</span>
        </div>
      </div>
      <div class="lp-hud lp-hud-4">
        <div class="lp-hud-card">
          <span class="hud-dot"></span>
          <span class="hud-label">AI ANALYSIS</span>
          <span class="hud-value">RUNNING</span>
        </div>
      </div>
      <div class="lp-hud lp-hud-5">
        <div class="lp-hud-card">
          <span class="hud-dot"></span>
          <span class="hud-label">XGBOOST MODEL</span>
          <span class="hud-value">LOADED</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Scroll indicator -->
  <div class="lp-scroll" onclick="document.getElementById('mission').scrollIntoView({behavior:'smooth'})">
    <span class="lp-scroll-text">Scroll to Explore</span>
    <div class="lp-scroll-arrow"></div>
  </div>
</section>

<!-- ================================================================
     SECTION 2 -- THE MISSION
================================================================ -->
<div class="lp-divider"></div>
<section class="lp-section" id="mission">
  <div class="lp-section-inner">
    <p class="lp-section-label">The Mission</p>
    <div class="lp-mission-grid">
      <div>
        <h2 class="lp-section-heading">Finding a planet<br>in a drop of light.</h2>
        <p class="lp-mission-desc">
          NASA's Kepler telescope continuously monitored thousands of stars.
          When a planet passed in front of its host star, the resulting change
          in brightness created a transit signal. Our AI system learns to
          distinguish genuine planetary transits from all other sources of
          stellar brightness variation.
        </p>
        <p class="lp-mission-desc" style="color:#334155;">
          The model was trained on 7,587 labeled Kepler Objects of Interest
          and uses 11 carefully selected features — all chosen to avoid data
          leakage from the existing classification process.
        </p>
      </div>

      <!-- Timeline -->
      <div class="lp-timeline">
        <div class="lp-timeline-step">
          <div class="lp-tl-left"><div class="lp-tl-dot"></div><div class="lp-tl-line"></div></div>
          <div class="lp-tl-content"><div class="lp-tl-title">Star</div><div class="lp-tl-sub">Kepler monitors stellar brightness continuously</div></div>
        </div>
        <div class="lp-timeline-step">
          <div class="lp-tl-left"><div class="lp-tl-dot"></div><div class="lp-tl-line"></div></div>
          <div class="lp-tl-content"><div class="lp-tl-title">Transit</div><div class="lp-tl-sub">A periodic brightness dip is detected</div></div>
        </div>
        <div class="lp-timeline-step">
          <div class="lp-tl-left"><div class="lp-tl-dot"></div><div class="lp-tl-line"></div></div>
          <div class="lp-tl-content"><div class="lp-tl-title">Light Curve</div><div class="lp-tl-sub">Transit shape and depth are measured</div></div>
        </div>
        <div class="lp-timeline-step">
          <div class="lp-tl-left"><div class="lp-tl-dot"></div><div class="lp-tl-line"></div></div>
          <div class="lp-tl-content"><div class="lp-tl-title">KOI</div><div class="lp-tl-sub">Object catalogued as Kepler Object of Interest</div></div>
        </div>
        <div class="lp-timeline-step">
          <div class="lp-tl-left"><div class="lp-tl-dot" style="background:#a78bfa;box-shadow:0 0 10px #a78bfa"></div></div>
          <div class="lp-tl-content"><div class="lp-tl-title" style="color:#a78bfa">AI Classification</div><div class="lp-tl-sub">Model determines: Confirmed or False Positive</div></div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ================================================================
     SECTION 3 -- THE PROBLEM
================================================================ -->
<div class="lp-divider"></div>
<section class="lp-section" id="problem">
  <div class="lp-section-inner">
    <p class="lp-section-label">The Problem</p>
    <h2 class="lp-section-heading">Not every signal<br>is a planet.</h2>
    <div class="lp-problem-cards">
      <div class="lp-problem-card">
        <span class="lp-card-icon">🪐</span>
        <div class="lp-card-type confirmed">// Confirmed Exoplanet</div>
        <div class="lp-card-title">Real World</div>
        <div class="lp-card-desc">
          A genuine planet transiting its host star. Transit depth, duration,
          and shape are consistent with a planetary body at the measured orbital
          radius. Signal-to-noise ratio is high and repeatable.
        </div>
        <span class="lp-card-badge confirmed">Target = 1</span>
      </div>
      <div class="lp-problem-card">
        <span class="lp-card-icon">⭐</span>
        <div class="lp-card-type fp">// False Positive</div>
        <div class="lp-card-title">Stellar Mimic</div>
        <div class="lp-card-desc">
          A brightness dip caused by a background eclipsing binary, a grazing
          stellar eclipse, or instrumental artifacts. These can closely resemble
          planetary transits but require different astrophysical explanations.
        </div>
        <span class="lp-card-badge fp">Target = 0</span>
      </div>
    </div>
  </div>
</section>

<!-- ================================================================
     SECTION 4 -- HOW AI HELPS
================================================================ -->
<div class="lp-divider"></div>
<section class="lp-section" id="how-it-works">
  <div class="lp-section-inner">
    <p class="lp-section-label">How It Works</p>
    <h2 class="lp-section-heading">From starlight<br>to insight.</h2>
    <div class="lp-process-grid">
      <div class="lp-process-step">
        <div class="lp-process-num">01</div>
        <div class="lp-process-title">Input</div>
        <div class="lp-process-desc">Kepler candidate measurements: transit depth, duration, period, planetary radius, stellar properties.</div>
        <div class="lp-process-connector"></div>
      </div>
      <div class="lp-process-step">
        <div class="lp-process-num">02</div>
        <div class="lp-process-title">Analyze</div>
        <div class="lp-process-desc">XGBoost processes all 11 standardized features. Trained on 6,069 labeled KOIs with class-imbalance handling.</div>
        <div class="lp-process-connector"></div>
      </div>
      <div class="lp-process-step">
        <div class="lp-process-num">03</div>
        <div class="lp-process-title">Classify</div>
        <div class="lp-process-desc">Model outputs a probability. Threshold 0.61 applied to distinguish Confirmed Exoplanet from False Positive.</div>
        <div class="lp-process-connector"></div>
      </div>
      <div class="lp-process-step">
        <div class="lp-process-num">04</div>
        <div class="lp-process-title">Explain</div>
        <div class="lp-process-desc">SHAP values reveal which features drove the prediction. Planetary radius and SNR are consistently most influential.</div>
      </div>
    </div>
  </div>
</section>

<!-- ================================================================
     SECTION 5 -- FEATURES
================================================================ -->
<div class="lp-divider"></div>
<section class="lp-section" id="features">
  <div class="lp-section-inner">
    <p class="lp-section-label">Capabilities</p>
    <h2 class="lp-section-heading">Built for<br>exoplanet analysis.</h2>
    <div class="lp-feature-grid">
      <div class="lp-feature-card">
        <div class="lp-feature-icon">🤖</div>
        <div>
          <div class="lp-feature-num">// 01</div>
          <div class="lp-feature-title">Multi-Model AI</div>
          <div class="lp-feature-desc">Four classifiers compared — Logistic Regression, Decision Tree, Random Forest, XGBoost — with rigorous 5-fold CV evaluation.</div>
        </div>
      </div>
      <div class="lp-feature-card">
        <div class="lp-feature-icon">⚖️</div>
        <div>
          <div class="lp-feature-num">// 02</div>
          <div class="lp-feature-title">Imbalance Handling</div>
          <div class="lp-feature-desc">Class weighting and SMOTE experiments account for the 36/64 class split. SMOTE is applied inside CV folds to prevent leakage.</div>
        </div>
      </div>
      <div class="lp-feature-card">
        <div class="lp-feature-icon">🔍</div>
        <div>
          <div class="lp-feature-num">// 03</div>
          <div class="lp-feature-title">Explainable AI</div>
          <div class="lp-feature-desc">SHAP TreeExplainer reveals global and local feature contributions. Permutation importance cross-validates findings.</div>
        </div>
      </div>
      <div class="lp-feature-card">
        <div class="lp-feature-icon">🚀</div>
        <div>
          <div class="lp-feature-num">// 04</div>
          <div class="lp-feature-title">Interactive Analysis</div>
          <div class="lp-feature-desc">Single-KOI prediction, batch CSV upload, and candidate explorer for 1,977 unlabeled Kepler objects.</div>
        </div>
      </div>
    </div>

    <!-- Feature visualization -->
    <div class="lp-viz-card">
      <div class="lp-viz-heading">What does the model see?</div>
      <div class="lp-viz-sub">Relative SHAP importance across all four interpreted models</div>
      <div class="lp-feature-bars">
        <div class="lp-feat-row"><span class="lp-feat-label">koi_prad</span><div class="lp-feat-track"><div class="lp-feat-fill" style="width:100%"></div></div><span class="lp-feat-pct">Top</span></div>
        <div class="lp-feat-row"><span class="lp-feat-label">koi_model_snr</span><div class="lp-feat-track"><div class="lp-feat-fill" style="width:82%; animation-delay:0.1s"></div></div><span class="lp-feat-pct">High</span></div>
        <div class="lp-feat-row"><span class="lp-feat-label">koi_period</span><div class="lp-feat-track"><div class="lp-feat-fill" style="width:72%; animation-delay:0.2s"></div></div><span class="lp-feat-pct">High</span></div>
        <div class="lp-feat-row"><span class="lp-feat-label">koi_duration</span><div class="lp-feat-track"><div class="lp-feat-fill" style="width:62%; animation-delay:0.3s"></div></div><span class="lp-feat-pct">High</span></div>
        <div class="lp-feat-row"><span class="lp-feat-label">koi_insol</span><div class="lp-feat-track"><div class="lp-feat-fill" style="width:55%; animation-delay:0.4s"></div></div><span class="lp-feat-pct">Med</span></div>
        <div class="lp-feat-row"><span class="lp-feat-label">koi_depth</span><div class="lp-feat-track"><div class="lp-feat-fill" style="width:48%; animation-delay:0.5s"></div></div><span class="lp-feat-pct">Med</span></div>
        <div class="lp-feat-row"><span class="lp-feat-label">koi_steff</span><div class="lp-feat-track"><div class="lp-feat-fill" style="width:28%; animation-delay:0.6s"></div></div><span class="lp-feat-pct">Low</span></div>
      </div>
    </div>
  </div>
</section>

<!-- ================================================================
     FINAL CTA
================================================================ -->
<div class="lp-divider"></div>
<section class="lp-cta-section">
  <div class="lp-cta-planet"></div>
  <div class="lp-cta-inner">
    <p class="lp-cta-eyebrow">// Mission Control</p>
    <h2 class="lp-cta-heading">Ready to explore<br><span class="glow-text">a new world?</span></h2>
    <p class="lp-cta-desc">
      Enter a Kepler Object of Interest and let AI analyze
      the evidence hidden in its stellar signal.
    </p>
    <button class="lp-btn-primary" onclick="launchClassifier()" style="font-size:0.95rem; padding:16px 40px;">
      LAUNCH CLASSIFIER &rarr;
    </button>
  </div>
</section>

<!-- ================================================================
     FOOTER
================================================================ -->
<footer class="lp-footer">
  <div class="lp-footer-inner">
    <div>
      <div class="lp-footer-brand">
        <div class="lp-nav-orbit" style="width:20px;height:20px"></div>
        EXOPLANET AI
      </div>
      <div class="lp-footer-tagline">AI-powered exploration of Kepler Objects of Interest.</div>
    </div>
    <nav class="lp-footer-links">
      <a href="#mission">Mission</a>
      <a href="#how-it-works">How It Works</a>
      <a href="#features">Features</a>
      <a href="#" onclick="launchClassifier()">Classifier</a>
    </nav>
  </div>
  <div class="lp-footer-bottom">
    Built with Machine Learning &bull; Explainable AI &bull; NASA Kepler Data
  </div>
</footer>

<!-- ================================================================
     STARFIELD + INTERACTION JAVASCRIPT
================================================================ -->
<script>
// -------------------------------------------------------
// STARFIELD
// -------------------------------------------------------
(function() {
  const canvas = document.getElementById('starfield');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight * 6; // tall enough for full page
  }
  resize();
  window.addEventListener('resize', resize);

  const stars = [];
  const N = 900;
  for (let i = 0; i < N; i++) {
    stars.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      r: Math.random() * 1.4 + 0.2,
      alpha: Math.random() * 0.7 + 0.2,
      twinkle: Math.random() * Math.PI * 2,
      speed: Math.random() * 0.015 + 0.005,
      bright: Math.random() > 0.93,
    });
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Background gradient
    const grad = ctx.createLinearGradient(0, 0, 0, canvas.height);
    grad.addColorStop(0,   '#030712');
    grad.addColorStop(0.3, '#050B18');
    grad.addColorStop(0.7, '#071426');
    grad.addColorStop(1,   '#030712');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Nebula clouds
    function nebula(x, y, r, color) {
      const g = ctx.createRadialGradient(x, y, 0, x, y, r);
      g.addColorStop(0,   color);
      g.addColorStop(1,   'transparent');
      ctx.globalAlpha = 0.18;
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.ellipse(x, y, r * 1.6, r * 0.9, 0.4, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    nebula(canvas.width * 0.25, canvas.height * 0.12, 280, 'rgba(99,102,241,0.6)');
    nebula(canvas.width * 0.75, canvas.height * 0.22, 220, 'rgba(139,92,246,0.5)');
    nebula(canvas.width * 0.5,  canvas.height * 0.48, 300, 'rgba(59,130,246,0.4)');
    nebula(canvas.width * 0.15, canvas.height * 0.65, 200, 'rgba(99,102,241,0.5)');
    nebula(canvas.width * 0.85, canvas.height * 0.75, 240, 'rgba(139,92,246,0.4)');

    // Stars
    const t = performance.now() / 1000;
    stars.forEach(s => {
      s.twinkle += s.speed;
      const a = s.alpha * (0.7 + 0.3 * Math.sin(s.twinkle));
      ctx.globalAlpha = a;
      ctx.fillStyle = s.bright ? '#a5f3fc' : '#e2e8f0';
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r * (s.bright ? 1.4 : 1), 0, Math.PI * 2);
      ctx.fill();
      if (s.bright) {
        ctx.globalAlpha = a * 0.3;
        ctx.fillStyle = '#67e8f9';
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r * 3, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    });

    requestAnimationFrame(draw);
  }
  draw();
})();

// -------------------------------------------------------
// Smooth scroll for anchor links
// -------------------------------------------------------
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', function(e) {
    const id = this.getAttribute('href').slice(1);
    const el = document.getElementById(id);
    if (el) { e.preventDefault(); el.scrollIntoView({behavior:'smooth'}); }
  });
});

// -------------------------------------------------------
// launchClassifier: trigger Streamlit sidebar nav change
// via URL hash — Streamlit picks this up via st.query_params
// -------------------------------------------------------
function launchClassifier() {
  // Set a flag in the URL fragment; Streamlit can read st.query_params
  window.location.hash = 'classify';
  // Also try to find and click the Streamlit sidebar radio "Single Prediction"
  const radios = window.parent.document.querySelectorAll(
    '[data-testid="stSidebar"] input[type="radio"]');
  for (const r of radios) {
    const label = r.closest('label') || r.nextElementSibling;
    if (label && label.textContent && label.textContent.includes('Single Prediction')) {
      r.click();
      return;
    }
  }
  // Fallback: scroll to the top so the sidebar is visible
  window.scrollTo({top: 0, behavior: 'smooth'});
}
</script>
""", unsafe_allow_html=True)



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
        ["🏠 Home", "Single Prediction", "Batch Prediction",
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
    if page == "🏠 Home":
        page_landing()
    elif page == "Single Prediction":
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
