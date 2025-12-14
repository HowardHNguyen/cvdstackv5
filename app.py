import numpy as np
import pandas as pd
import json
import joblib
import streamlit as st
from pathlib import Path

# Optional SHAP imports
try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# =========================
# 1) PAGE CONFIG
# =========================
st.set_page_config(
    page_title="CVD Risk – Stacking GenAI v5.0 (Patient & Clinician)",
    page_icon="🫀",
    layout="wide"
)

# =========================
# 2) LOAD ARTIFACTS
# =========================
DEFAULT_THRESHOLD = 0.40

ARTIFACTS = {
    "scaler": Path("scaler_24.pkl"),
    "rf": Path("rf_clin24.pkl"),
    "xgb": Path("xgb_clin24.pkl"),
    "meta": Path("stack_meta_clin24.pkl"),
    "features": Path("features_24.json"),
}

@st.cache_resource
def load_artifacts():
    missing = [k for k, p in ARTIFACTS.items() if not p.exists()]
    if missing:
        st.error("Missing model files. Please upload all required artifacts.")
        st.stop()

    scaler = joblib.load(ARTIFACTS["scaler"])
    rf_model = joblib.load(ARTIFACTS["rf"])
    xgb_model = joblib.load(ARTIFACTS["xgb"])
    meta_model = joblib.load(ARTIFACTS["meta"])

    with open(ARTIFACTS["features"], "r") as f:
        features_24 = json.load(f)

    return scaler, rf_model, xgb_model, meta_model, features_24

scaler, rf_model, xgb_model, meta_model, FEATURES_24 = load_artifacts()


# =========================
# 3) HELPERS
# =========================
def interpret_risk(prob):
    if prob < 0.05:
        return "Low risk", "🟢"
    if prob < 0.10:
        return "Borderline risk", "🟡"
    if prob < 0.20:
        return "Intermediate risk", "🟠"
    return "High risk", "🔴"

def stacking_predict_proba_24(df_input, threshold):
    Xs = scaler.transform(df_input.values.astype(float))
    p_rf = rf_model.predict_proba(Xs)[:, 1]
    p_xgb = xgb_model.predict_proba(Xs)[:, 1]
    p_final = meta_model.predict_proba(np.column_stack([p_rf, p_xgb]))[:, 1]
    return float(p_final[0]), int(p_final[0] >= threshold), {
        "RF (Clinical)": float(p_rf[0]),
        "XGB (Clinical)": float(p_xgb[0])
    }


# =========================
# 4) SIDEBAR – USER MODE
# =========================
with st.sidebar:
    st.markdown("## 🫀 CVD Stacking GenAI")
    st.caption("v5.0 – Patient & Clinician View")

    user_mode = st.radio(
        "Choose display mode:",
        ["Patient Mode", "Clinician / Research Mode"],
        index=0
    )
    IS_PATIENT_MODE = user_mode == "Patient Mode"

    st.markdown("---")
    threshold = st.slider(
        "Alert threshold",
        0.10, 0.90, DEFAULT_THRESHOLD, 0.05
    )

    if not IS_PATIENT_MODE:
        show_components = st.checkbox("Show RF/XGB probabilities", True)
        show_shap = st.checkbox("Show SHAP plots", False)
    else:
        show_components = False
        show_shap = False

    st.markdown("---")
    st.caption("Research & education only.")


# =========================
# 5) HEADER
# =========================
st.markdown(
    """
    <div style="background:#0f4c75;padding:18px;border-radius:8px;">
    <h1 style="color:white;">CVD Risk Prediction</h1>
    <p style="color:#e0f2f1;">
    10-Year Cardiovascular Risk Estimation (Stacking GenAI v5.0)
    </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# 6) INPUT FORM (UNCHANGED)
# =========================
st.markdown("### Enter your health information")
df_input = build_input_df_24()

run = st.button("Run Risk Prediction", type="primary")

if run:
    prob, label, components = stacking_predict_proba_24(df_input, threshold)
    category, color = interpret_risk(prob)

    st.markdown("## Your Results")

    if IS_PATIENT_MODE:
        st.metric(
            "Your estimated 10-year heart disease risk",
            f"{prob*100:.1f}%"
        )
        st.markdown(
            f"""
            This means about **{int(round(prob*100))} out of 100 people**
            with similar health profiles may develop heart disease over
            the next 10 years.
            """
        )
        st.markdown(f"**Risk level:** {color} {category}")

    else:
        st.metric("Estimated 10-year CVD risk", f"{prob*100:.2f}%")
        st.markdown(f"**Risk category:** {color} **{category}**")
        st.markdown(
            f"**Model decision:** {'At Risk' if label else 'Not Flagged'}"
        )

    # =========================
    # SHAP – PATIENT SUMMARY
    # =========================
    if IS_PATIENT_MODE:
        st.markdown("### Why the model rated your risk this way")

        rf_exp, xgb_exp = None, None
        if SHAP_AVAILABLE:
            rf_exp = shap.TreeExplainer(rf_model)
            xgb_exp = shap.TreeExplainer(xgb_model)

            Xs = scaler.transform(df_input.values.astype(float))
            sv = xgb_exp.shap_values(Xs)
            if isinstance(sv, list):
                sv = sv[1]
            vals = sv[0]

            idx = np.argsort(np.abs(vals))[::-1][:5]

            FRIENDLY = {
                "SYSBP": "High blood pressure",
                "AGE": "Age",
                "SEX": "Male sex",
                "CIGPDAY": "Smoking",
                "GLUCOSE": "Blood sugar",
                "PREVMI": "Past heart attack",
                "STROKE": "Past stroke"
            }

            positives = []
            negatives = []

            for i in idx:
                name = FEATURES_24[i]
                label_txt = FRIENDLY.get(name, name)
                if vals[i] > 0:
                    positives.append(label_txt)
                else:
                    negatives.append(label_txt)

            if positives:
                st.markdown("**What increases your risk:**")
                for p in positives[:3]:
                    st.markdown(f"- {p}")

            if negatives:
                st.markdown("**What helps lower your risk:**")
                for n in negatives[:3]:
                    st.markdown(f"- {n}")

        st.caption(
            "These factors influenced your score compared with an average person "
            "in the study. This is not medical advice."
        )

    # =========================
    # SHAP – FULL (CLINICIAN)
    # =========================
    if not IS_PATIENT_MODE and show_shap and SHAP_AVAILABLE:
        st.markdown("### Local SHAP explanation (RF & XGB)")
        Xs = scaler.transform(df_input.values.astype(float))
        rf_exp = shap.TreeExplainer(rf_model)
        xgb_exp = shap.TreeExplainer(xgb_model)

        st.markdown("**Random Forest**")
        _shap_local_bar(rf_exp, Xs, FEATURES_24, "RF SHAP")

        st.markdown("**XGBoost**")
        _shap_local_bar(xgb_exp, Xs, FEATURES_24, "XGB SHAP")

    # =========================
    # BIE – PATIENT FRIENDLY
    # =========================
    st.markdown("---")
    st.markdown(
        "## What-If Simulator (Research-Based)"
        if IS_PATIENT_MODE else
        "## Behavioral Impact Engine (BIE)"
    )

    base_prob, table, best = bie_scenarios_24(
        df_input,
        threshold,
        include_advanced=not IS_PATIENT_MODE
    )

    if IS_PATIENT_MODE:
        table = table.rename(columns={
            "Δ abs (pp)": "Change in risk",
            "Δ rel (%)": "Relative change"
        })

    st.dataframe(table, use_container_width=True)

    if IS_PATIENT_MODE and best:
        st.success(
            f"""
            **Most helpful next step (based on this model):**
            {best['name']}

            Estimated change:
            **{base_prob*100:.1f}% → {best['p']*100:.1f}%**
            """
        )

st.markdown(
    """
    <hr>
    <div style="text-align:center;font-size:12px;color:gray;">
    Research & Education Only • Not medical advice<br>
    © 2025 Howard Nguyen, PhD
    </div>
    """,
    unsafe_allow_html=True
)
