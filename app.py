
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# =========================================================
# v5.0 — 24-Feature Stacking (RF + XGB + Meta LR)
# Root-level artifacts (same folder as app.py)
# =========================================================

APP_VERSION = "v5.0"
ARTIFACTS = {
    "feature_list": "feature_list_24.json",
    "scaler": "scaler_24.pkl",
    "rf": "rf_24.pkl",
    "xgb": "xgb_24.pkl",
    "meta": "stack_meta_24.pkl",
}

HISTORY_FLAGS = [
    "PREVCHD", "PREVAP", "PREVMI", "PREVSTRK", "PREVHYP", "HOSPMI",
    "ANGINA", "MI_FCHD", "STROKE", "HYPERTEN",
]

st.set_page_config(
    page_title=f"CVD Risk Prediction {APP_VERSION}",
    page_icon="🫀",
    layout="wide",
)

def _header():
    st.markdown(
        f"""
        <div style="padding:18px 18px;border-radius:14px;background:linear-gradient(90deg,#0f4c75,#1b6ca8);">
          <div style="font-size:30px;font-weight:850;color:white;line-height:1.15;">
            🫀 CVD Risk Prediction — Enhanced Clinical History Model ({APP_VERSION})
          </div>
          <div style="font-size:14px;color:#e8f6ff;margin-top:6px;">
            10‑Year Cardiovascular Disease (CVD) Risk Estimation • 24 inputs • Stacking (RF + XGB → Meta‑Learner)
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

def _footer():
    st.markdown(
        """
        <hr style="margin-top:24px;margin-bottom:10px;">
        <div style="color:gray;font-size:12px;">
          © CVDStack • Prototype CDS • Not medical advice • Use requires local validation & governance approval.
        </div>
        """,
        unsafe_allow_html=True,
    )

@st.cache_resource
def load_artifacts():
    base = Path(__file__).resolve().parent
    missing = [fn for fn in ARTIFACTS.values() if not (base / fn).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required deployment files in repo root:\n"
            + "\n".join([f" - {m}" for m in missing])
            + "\n\nExpected all artifacts to be in the SAME folder as app.py."
        )

    with open(base / ARTIFACTS["feature_list"], "r") as f:
        features = json.load(f)["features"]

    scaler = joblib.load(base / ARTIFACTS["scaler"])
    rf = joblib.load(base / ARTIFACTS["rf"])
    xgb = joblib.load(base / ARTIFACTS["xgb"])
    meta = joblib.load(base / ARTIFACTS["meta"])
    return features, scaler, rf, xgb, meta

FEATURE_META = {
    "SEX":       {"label": "Sex", "type": "cat01", "help": "0 = Female, 1 = Male"},
    "DIABETES":  {"label": "Diabetes", "type": "cat01"},
    "BPMEDS":    {"label": "BP meds", "type": "cat01"},
    "HYPERTEN":  {"label": "Hypertension (flag)", "type": "cat01"},
    "ANGINA":    {"label": "Angina history", "type": "cat01"},
    "MI_FCHD":   {"label": "MI / FHCHD history", "type": "cat01"},
    "STROKE":    {"label": "Stroke history", "type": "cat01"},
    "PREVCHD":   {"label": "Prevalent CHD", "type": "cat01"},
    "PREVAP":    {"label": "Prevalent angina pectoris", "type": "cat01"},
    "PREVMI":    {"label": "Previous MI", "type": "cat01"},
    "PREVSTRK":  {"label": "Previous stroke", "type": "cat01"},
    "PREVHYP":   {"label": "Previous hypertension", "type": "cat01"},
    "HOSPMI":    {"label": "Hospitalized MI", "type": "cat01"},
    "educ":      {"label": "Education (educ)", "type": "int", "min": 0, "max": 4, "default": 1},
    "AGE":       {"label": "Age (years)", "type": "float", "min": 18, "max": 95, "default": 55},
    "SYSBP":     {"label": "Systolic BP (mmHg)", "type": "float", "min": 70, "max": 240, "default": 130},
    "DIABP":     {"label": "Diastolic BP (mmHg)", "type": "float", "min": 40, "max": 140, "default": 80},
    "TOTCHOL":   {"label": "Total Cholesterol (mg/dL)", "type": "float", "min": 80, "max": 500, "default": 190},
    "HDLC":      {"label": "HDL (mg/dL)", "type": "float", "min": 10, "max": 150, "default": 45},
    "LDLC":      {"label": "LDL (mg/dL)", "type": "float", "min": 10, "max": 300, "default": 120},
    "GLUCOSE":   {"label": "Glucose (mg/dL)", "type": "float", "min": 40, "max": 400, "default": 100},
    "BMI":       {"label": "BMI", "type": "float", "min": 12, "max": 60, "default": 27},
    "HEARTRTE":  {"label": "Heart Rate (bpm)", "type": "float", "min": 30, "max": 200, "default": 72},
    "CIGPDAY":   {"label": "Cigarettes per day", "type": "float", "min": 0, "max": 80, "default": 0},
}

def risk_bucket(p: float) -> str:
    if p < 0.05:
        return "Low"
    if p < 0.075:
        return "Borderline"
    if p < 0.20:
        return "Intermediate"
    return "High"

def build_default_inputs(features):
    inputs = {}
    for f in features:
        meta = FEATURE_META.get(f, {"type": "float", "default": 0.0})
        if meta.get("type") == "cat01":
            inputs[f] = 0
        elif meta.get("type") == "int":
            inputs[f] = int(meta.get("default", 0))
        else:
            inputs[f] = float(meta.get("default", 0.0))
    return inputs

def make_row(inputs: dict, features: list[str]) -> pd.DataFrame:
    row = {f: inputs.get(f, 0) for f in features}
    return pd.DataFrame([row], columns=features)

def predict_proba(inputs: dict, features, scaler, rf, xgb, meta) -> float:
    X = make_row(inputs, features)
    Xs = scaler.transform(X.values)
    p_rf = rf.predict_proba(Xs)[:, 1]
    p_xgb = xgb.predict_proba(Xs)[:, 1]
    stack = np.vstack([p_rf, p_xgb]).T
    p = meta.predict_proba(stack)[:, 1]
    return float(p[0])

def apply_history_toggle(inputs: dict, use_history: bool, features: list[str]) -> dict:
    if use_history:
        return inputs
    out = dict(inputs)
    for k in HISTORY_FLAGS:
        if k in features:
            out[k] = 0
    return out

try:
    FEATURES, scaler, rf, xgb, meta = load_artifacts()
except Exception as e:
    st.error("Model artifacts failed to load.")
    st.exception(e)
    st.stop()

if "v5_inputs" not in st.session_state:
    st.session_state.v5_inputs = build_default_inputs(FEATURES)

if "v5_bie" not in st.session_state:
    st.session_state.v5_bie = build_default_inputs(FEATURES)

_header()

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🧮 Risk Calculator", "🧠 Behavioral Impact Engine (BIE)", "🧬 Model & Data", "❓ FAQ & Notes"],
    key="nav_v5",
)
st.sidebar.markdown("---")
use_history = st.sidebar.checkbox("Include clinical history variables", value=True, key="hist_toggle_v5")

if use_history:
    st.warning(
        "History variables (e.g., prior MI/stroke/angina) can dominate predictions. "
        "For EHR deployment, ensure all history features are defined strictly prior to the prediction index date."
    )

def render_inputs(state_key: str, prefix: str, show_history: bool):
    inputs = st.session_state[state_key]
    cols = st.columns(3)

    for i, f in enumerate(FEATURES):
        if (not show_history) and (f in HISTORY_FLAGS):
            continue

        meta_f = FEATURE_META.get(f, {"label": f, "type": "float"})
        label = meta_f.get("label", f)
        ftype = meta_f.get("type", "float")
        help_txt = meta_f.get("help", "0 = No, 1 = Yes" if ftype == "cat01" else "")

        with cols[i % 3]:
            widget_key = f"{prefix}__{f}"
            if ftype == "cat01":
                cur = int(inputs.get(f, 0))
                cur = 1 if cur == 1 else 0
                inputs[f] = st.selectbox(label, [0, 1], index=cur, key=widget_key, help=help_txt)
            elif ftype == "int":
                inputs[f] = st.number_input(
                    label,
                    min_value=int(meta_f.get("min", 0)),
                    max_value=int(meta_f.get("max", 999)),
                    value=int(inputs.get(f, meta_f.get("default", 0))),
                    step=1,
                    key=widget_key,
                )
            else:
                inputs[f] = st.number_input(
                    label,
                    min_value=float(meta_f.get("min", -1e9)),
                    max_value=float(meta_f.get("max", 1e9)),
                    value=float(inputs.get(f, meta_f.get("default", inputs.get(f, 0.0)))),
                    step=1.0,
                    key=widget_key,
                )

    st.session_state[state_key] = inputs

if page == "🧮 Risk Calculator":
    st.subheader("🧮 Risk Calculator")
    st.caption("Enter patient factors. The model returns an estimated probability of a CVD event within 10 years.")

    with st.expander("Patient Inputs", expanded=True):
        render_inputs("v5_inputs", prefix="calc", show_history=use_history)

    inputs = apply_history_toggle(st.session_state.v5_inputs, use_history, FEATURES)
    proba = predict_proba(inputs, FEATURES, scaler, rf, xgb, meta)

    c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
    with c1:
        st.metric("Estimated 10‑Year CVD Risk", f"{proba*100:.1f}%")
    with c2:
        st.metric("Risk Category", risk_bucket(proba))
    with c3:
        st.metric("Model", f"Stacking (RF + XGB) • {APP_VERSION}")

    st.markdown("---")
    st.caption("Tip: For prevention-style interpretation, turn OFF clinical history in the sidebar toggle.")

elif page == "🧠 Behavioral Impact Engine (BIE)":
    st.subheader("🧠 Behavioral Impact Engine (BIE)")
    st.caption(
        "The BIE runs counterfactual scenarios and estimates how risk may change if a single factor is modified. "
        "This is educational decision support—not a prescription."
    )

    with st.expander("BIE Patient Inputs", expanded=True):
        render_inputs("v5_bie", prefix="bie", show_history=use_history)

    base_inputs = apply_history_toggle(st.session_state.v5_bie, use_history, FEATURES)
    p0 = predict_proba(base_inputs, FEATURES, scaler, rf, xgb, meta)

    st.write(f"**Baseline Risk:** {p0*100:.1f}%  —  **{risk_bucket(p0)}**")

    scenarios = []

    if "CIGPDAY" in FEATURES:
        s = dict(base_inputs)
        s["CIGPDAY"] = 0
        scenarios.append(("No smoking (CIGPDAY = 0)", predict_proba(s, FEATURES, scaler, rf, xgb, meta)))

    if "SYSBP" in FEATURES:
        s = dict(base_inputs)
        s["SYSBP"] = max(70.0, float(s.get("SYSBP", 0.0)) - 10.0)
        scenarios.append(("Lower SBP by 10 mmHg", predict_proba(s, FEATURES, scaler, rf, xgb, meta)))

    if "LDLC" in FEATURES:
        s = dict(base_inputs)
        s["LDLC"] = max(10.0, float(s.get("LDLC", 0.0)) - 30.0)
        scenarios.append(("Lower LDL by 30 mg/dL", predict_proba(s, FEATURES, scaler, rf, xgb, meta)))

    if "TOTCHOL" in FEATURES:
        s = dict(base_inputs)
        s["TOTCHOL"] = max(80.0, float(s.get("TOTCHOL", 0.0)) - 20.0)
        scenarios.append(("Lower Total Chol by 20 mg/dL", predict_proba(s, FEATURES, scaler, rf, xgb, meta)))

    if "BMI" in FEATURES:
        s = dict(base_inputs)
        s["BMI"] = max(12.0, float(s.get("BMI", 0.0)) - 2.0)
        scenarios.append(("Reduce BMI by 2 points", predict_proba(s, FEATURES, scaler, rf, xgb, meta)))

    if "GLUCOSE" in FEATURES:
        s = dict(base_inputs)
        s["GLUCOSE"] = max(40.0, float(s.get("GLUCOSE", 0.0)) - 15.0)
        scenarios.append(("Lower Glucose by 15 mg/dL", predict_proba(s, FEATURES, scaler, rf, xgb, meta)))

    rows = []
    for name, p1 in scenarios:
        abs_drop = (p0 - p1) * 100.0
        rel_drop = ((p0 - p1) / p0 * 100.0) if p0 > 1e-9 else 0.0
        rows.append([name, p0 * 100.0, p1 * 100.0, abs_drop, rel_drop])

    df = pd.DataFrame(rows, columns=[
        "Scenario",
        "Baseline Risk (%)",
        "New Risk (%)",
        "Δ Absolute (pp) ↓",
        "Δ Relative (%) ↓",
    ]).sort_values("Δ Absolute (pp) ↓", ascending=False)

    st.dataframe(df, use_container_width=True)

    if len(df) > 0:
        top = df.iloc[0]
        st.success(
            f"Most impactful scenario (within this simple set): **{top['Scenario']}** "
            f"→ **{top['Baseline Risk (%)']:.1f}%** to **{top['New Risk (%)']:.1f}%** "
            f"(drop **{top['Δ Absolute (pp) ↓']:.1f} pp**, **{top['Δ Relative (%) ↓']:.0f}%** relative)."
        )

    st.markdown("### Evidence‑based guidance (template)")
    st.markdown(
        """
- **Smoking cessation:** counseling + pharmacotherapy options per guideline and shared decision-making  
- **Blood pressure control:** confirm measurements, home BP, medication optimization if indicated  
- **Lipids:** evaluate ASCVD risk context, consider statin intensity as appropriate  
- **Diabetes / glucose:** lifestyle + medication management per guidelines  
- **Weight:** nutrition, activity, and structured weight-loss support where appropriate  

*Localize this section to your hospital’s clinical pathways.*
        """
    )

elif page == "🧬 Model & Data":
    st.subheader("🧬 Model & Data")
    st.markdown(
        f"""
**Model (deployment):** Stacking ensemble  
- Base learners: **Random Forest** + **XGBoost**  
- Meta‑learner: **Logistic Regression**  
- Output: **probability** of 10‑year CVD event (model‑based estimate)

**Feature set:** 24 inputs (includes clinical history variables)  
**Recommended use:** clinical decision support prototype with local validation + calibration

**Important note on history variables:**  
Prior diagnoses/events (e.g., prior MI/stroke/angina/hospitalized MI) may strongly increase risk and can
inflate apparent performance if not time-indexed properly. For EHR use, define an **index date** and ensure all history features occur **before** that date.
        """
    )

    st.markdown("**Deployed feature order (must match training):**")
    st.code(", ".join(FEATURES))

    st.markdown("**Artifacts expected in repo root:**")
    st.code("\n".join([ARTIFACTS[k] for k in ARTIFACTS]))

else:
    st.subheader("❓ FAQ & Notes")
    st.markdown(
        """
### What is 10‑year cardiovascular disease (CVD) risk estimation?
A **10‑year CVD risk estimate** is a **probability** that a person will experience a cardiovascular event within the next **10 years**, based on their current profile.
It is **not** a diagnosis and does not guarantee that an event will or will not occur.

### How should I interpret the percentage?
A predicted risk of **15%** means that among **100 people with similar characteristics**, about **15 may experience** a CVD event within 10 years (under similar conditions).

### Why might some factors appear to have small impact?
Two reasons are common:
1) **Nonlinear interactions:** models can learn complex combinations where one factor matters only in certain contexts.  
2) **History variables dominate:** if prior-event features are present (e.g., prior MI/stroke/angina), they often outweigh lifestyle factors in risk prediction.

### Why can some factors behave “oddly” in a single profile?
Machine learning models are not guaranteed to be monotonic unless explicitly constrained and trained on strictly pre-index features.
If you want monotonic behavior for selected variables, consider:
- monotonic constraints (XGBoost),
- probability calibration,
- time-indexed feature definitions for EHR integration.

### Is this hospital/EHR ready?
This app is a **prototype CDS**. Hospital readiness typically requires:
- local retrospective validation and calibration,
- subgroup performance checks,
- documentation (model card),
- monitoring plan,
- and governance approvals before clinician-facing deployment.
        """
    )

_footer()
