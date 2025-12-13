import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from pathlib import Path

# =========================
# Page config + styling
# =========================
st.set_page_config(page_title="CVD Risk (v5.0)", page_icon="🫀", layout="wide")

def header(title, subtitle):
    st.markdown(
        f"""
        <div style="padding:16px 18px;border-radius:10px;background:#0f4c75;margin-bottom:14px;">
          <div style="font-size:28px;font-weight:800;color:white;line-height:1.15;">{title}</div>
          <div style="font-size:14px;color:#e0f2f1;margin-top:6px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def footer():
    st.markdown(
        """
        <hr style="margin-top:24px;margin-bottom:10px;">
        <div style="color:gray;font-size:12px;">
          © CVDStack • v5.0 • Research/Clinical Decision Support Prototype • Not medical advice.
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# Load artifacts
# =========================
BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE / "models"

@st.cache_resource
def load_artifacts():
    # Required
    scaler = joblib.load(MODELS_DIR / "scaler_24.pkl")

    # Choose ONE deployment mode:
    # (A) single best model
    # model = joblib.load(MODELS_DIR / "gbm_24.pkl")
    # model_type = "single"

    # (B) stacking (recommended if you trained it)
    rf = joblib.load(MODELS_DIR / "rf_24.pkl")
    xgb = joblib.load(MODELS_DIR / "xgb_24.pkl")
    meta = joblib.load(MODELS_DIR / "stack_meta_24.pkl")
    model_type = "stacking"

    # Feature list (preferred)
    feat_path = MODELS_DIR / "feature_list_24.json"
    if feat_path.exists():
        feature_list = json.loads(feat_path.read_text())["features"]
    else:
        # fallback hard-coded (still okay)
        feature_list = [
            'SEX', 'TOTCHOL', 'AGE', 'SYSBP', 'DIABP', 'CIGPDAY', 'BMI', 'DIABETES', 'BPMEDS',
            'HEARTRTE', 'GLUCOSE', 'educ', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK',
            'PREVHYP', 'HOSPMI', 'HDLC', 'LDLC', 'ANGINA', 'MI_FCHD', 'STROKE', 'HYPERTEN'
        ]

    return scaler, model_type, rf, xgb, meta, feature_list

scaler, model_type, rf, xgb, meta, FEATURES = load_artifacts()

# =========================
# Feature metadata (UI)
# =========================
FEATURE_META = {
    "SEX": {"label":"Sex", "type":"cat01", "help":"0=female, 1=male"},
    "educ": {"label":"Education level", "type":"int", "min":0, "max":4, "default":1},
    "DIABETES": {"label":"Diabetes", "type":"cat01"},
    "BPMEDS": {"label":"On BP meds", "type":"cat01"},
    "HYPERTEN": {"label":"Hypertension (flag)", "type":"cat01"},
    "ANGINA": {"label":"Angina history", "type":"cat01"},
    "MI_FCHD": {"label":"MI / FHCHD history", "type":"cat01"},
    "STROKE": {"label":"Stroke history", "type":"cat01"},
    "PREVCHD": {"label":"Prevalent CHD", "type":"cat01"},
    "PREVAP": {"label":"Prevalent angina pectoris", "type":"cat01"},
    "PREVMI": {"label":"Previous MI", "type":"cat01"},
    "PREVSTRK": {"label":"Previous stroke", "type":"cat01"},
    "PREVHYP": {"label":"Previous hypertension", "type":"cat01"},
    "HOSPMI": {"label":"Hospitalized MI", "type":"cat01"},
    # continuous
    "AGE": {"label":"Age", "type":"float", "min":18, "max":95, "default":55},
    "SYSBP": {"label":"Systolic BP", "type":"float", "min":70, "max":240, "default":130},
    "DIABP": {"label":"Diastolic BP", "type":"float", "min":40, "max":140, "default":80},
    "TOTCHOL": {"label":"Total Cholesterol", "type":"float", "min":80, "max":500, "default":190},
    "HDLC": {"label":"HDL", "type":"float", "min":10, "max":150, "default":45},
    "LDLC": {"label":"LDL", "type":"float", "min":10, "max":300, "default":120},
    "GLUCOSE": {"label":"Glucose", "type":"float", "min":40, "max":400, "default":100},
    "BMI": {"label":"BMI", "type":"float", "min":12, "max":60, "default":27},
    "HEARTRTE": {"label":"Heart rate", "type":"float", "min":30, "max":200, "default":72},
    "CIGPDAY": {"label":"Cigarettes per day", "type":"float", "min":0, "max":80, "default":0},
}

HISTORY_FLAGS = ["ANGINA","MI_FCHD","STROKE","PREVCHD","PREVAP","PREVMI","PREVSTRK","PREVHYP","HOSPMI"]

def default_state():
    s = {}
    for f in FEATURES:
        meta = FEATURE_META.get(f, {"type":"float", "default":0})
        if meta["type"] in ["cat01"]:
            s[f] = 0
        else:
            s[f] = meta.get("default", 0)
    return s

if "inputs_v5" not in st.session_state:
    st.session_state.inputs_v5 = default_state()

# =========================
# Model inference
# =========================
def make_row(inputs: dict) -> pd.DataFrame:
    row = {f: inputs.get(f, 0) for f in FEATURES}
    return pd.DataFrame([row], columns=FEATURES)

def predict_risk_proba(X_df: pd.DataFrame) -> float:
    X_scaled = scaler.transform(X_df.values)

    if model_type == "single":
        # return float(model.predict_proba(X_scaled)[:,1][0])
        raise RuntimeError("Single-model mode not enabled in this template.")

    # stacking (rf + xgb -> meta LR)
    p_rf = rf.predict_proba(X_scaled)[:, 1]
    p_xgb = xgb.predict_proba(X_scaled)[:, 1]
    stack = np.vstack([p_rf, p_xgb]).T
    p = meta.predict_proba(stack)[:, 1]
    return float(p[0])

def risk_bucket(p):
    if p < 0.05: return "Low"
    if p < 0.075: return "Borderline"
    if p < 0.20: return "Intermediate"
    return "High"

# =========================
# Sidebar
# =========================
st.sidebar.title("🧭 Navigation")
tab = st.sidebar.radio(
    "Go to",
    ["🧮 Risk Calculator", "🧠 Behavioral Impact Engine (BIE)", "🧬 Model & Data", "❓ FAQ & Notes"],
)

st.sidebar.markdown("---")
use_history = st.sidebar.checkbox("Include clinical history inputs", value=True)

# If user turns OFF history in UI, zero them out for inference (still keeps 24-feature vector consistent)
def apply_history_toggle(inputs):
    if use_history:
        return inputs
    inputs2 = dict(inputs)
    for k in HISTORY_FLAGS:
        if k in inputs2:
            inputs2[k] = 0
    return inputs2

# =========================
# Header
# =========================
header(
    "CVD Risk Prediction – Enhanced Clinical History Model (v5.0)",
    "10-Year Cardiovascular Risk Estimation with Extended History & Comorbidity Features (Research / CDS Support)"
)

# Safety note about inflated performance
if use_history:
    st.warning(
        "Clinical history variables can heavily influence predictions and may inflate apparent accuracy if not strictly time-indexed "
        "(pre-index only). For real EHR deployment, ensure all history features are defined prior to the prediction date."
    )

# =========================
# UI controls builder
# =========================
def render_inputs(panel_key_prefix="main"):
    cols = st.columns(3)
    for i, f in enumerate(FEATURES):
        meta = FEATURE_META.get(f, {"label":f, "type":"float", "default":0})
        label = meta.get("label", f)
        t = meta.get("type","float")

        with cols[i % 3]:
            key = f"{panel_key_prefix}_{f}"
            if t == "cat01":
                st.session_state.inputs_v5[f] = st.selectbox(
                    label, [0, 1],
                    index=int(st.session_state.inputs_v5.get(f, 0)),
                    key=key,
                    help=meta.get("help","0=No, 1=Yes")
                )
            elif t == "int":
                st.session_state.inputs_v5[f] = st.number_input(
                    label,
                    min_value=int(meta.get("min", 0)),
                    max_value=int(meta.get("max", 999)),
                    value=int(st.session_state.inputs_v5.get(f, meta.get("default", 0))),
                    step=1,
                    key=key
                )
            else:
                st.session_state.inputs_v5[f] = st.number_input(
                    label,
                    min_value=float(meta.get("min", -1e9)),
                    max_value=float(meta.get("max", 1e9)),
                    value=float(st.session_state.inputs_v5.get(f, meta.get("default", 0.0))),
                    step=1.0,
                    key=key
                )

# =========================
# Tabs
# =========================
if tab == "🧮 Risk Calculator":
    st.subheader("🧮 Risk Calculator")
    st.caption("Enter patient factors below. The model returns a 10-year CVD risk probability.")

    with st.expander("Patient Inputs", expanded=True):
        render_inputs("calc")

    inputs = apply_history_toggle(st.session_state.inputs_v5)
    X = make_row(inputs)
    p = predict_risk_proba(X)

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        st.metric("Estimated 10-Year CVD Risk", f"{p*100:.1f}%")
    with c2:
        st.metric("Risk Category", risk_bucket(p))
    with c3:
        st.metric("Model Version", "v5.0 (24 features)")

elif tab == "🧠 Behavioral Impact Engine (BIE)":
    st.subheader("🧠 Behavioral Impact Engine (BIE)")
    st.caption("Counterfactual scenarios show how risk changes if a single factor is modified.")

    base_inputs = apply_history_toggle(st.session_state.inputs_v5)
    X0 = make_row(base_inputs)
    p0 = predict_risk_proba(X0)

    st.write(f"**Baseline Risk:** {p0*100:.1f}% ({risk_bucket(p0)})")

    # Simple scenarios (edit these to match what you used in v3.0)
    scenarios = []

    # Smoking scenario
    if "CIGPDAY" in FEATURES:
        s = dict(base_inputs)
        s["CIGPDAY"] = 0
        p1 = predict_risk_proba(make_row(s))
        scenarios.append(("Set cigarettes/day to 0", p1))

    # SBP scenario
    if "SYSBP" in FEATURES:
        s = dict(base_inputs)
        s["SYSBP"] = max(80, float(s["SYSBP"]) - 10)
        p1 = predict_risk_proba(make_row(s))
        scenarios.append(("Lower SBP by 10 mmHg", p1))

    # LDL scenario
    if "LDLC" in FEATURES:
        s = dict(base_inputs)
        s["LDLC"] = max(10, float(s["LDLC"]) - 30)
        p1 = predict_risk_proba(make_row(s))
        scenarios.append(("Lower LDL by 30 mg/dL", p1))

    # BMI scenario
    if "BMI" in FEATURES:
        s = dict(base_inputs)
        s["BMI"] = max(12, float(s["BMI"]) - 2)
        p1 = predict_risk_proba(make_row(s))
        scenarios.append(("Reduce BMI by 2 points", p1))

    rows = []
    for name, p1 in scenarios:
        abs_drop = (p0 - p1)
        rel_drop = (abs_drop / p0) if p0 > 1e-9 else 0.0
        rows.append([name, p0*100, p1*100, abs_drop*100, rel_drop*100])

    df = pd.DataFrame(rows, columns=["Scenario", "Baseline (%)", "New Risk (%)", "Δ Abs (pp)", "Δ Rel (%)"])
    df = df.sort_values("Δ Abs (pp)", ascending=False)

    st.dataframe(df, use_container_width=True)

elif tab == "🧬 Model & Data":
    st.subheader("🧬 Model & Data")
    st.markdown(
        """
- **Model:** Stacking ensemble (RF + XGB → Logistic Regression meta-learner)  
- **Output:** Probability of CVD event within 10 years (model-based estimate)  
- **Feature set:** 24 inputs (includes extended clinical history variables)  
- **Recommended:** Site-specific temporal validation + calibration before EHR deployment
        """
    )
    st.markdown("**Features (24):**")
    st.code(", ".join(FEATURES))

elif tab == "❓ FAQ & Notes":
    st.subheader("❓ FAQ & Notes")
    st.markdown(
        """
### What does “10-year CVD risk” mean?
The **10-year CVD risk** represents the **estimated probability** that an individual will experience a cardiovascular event within the next 10 years, based on their current risk profile.

### How should I interpret the percentage?
A predicted risk of **15%** means that, among **100 people with similar characteristics**, about **15 may experience a CVD event** within 10 years. This is a probabilistic estimate—not a diagnosis.

### Important limitations
- This model reflects patterns learned from the training population and may not generalize without local validation.
- Clinical history variables can dominate predictions and may inflate apparent performance if not strictly pre-index.
- Outputs are intended for research/education/CDS support and do not replace clinical judgment.
        """
    )

footer()
