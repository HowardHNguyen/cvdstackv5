import numpy as np
import pandas as pd
import json
import joblib
import streamlit as st
from pathlib import Path

# =========================
# 1) PAGE CONFIG
# =========================
st.set_page_config(
    page_title="CVD Risk – Stacking GenAI v5.0 (24 Features Clinical+History)",
    page_icon="🫀",
    layout="wide"
)

# =========================
# 2) LOAD ARTIFACTS (24-feature clinical+history)
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
        st.error(
            "Missing required model files:\n\n"
            + "\n".join([f"- {k}: {ARTIFACTS[k]}" for k in missing])
            + "\n\nUpload these into the same repo folder as app.py."
        )
        st.stop()

    scaler = joblib.load(ARTIFACTS["scaler"])
    rf_model = joblib.load(ARTIFACTS["rf"])
    xgb_model = joblib.load(ARTIFACTS["xgb"])
    meta_model = joblib.load(ARTIFACTS["meta"])

    with open(ARTIFACTS["features"], "r") as f:
        features_24 = json.load(f)

    # Safety checks
    if not isinstance(features_24, list) or len(features_24) != 24:
        st.error("features_24.json must be a JSON list of exactly 24 feature names.")
        st.stop()

    return scaler, rf_model, xgb_model, meta_model, features_24

scaler, rf_model, xgb_model, meta_model, FEATURES_24 = load_artifacts()

# =========================
# 3) HELPERS
# =========================
def interpret_risk(prob: float):
    if prob < 0.05:
        return "Low risk", "🟢"
    if prob < 0.10:
        return "Borderline risk", "🟡"
    if prob < 0.20:
        return "Intermediate risk", "🟠"
    return "High risk", "🔴"

def _as_int_yesno(v: str) -> int:
    return 1 if v == "Yes" else 0

def build_input_df_24():
    """
    Render UI for 24 features and return a single-row DataFrame.
    IMPORTANT: final ordering MUST match FEATURES_24 (training order).
    """
    st.subheader("Patient Profile, Clinical Risk Factors & Prior History (24 Features)")

    col1, col2, col3 = st.columns(3)

    # --- Column 1 (core + lifestyle + meds)
    with col1:
        age = st.number_input("Age (years)", 20, 95, 55, 1, key="calc_age")
        sex = st.selectbox("Sex", ["Female", "Male"], index=0, key="calc_sex")  # SEX: 0=f, 1=m
        bmi = st.number_input("BMI (kg/m²)", 15.0, 60.0, 27.0, 0.1, key="calc_bmi")
        cigs = st.number_input("Cigarettes per day", 0.0, 80.0, 10.0, 1.0, key="calc_cigs")
        diabetes = st.selectbox("Diabetes", ["No", "Yes"], index=0, key="calc_diabetes")
        bpmeds = st.selectbox("On BP medications", ["No", "Yes"], index=0, key="calc_bpmeds")

        educ = st.selectbox(
            "Education level",
            [
                "1 – Some High School",
                "2 – High School/GED",
                "3 – Some College/Vocational",
                "4 – College Graduate+",
            ],
            index=3,
            key="calc_educ"
        )

    # --- Column 2 (BP + metabolic/cardiac)
    with col2:
        sysbp = st.number_input("Systolic BP (mmHg)", 80.0, 260.0, 130.0, 1.0, key="calc_sysbp")
        diabp = st.number_input("Diastolic BP (mmHg)", 40.0, 150.0, 80.0, 1.0, key="calc_diabp")
        heartrate = st.number_input("Heart rate (bpm)", 40.0, 150.0, 72.0, 1.0, key="calc_hr")
        glucose = st.number_input("Fasting Glucose (mg/dL)", 60.0, 400.0, 100.0, 1.0, key="calc_glucose")
        hyperten = st.selectbox("Hypertension diagnosed (HYPERTEN)", ["No", "Yes"], index=0, key="calc_hyperten")
        prevhyp = st.selectbox("Prior hypertension history (PREVHYP)", ["No", "Yes"], index=0, key="calc_prevhyp")

    # --- Column 3 (lipids + symptoms + history)
    with col3:
        totchol = st.number_input("Total Cholesterol (mg/dL)", 100.0, 500.0, 210.0, 1.0, key="calc_totchol")
        hdlc = st.number_input("HDL Cholesterol (mg/dL)", 10.0, 150.0, 45.0, 1.0, key="calc_hdlc")
        ldlc = st.number_input("LDL Cholesterol (mg/dL)", 30.0, 300.0, 120.0, 1.0, key="calc_ldlc")

        angina = st.selectbox("Angina / chest pain (ANGINA)", ["No", "Yes"], index=0, key="calc_angina")
        prevap = st.selectbox("Prior angina (PREVAP)", ["No", "Yes"], index=0, key="calc_prevap")

        prevchd = st.selectbox("Prior CHD (PREVCHD)", ["No", "Yes"], index=0, key="calc_prevchd")
        prevmi = st.selectbox("Prior MI (PREVMI)", ["No", "Yes"], index=0, key="calc_prevmi")
        hospmi = st.selectbox("Hospitalized MI (HOSPMI)", ["No", "Yes"], index=0, key="calc_hospmi")

        stroke = st.selectbox("Prior Stroke (STROKE)", ["No", "Yes"], index=0, key="calc_stroke")
        prevstrk = st.selectbox("Prior stroke history (PREVSTRK)", ["No", "Yes"], index=0, key="calc_prevstrk")

        mi_fchd = st.selectbox("MI_FCHD", ["No", "Yes"], index=0, key="calc_mi_fchd")

    # Map to numeric row
    row = {
        "SEX": 1 if sex == "Male" else 0,
        "TOTCHOL": float(totchol),
        "AGE": float(age),
        "SYSBP": float(sysbp),
        "DIABP": float(diabp),
        "CIGPDAY": float(cigs),
        "BMI": float(bmi),
        "DIABETES": _as_int_yesno(diabetes),
        "BPMEDS": _as_int_yesno(bpmeds),
        "HEARTRTE": float(heartrate),
        "GLUCOSE": float(glucose),
        "educ": int(educ.split("–")[0].strip()),

        "PREVCHD": _as_int_yesno(prevchd),
        "PREVAP": _as_int_yesno(prevap),
        "PREVMI": _as_int_yesno(prevmi),
        "PREVSTRK": _as_int_yesno(prevstrk),
        "PREVHYP": _as_int_yesno(prevhyp),
        "HOSPMI": _as_int_yesno(hospmi),

        "HDLC": float(hdlc),
        "LDLC": float(ldlc),
        "ANGINA": _as_int_yesno(angina),
        "MI_FCHD": _as_int_yesno(mi_fchd),
        "STROKE": _as_int_yesno(stroke),
        "HYPERTEN": _as_int_yesno(hyperten),
    }

    # Ensure training order
    try:
        row_ordered = {feat: row[feat] for feat in FEATURES_24}
    except KeyError as e:
        st.error(f"features_24.json expects feature missing from UI mapping: {e}")
        st.stop()

    return pd.DataFrame([row_ordered])

def stacking_predict_proba_24(df_input: pd.DataFrame, threshold: float):
    """
    Correct stacking (same pattern as v4):
    - scale 24 features
    - p_rf, p_xgb from base models
    - meta LR uses ONLY [p_rf, p_xgb] (2 columns)
    """
    X = df_input.values.astype(float)
    Xs = scaler.transform(X)

    p_rf = rf_model.predict_proba(Xs)[:, 1]
    p_xgb = xgb_model.predict_proba(Xs)[:, 1]

    stack_in = np.column_stack([p_rf, p_xgb])
    p_final = meta_model.predict_proba(stack_in)[:, 1]

    final_prob = float(p_final[0])
    final_label = int(final_prob >= threshold)

    component_probs = {
        "RF (Clinical)": float(p_rf[0]),
        "XGB (Clinical)": float(p_xgb[0]),
    }
    return final_prob, final_label, component_probs

def bie_scenarios_24(df_patient: pd.DataFrame, threshold: float):
    """
    BIE for v5 clinical+history model:
    - baseline
    - no smoking
    - lower SBP by 10
    - BMI -2
    - TOTCHOL -20
    - GLUCOSE -10
    """
    base_prob, _, _ = stacking_predict_proba_24(df_patient, threshold=threshold)

    scenarios = []
    scenarios.append(("Baseline", "Current profile", df_patient.copy()))

    if float(df_patient["CIGPDAY"].iloc[0]) > 0:
        d = df_patient.copy()
        d["CIGPDAY"] = 0.0
        scenarios.append(("No smoking", "Set cigarettes/day → 0", d))

    sysbp = float(df_patient["SYSBP"].iloc[0])
    d = df_patient.copy()
    d["SYSBP"] = max(sysbp - 10.0, 90.0)
    scenarios.append(("Lower SBP by 10 mmHg", f"SYSBP: {sysbp:.0f} → {d['SYSBP'].iloc[0]:.0f}", d))

    bmi = float(df_patient["BMI"].iloc[0])
    d = df_patient.copy()
    d["BMI"] = max(bmi - 2.0, 15.0)
    scenarios.append(("Reduce BMI by 2 kg/m²", f"BMI: {bmi:.1f} → {d['BMI'].iloc[0]:.1f}", d))

    tc = float(df_patient["TOTCHOL"].iloc[0])
    d = df_patient.copy()
    d["TOTCHOL"] = max(tc - 20.0, 100.0)
    scenarios.append(("Lower Total Chol by 20 mg/dL", f"TOTCHOL: {tc:.0f} → {d['TOTCHOL'].iloc[0]:.0f}", d))

    gl = float(df_patient["GLUCOSE"].iloc[0])
    d = df_patient.copy()
    d["GLUCOSE"] = max(gl - 10.0, 60.0)
    scenarios.append(("Lower Glucose by 10 mg/dL", f"GLUCOSE: {gl:.0f} → {d['GLUCOSE'].iloc[0]:.0f}", d))

    rows = []
    best = None

    for name, desc, dfx in scenarios:
        p, _, _ = stacking_predict_proba_24(dfx, threshold=threshold)
        abs_change = (p - base_prob) * 100.0
        rel_change = (p - base_prob) / base_prob * 100.0 if base_prob > 0 else 0.0

        rows.append({
            "Scenario": name,
            "Description": desc,
            "Risk (%)": p * 100.0,
            "Δ abs (pp)": abs_change,
            "Δ rel (%)": rel_change,
        })

        if name != "Baseline":
            drop = base_prob - p
            if best is None or drop > best["drop"]:
                best = {"name": name, "p": p, "drop": drop}

    return base_prob, pd.DataFrame(rows), best

# =========================
# 4) SIDEBAR
# =========================
with st.sidebar:
    st.markdown(
        """
        <h2 style='margin-bottom:0;'>🫀 CVD Stacking GenAI</h2>
        <p style='margin-top:4px;font-size:13px;'>
        <b>v5.0 – 24 Features (Clinical+History)</b><br>
        Framingham-based • Expanded clinical history (PREV* / HOSPMI)
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    threshold = st.slider(
        "Alert threshold (probability of CVD)",
        0.10, 0.90, DEFAULT_THRESHOLD, 0.05,
        help="If predicted risk ≥ threshold, the model flags the patient as 'At Risk'.",
        key="sidebar_threshold"
    )

    show_components = st.checkbox(
        "Show component model probabilities (RF/XGB)",
        value=True,
        key="sidebar_show_components"
    )

    st.markdown("---")
    st.markdown(
        """
        **Disclaimer**  
        This tool is for **research & education** only and must not be used as
        a standalone diagnostic system.
        """
    )

# =========================
# 5) HERO HEADER
# =========================
st.markdown(
    """
    <div style="background-color:#0f4c75;padding:18px;border-radius:8px;margin-bottom:16px;">
      <h1 style="color:white;margin-bottom:4px;">CVD Risk Prediction – Clinical Risk Stratification Model (v5.0)</h1>
      <p style="color:#e0f2f1;margin:0;font-size:14px;">
        10-Year Cardiovascular Risk Estimation with Expanded Clinical History (PREV* / HOSPMI)
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# 6) TABS (match v4/v3)
# =========================
tab_calc, tab_bie, tab_model, tab_faq = st.tabs(
    ["🧮 Risk Calculator", "🧠 Behavioral Impact Engine (BIE)", "🧬 Model & Data", "❓ FAQ & Notes"]
)

# -------------------------
# TAB 1 – RISK CALCULATOR
# -------------------------
with tab_calc:
    st.markdown(
        "Use this tab to enter patient information and obtain a **10-year CVD risk estimate** (24-feature clinical+history model)."
    )

    df_input = build_input_df_24()

    with st.expander("View encoded feature vector (for experts)", expanded=False):
        st.dataframe(df_input.style.format(precision=2), use_container_width=True)

    run_btn = st.button("Run CVD Risk Prediction", type="primary", key="btn_run_calc")

    if run_btn:
        with st.spinner("Running clinical stacking model..."):
            final_prob, final_label, component_probs = stacking_predict_proba_24(df_input, threshold=threshold)

        st.session_state["v5_last_input_df"] = df_input
        st.session_state["v5_last_prob"] = final_prob
        st.session_state["v5_last_label"] = final_label
        st.session_state["v5_last_components"] = component_probs
        st.session_state["v5_last_threshold"] = threshold

        category, color = interpret_risk(final_prob)

        st.markdown("### Prediction Result")
        col_res1, col_res2 = st.columns([2, 1])

        with col_res1:
            st.metric("Estimated 10-year CVD risk", f"{final_prob*100:.1f} %")
            st.markdown(f"**Risk category:** {color} **{category}**")
            st.markdown(
                f"**Model decision at threshold {threshold:.2f}:** "
                f"{'⚠️ At Risk (1)' if final_label == 1 else '✅ Not Flagged (0)'}"
            )

        with col_res2:
            st.markdown(
                """
                **Interpretation guide**  
                - <5%: Low risk  
                - 5–9%: Borderline  
                - 10–19%: Intermediate  
                - ≥20%: High risk  
                """
            )

        if show_components:
            st.markdown("### Component Model Contributions")
            comp_df = pd.DataFrame(
                {"Model": list(component_probs.keys()),
                 "Predicted CVD risk (%)": [p * 100 for p in component_probs.values()]}
            )
            st.bar_chart(comp_df.set_index("Model"))

        st.info("Next: open **Behavioral Impact Engine (BIE)** to see patient-specific what-if scenarios.")
    else:
        st.info("Fill in the patient information and click **Run CVD Risk Prediction**.")

# -------------------------
# TAB 2 – BIE
# -------------------------
with tab_bie:
    st.subheader("Behavioral Impact Engine (BIE)")

    st.markdown(
        """
        The **Behavioral Impact Engine (BIE)** evaluates which modifiable factor matters most for a specific patient
        and provides **model-based counterfactual scenarios** (what-if changes → re-score → compare).
        """
    )

    st.markdown("---")

    if "v5_last_input_df" not in st.session_state:
        st.warning("Please run a prediction in **Risk Calculator** first.")
    else:
        df_patient = st.session_state["v5_last_input_df"]
        used_threshold = st.session_state.get("v5_last_threshold", threshold)

        run_bie = st.button("Run BIE Analysis", key="btn_run_bie")

        if run_bie:
            base_prob, scenario_df, best = bie_scenarios_24(df_patient, threshold=used_threshold)
            category, color = interpret_risk(base_prob)

            st.markdown("### Patient-Specific BIE Summary")
            st.markdown(f"Baseline risk (same model): **{base_prob*100:.1f}%**  ({color} {category})")

            st.markdown("### Scenario Table (what-if)")
            st.dataframe(
                scenario_df.style.format(
                    {"Risk (%)": "{:.2f}", "Δ abs (pp)": "{:+.2f}", "Δ rel (%)": "{:+.1f}"}
                ),
                use_container_width=True
            )

            st.markdown("### Most impactful lever (for this profile)")
            if best is None:
                st.write("No scenario produced a measurable change (rare).")
            else:
                abs_pp = best["drop"] * 100.0
                rel_pct = (best["drop"] / base_prob * 100.0) if base_prob > 0 else 0.0
                st.write(
                    f"**{best['name']}** → estimated risk becomes **{best['p']*100:.2f}%** "
                    f"(**-{abs_pp:.2f} pp absolute**, **-{rel_pct:.1f}% relative**)."
                )

            st.markdown("### Evidence-based recommendations (starter set)")
            st.markdown(
                """
                **Smoking**
                - If the patient smokes, discuss smoking cessation options and supports.

                **Blood Pressure**
                - If SBP/DBP are elevated, discuss lifestyle and clinician-guided management.

                **Lipids & Metabolic**
                - Encourage guideline-aligned diet/exercise; evaluate lipid/glucose management clinically.
                """
            )

            st.markdown(
                """
                ---
                ⚠️ **Important:** BIE outputs are **model-based simulations**, not prescriptive treatment recommendations.
                """
            )
        else:
            st.info("Click **Run BIE Analysis** to generate the scenario table and recommendations.")

# -------------------------
# TAB 3 – MODEL & DATA
# -------------------------
with tab_model:
    st.subheader("Model & Data Overview")

    st.markdown(
        """
        ### Data Source
        - Based on the **Framingham Heart Study** dataset.
        - Target: 10-year cardiovascular disease (**CVD**) outcome.

        ### Feature Set (24 features – clinical+history)
        Includes core risk factors + expanded prior-history fields:
        - Demographics: AGE, SEX, educ
        - BP: SYSBP, DIABP
        - Lipids: TOTCHOL, HDLC, LDLC
        - Metabolic: BMI, GLUCOSE, DIABETES
        - Lifestyle/Treatment: CIGPDAY, BPMEDS
        - Symptoms/History: ANGINA, MI_FCHD, STROKE, HYPERTEN
        - Expanded history: PREVCHD, PREVAP, PREVMI, PREVSTRK, PREVHYP, HOSPMI

        ### Architecture
        - Base learners: **RandomForest** + **XGBoost**
        - Stacking meta-model: **LogisticRegression** trained on **two inputs**:
          \\( z = [p_{RF}, p_{XGB}] \\)
        - Output: probability of 10-year CVD risk (research/education).
        """
    )

# -------------------------
# TAB 4 – FAQ & NOTES
# -------------------------
with tab_faq:
    st.subheader("FAQ & Notes")

    st.markdown(
        """
        **Q1. Why did we see “LogisticRegression expects 2 features”?**  
        Because the meta-model is trained on **two probabilities** from base models: **[p_rf, p_xgb]**.
        The app must never pass the 24 raw features into the meta-model.

        **Q2. Is this clinical decision support?**  
        No. Research & education only. Not medical advice.

        **Q3. What changed from v4 to v5?**  
        v5 expands the clinical+history feature set to 24 inputs (adds PREV* history fields and HOSPMI).
        """
    )

# =========================
# 7) FOOTER
# =========================
st.markdown(
    """
    <hr style="margin-top:32px;margin-bottom:8px;">
    <div style="text-align:center;font-size:12px;color:gray;">
      Stacking Generative AI CVD Risk Model v5.0 • 24 features (clinical+history) • Research & Education Only<br>
      This application does not provide medical advice, diagnosis, or treatment.<br>
      © 2025 Howard Nguyen, PhD. For demonstration only — not for clinical decision-making.
    </div>
    """,
    unsafe_allow_html=True
)
