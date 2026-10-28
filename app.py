"""
Streamlit demo (IDEAS.md #1).

Two tabs:
  1. "Live prediction" -- type in any customer profile, get the classical
     model's prediction + SHAP explanation AND a live LLM call, side by
     side, with real measured latency and cost for each.
  2. "Project report" -- the saved figures from every analysis in this
     repo (calibration, SHAP summary, ROI curve, survival curves, the
     LLM cost/accuracy comparison), browsable without rerunning anything.

Run locally:
    streamlit run app.py
Needs GROQ_API_KEY as an environment variable, a Streamlit secret
(`st.secrets["GROQ_API_KEY"]`), or typed into the sidebar for this
session only (never stored).
"""
import os
import time

import joblib
import pandas as pd
import streamlit as st

from src import config
from src.features import engineer_features, prepare_model_frame, FEATURE_COLUMNS
from src.shap_analysis import compute_shap_values
from src.llm_baseline import serialize_customer, call_groq, SYSTEM_PROMPT, RESPONSE_PATTERN

st.set_page_config(page_title="Churn Prediction: ML vs. LLM", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(config.REPORTS_DIR / "final_model.joblib")


def get_api_key():
    if os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"]
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return st.sidebar.text_input("Groq API key (session only, never stored)", type="password")


CATEGORY_OPTIONS = {
    "gender": ["Female", "Male"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}


def customer_input_form():
    st.subheader("Customer profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", CATEGORY_OPTIONS["gender"])
        senior = st.checkbox("Senior citizen")
        partner = st.checkbox("Has partner")
        dependents = st.checkbox("Has dependents")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    with c2:
        contract = st.selectbox("Contract", CATEGORY_OPTIONS["Contract"])
        payment = st.selectbox("Payment method", CATEGORY_OPTIONS["PaymentMethod"])
        paperless = st.checkbox("Paperless billing", value=True)
        phone = st.checkbox("Phone service", value=True)
        multiple_lines = st.selectbox("Multiple lines", CATEGORY_OPTIONS["MultipleLines"])
    with c3:
        internet = st.selectbox("Internet service", CATEGORY_OPTIONS["InternetService"])
        online_security = st.selectbox("Online security", CATEGORY_OPTIONS["OnlineSecurity"])
        online_backup = st.selectbox("Online backup", CATEGORY_OPTIONS["OnlineBackup"])
        device_protection = st.selectbox("Device protection", CATEGORY_OPTIONS["DeviceProtection"])
        tech_support = st.selectbox("Tech support", CATEGORY_OPTIONS["TechSupport"])

    c4, c5 = st.columns(2)
    with c4:
        streaming_tv = st.selectbox("Streaming TV", CATEGORY_OPTIONS["StreamingTV"])
        streaming_movies = st.selectbox("Streaming movies", CATEGORY_OPTIONS["StreamingMovies"])
    with c5:
        monthly_charges = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, step=1.0)
        total_charges = st.number_input("Total charges ($)", 0.0, 10000.0, float(monthly_charges * max(tenure, 1)), step=10.0)

    return pd.DataFrame([{
        "customerID": "demo-customer", "gender": gender, "SeniorCitizen": int(senior),
        "Partner": "Yes" if partner else "No", "Dependents": "Yes" if dependents else "No",
        "tenure": tenure, "PhoneService": "Yes" if phone else "No", "MultipleLines": multiple_lines,
        "InternetService": internet, "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies, "Contract": contract,
        "PaperlessBilling": "Yes" if paperless else "No", "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
    }])


def render_live_prediction_tab():
    raw_row = customer_input_form()
    run = st.button("Predict", type="primary")
    if not run:
        return

    engineered = engineer_features(raw_row)
    model_row = prepare_model_frame(engineered)

    pipe = load_model()

    col_classical, col_llm = st.columns(2)

    with col_classical:
        st.markdown("### Classical model (Random Forest)")
        t0 = time.perf_counter()
        proba = pipe.predict_proba(model_row[FEATURE_COLUMNS])[0, 1]
        latency = time.perf_counter() - t0
        st.metric("Predicted churn probability", f"{proba:.0%}")
        st.caption(f"Latency: {latency*1000:.1f} ms | Cost: $0 (local inference)")

        shap_values, X_t, explainer = compute_shap_values(pipe, model_row[FEATURE_COLUMNS])
        clean_names = [c.split("__", 1)[-1] for c in X_t.columns]
        contrib = pd.DataFrame({
            "feature": clean_names, "shap_value": shap_values[0],
        }).sort_values("shap_value", key=abs, ascending=False).head(8)
        st.caption("Top factors driving this prediction (SHAP):")
        st.dataframe(
            contrib.assign(direction=lambda d: d["shap_value"].apply(
                lambda v: "raises risk" if v > 0 else "lowers risk"))
            .assign(shap_value=lambda d: d["shap_value"].round(3)),
            hide_index=True, use_container_width=True,
        )

    with col_llm:
        st.markdown("### LLM (openai/gpt-oss-20b via Groq, zero-shot)")
        api_key = get_api_key()
        if not api_key:
            st.warning("Enter a Groq API key in the sidebar to get a live LLM prediction.")
        else:
            with st.spinner("Calling Groq..."):
                profile_text = serialize_customer(raw_row.iloc[0])
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": profile_text + "\n\nWill this customer churn?"},
                ]
                try:
                    result = call_groq(messages, api_key)
                    match = RESPONSE_PATTERN.search(result["content"])
                    if match:
                        llm_proba = int(match.group(1)) / 100.0
                        st.metric("Predicted churn probability", f"{llm_proba:.0%}")
                    else:
                        st.error(f"Couldn't parse a probability from: {result['content']!r}")
                    cost = (result["prompt_tokens"] / 1e6 * config.LLM_PRICE_PER_1M_INPUT_TOKENS
                            + result["completion_tokens"] / 1e6 * config.LLM_PRICE_PER_1M_OUTPUT_TOKENS)
                    st.caption(f"Latency: {result['latency_sec']*1000:.0f} ms | "
                               f"Cost: ${cost:.5f} | Tokens: {result['prompt_tokens']} in / "
                               f"{result['completion_tokens']} out")
                    with st.expander("Raw LLM response"):
                        st.code(result["content"])
                except Exception as exc:
                    st.error(f"Groq API call failed: {exc}")

    st.caption(
        "Both models are shown on the exact same profile you entered, with their "
        "real measured latency and cost for this one call -- not averages. See the "
        "'Project report' tab for the full 200-customer comparison this is sampled from."
    )


REPORT_FIGURES = [
    ("Confusion matrix", "classification_confusion_matrix.png"),
    ("Feature importance (permutation)", "feature_importance.png"),
    ("SHAP summary (global)", "shap_summary.png"),
    ("Calibration / reliability diagram", "calibration_reliability_diagram.png"),
    ("ROI simulation: model-ranked vs. random targeting", "roi_simulation.png"),
    ("Survival curves by contract type", "survival_km_by_contract.png"),
    ("Survival curves by internet service", "survival_km_by_internet.png"),
    ("Kaplan-Meier overall survival curve", "survival_km_overall.png"),
    ("LLM vs. classical: accuracy & cost", "llm_vs_classical_comparison.png"),
]


def render_report_tab():
    st.caption("Static figures generated by the pipeline scripts in this repo -- "
               "see README.md and RESULTS.md for the full numeric writeup.")
    cols = st.columns(2)
    for i, (title, filename) in enumerate(REPORT_FIGURES):
        path = config.FIGURES_DIR / filename
        with cols[i % 2]:
            st.markdown(f"**{title}**")
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.info(f"Not generated yet -- run the corresponding `run_*.py` script.")


def main():
    st.title("Customer Churn Prediction: Classical ML vs. LLM")
    st.caption(
        "A churn classifier (Random Forest) benchmarked against an LLM (openai/gpt-oss-20b "
        "via Groq) on cost, accuracy, latency, and calibration -- plus SHAP explainability, "
        "an ROI simulator, and a survival-analysis reframing of the same data."
    )
    tab1, tab2 = st.tabs(["Live prediction", "Project report"])
    with tab1:
        render_live_prediction_tab()
    with tab2:
        render_report_tab()


if __name__ == "__main__":
    main()
