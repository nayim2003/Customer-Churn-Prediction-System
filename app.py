import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Telco Churn Intelligence", page_icon="📊", layout="wide")

MODEL_PATH = Path("models/churn_pipeline.pkl")
st.markdown("""
<style>
.main {background-color:#f7f9fc}
.block-container {padding-top:2rem}
[data-testid="stMetric"] {background:#fff;padding:16px;border:1px solid #e8edf4;border-radius:12px}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

bundle = load_bundle()
st.title("📊 Telco Customer Churn Intelligence")
st.caption("Customer-level churn risk prediction • IBM Telco Customer Churn")
if bundle is None:
    st.error("Model bundle not found. Place your notebook-generated file at models/churn_pipeline.pkl.")
    st.code("project/\\n  app.py\\n  models/churn_pipeline.pkl\\n  requirements.txt")
    st.stop()

model = bundle["model"]
scaler = bundle["scaler"]
encoder = bundle["encoder"]
threshold = float(bundle["threshold"])
feature_names = bundle.get("feature_names", [])

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}
# Keep input columns aligned to the fitted scaler/encoder.
numeric_cols = list(getattr(scaler, "feature_names_in_", NUMERIC))
categorical_cols = list(getattr(encoder, "feature_names_in_", list(CATEGORICAL_OPTIONS)))
if "SeniorCitizen" in numeric_cols:
    NUMERIC.insert(0, "SeniorCitizen")

def transform(raw):
    raw = raw.copy()
    for col in numeric_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    for col in categorical_cols:
        raw[col] = raw[col].astype(str)
    if raw[numeric_cols].isna().any().any():
        raise ValueError("Numeric fields contain missing or invalid values.")
    num = scaler.transform(raw[numeric_cols])
    cat = encoder.transform(raw[categorical_cols])
    return np.hstack([num, cat])

def risk_band(p):
    if p >= max(threshold, .70): return "High"
    if p >= threshold: return "Elevated"
    return "Lower"

tab1, tab2, tab3 = st.tabs(["🔎 Single customer", "📁 Batch predictions", "ℹ️ About"])

with tab1:
    st.subheader("Customer profile")
    with st.form("customer_form"):
        cols = st.columns(3)
        vals = {}
        for i, col in enumerate(numeric_cols):
            label = {"tenure":"Tenure (months)","MonthlyCharges":"Monthly charges","TotalCharges":"Total charges","SeniorCitizen":"Senior citizen (0/1)"}.get(col,col)
            with cols[i % 3]:
                if col == "tenure": vals[col] = st.number_input(label, min_value=0, max_value=100, value=12)
                elif col == "SeniorCitizen": vals[col] = st.selectbox(label,[0,1],format_func=lambda x:"Yes" if x else "No")
                else: vals[col] = st.number_input(label,min_value=0.0,value=70.0,step=5.0)
        for i, col in enumerate(categorical_cols):
            options = CATEGORICAL_OPTIONS.get(col, ["No", "Yes"])
            with cols[i % 3]:
                vals[col] = st.selectbox(col, options)
        submitted = st.form_submit_button("Predict churn risk", type="primary", use_container_width=True)
    if submitted:
        try:
            row = pd.DataFrame([{c: vals[c] for c in numeric_cols + categorical_cols}])
            probability = float(model.predict_proba(transform(row))[0,1])
            prediction = probability >= threshold
            band = risk_band(probability)
            a,b,c = st.columns(3)
            a.metric("Churn probability",f"{probability:.1%}")
            b.metric("Model decision", "Churn risk" if prediction else "No churn flag")
            c.metric("Risk band",band)
            st.progress(min(max(probability,0),1),text="Estimated churn probability")
            st.caption(f"Decision threshold: {threshold:.2f}. Risk bands are presentation labels; the probability is the model output.")
            if prediction:
                st.warning("This profile meets the model's churn decision threshold. Consider reviewing the customer's situation before taking action.")
            else:
                st.success("This profile is below the model's churn decision threshold.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tab2:
    st.subheader("Upload customer records")
    st.write("Upload a CSV containing the same raw input columns used during training. `customerID` and `Churn` are optional and are not used as model inputs.")
    uploaded = st.file_uploader("Choose CSV", type=["csv"])
    if uploaded:
        try:
            data = pd.read_csv(uploaded)
            ids = data["customerID"] if "customerID" in data.columns else pd.Series(data.index.astype(str),name="row")
            missing = [c for c in numeric_cols + categorical_cols if c not in data.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                if st.button("Run batch prediction",type="primary"):
                    probs = model.predict_proba(transform(data[numeric_cols + categorical_cols]))[:,1]
                    out = pd.DataFrame({"customerID":ids,"churn_probability":probs,
                                        "predicted_churn":np.where(probs >= threshold,"Yes","No"),
                                        "risk_band":[risk_band(float(p)) for p in probs]})
                    st.dataframe(out,use_container_width=True)
                    st.download_button("Download predictions CSV",out.to_csv(index=False).encode("utf-8"),
                                       "churn_predictions.csv","text/csv")
        except Exception as e:
            st.error(f"Could not process file: {e}")

with tab3:
    st.subheader("Model and usage notes")
    st.write(f"**Model:** {type(model).__name__}")
    st.write(f"**Decision threshold:** {threshold:.3f}")
    st.write(f"**Input features:** {len(numeric_cols) + len(categorical_cols)} raw fields")
    st.info("Predictions are estimates based on the training data and should support—not replace—customer-service judgment. Avoid uploading unnecessary personal identifiers.")
