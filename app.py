import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# Load Model
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("models/churn_pipeline.pkl")


best_model = load_model()

preprocessor = best_model.named_steps["preprocessor"]
xgb_model = best_model.named_steps["model"]


# =========================================================
# SHAP Explainer
# =========================================================

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


explainer = load_explainer(xgb_model)


# =========================================================
# Feature Engineering
# =========================================================

def create_customer(
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    tenure,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    MonthlyCharges,
    TotalCharges
):

    # Average charges per month
    AvgChargesPerMonth = TotalCharges / (tenure + 1)

    # Tenure group
    if tenure <= 12:
        TenureGroup = "0-1yr"

    elif tenure <= 24:
        TenureGroup = "1-2yr"

    elif tenure <= 48:
        TenureGroup = "2-4yr"

    else:
        TenureGroup = "4-6yr"

    # Number of services
    service_values = [
        PhoneService,
        MultipleLines,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection,
        TechSupport,
        StreamingTV,
        StreamingMovies
    ]

    NumServices = sum(
        1 for value in service_values
        if value == "Yes"
    )

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "AvgChargesPerMonth": AvgChargesPerMonth,
        "TenureGroup": TenureGroup,
        "NumServices": NumServices
    }])

    return customer


# =========================================================
# App Header
# =========================================================

st.title("📊 Customer Churn Prediction System")

st.write(
    "Enter customer information to predict the probability "
    "of customer churn."
)

st.divider()


# =========================================================
# Customer Information
# =========================================================

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    SeniorCitizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    Partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    Dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )


with col2:

    PhoneService = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    MultipleLines = st.selectbox(
        "Multiple Lines",
        [
            "Yes",
            "No",
            "No phone service"
        ]
    )

    InternetService = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    OnlineSecurity = st.selectbox(
        "Online Security",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    OnlineBackup = st.selectbox(
        "Online Backup",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    DeviceProtection = st.selectbox(
        "Device Protection",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


with col3:

    TechSupport = st.selectbox(
        "Tech Support",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    StreamingTV = st.selectbox(
        "Streaming TV",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    StreamingMovies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    Contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    PaperlessBilling = st.selectbox(
        "Paperless Billing",
        [
            "Yes",
            "No"
        ]
    )

    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


# =========================================================
# Billing Information
# =========================================================

st.subheader("💰 Billing Information")

col1, col2 = st.columns(2)

with col1:

    MonthlyCharges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )

with col2:

    TotalCharges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=MonthlyCharges * tenure,
        step=10.0
    )


# =========================================================
# Prediction
# =========================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Customer Churn",
    type="primary",
    use_container_width=True
)


if predict_button:

    # Create raw customer dataframe
    customer = create_customer(
        gender=gender,
        SeniorCitizen=SeniorCitizen,
        Partner=Partner,
        Dependents=Dependents,
        tenure=tenure,
        PhoneService=PhoneService,
        MultipleLines=MultipleLines,
        InternetService=InternetService,
        OnlineSecurity=OnlineSecurity,
        OnlineBackup=OnlineBackup,
        DeviceProtection=DeviceProtection,
        TechSupport=TechSupport,
        StreamingTV=StreamingTV,
        StreamingMovies=StreamingMovies,
        Contract=Contract,
        PaperlessBilling=PaperlessBilling,
        PaymentMethod=PaymentMethod,
        MonthlyCharges=MonthlyCharges,
        TotalCharges=TotalCharges
    )

    # Model prediction
    prediction = best_model.predict(customer)[0]

    probability = best_model.predict_proba(customer)[0, 1]


    # =====================================================
    # Prediction Result
    # =====================================================

    st.subheader("📌 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if prediction == 1:

            st.error("⚠️ Customer is predicted to CHURN")

        else:

            st.success("✅ Customer is predicted to STAY")


    with result_col2:

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )


    # =====================================================
    # SHAP Explanation
    # =====================================================

    st.subheader("🔍 SHAP Explanation")

    # Transform raw customer
    customer_transformed = preprocessor.transform(customer)

    # SHAP values
    customer_shap = explainer.shap_values(
        customer_transformed
    )

    # Handle multi-output SHAP
    customer_shap_churn = customer_shap[0][:, 1]

    base_value_churn = explainer.expected_value[1]

    # Exact feature names from pipeline
    shap_feature_names = (
        preprocessor.get_feature_names_out()
    )

    shap_explanation = shap.Explanation(
        values=customer_shap_churn,
        base_values=base_value_churn,
        data=customer_transformed[0],
        feature_names=shap_feature_names
    )

    # Waterfall plot
    fig, ax = plt.subplots(figsize=(10, 7))

    shap.plots.waterfall(
        shap_explanation,
        max_display=15,
        show=False
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    st.caption(
        "SHAP values show which features contributed "
        "towards or against the churn prediction."
    )


    # =====================================================
    # Customer Details
    # =====================================================

    with st.expander("View Processed Customer Data"):

        st.dataframe(
            customer,
            use_container_width=True
        )
