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
# Custom CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       Main App Background
       ===================================================== */

    .stApp {
        background-color: #f5f7fb;
    }


    /* =====================================================
       Main Content
       ===================================================== */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }


    /* =====================================================
       Headings
       ===================================================== */

    h1 {
        color: #172033 !important;
        font-weight: 700 !important;
    }

    h2 {
        color: #172033 !important;
        font-weight: 700 !important;
    }

    h3 {
        color: #172033 !important;
        font-weight: 650 !important;
    }


    /* =====================================================
       FIX: Streamlit Input Labels
       ===================================================== */

    div[data-testid="stWidgetLabel"] p {
        color: #172033 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        opacity: 1 !important;
    }


    /* Selectbox labels */

    div[data-testid="stSelectbox"] label,
    div[data-testid="stSelectbox"] label p {
        color: #172033 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        opacity: 1 !important;
    }


    /* Number input labels */

    div[data-testid="stNumberInput"] label,
    div[data-testid="stNumberInput"] label p {
        color: #172033 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        opacity: 1 !important;
    }


    /* Text input labels */

    div[data-testid="stTextInput"] label,
    div[data-testid="stTextInput"] label p {
        color: #172033 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        opacity: 1 !important;
    }


    /* =====================================================
       Selectbox
       ===================================================== */

    div[data-baseweb="select"] > div {
        background-color: #252733 !important;
        border-radius: 8px !important;
        border: 1px solid #30323d !important;
        min-height: 42px;
    }


    div[data-baseweb="select"] > div:hover {
        border-color: #4f8cff !important;
    }


    /* Selectbox text */

    div[data-baseweb="select"] div {
        color: #ffffff !important;
    }


    /* =====================================================
       Number Input
       ===================================================== */

    div[data-testid="stNumberInput"] input {
        background-color: #252733 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid #30323d !important;
    }


    div[data-testid="stNumberInput"] input:focus {
        border-color: #4f8cff !important;
        box-shadow: 0 0 0 1px #4f8cff !important;
    }


    /* Number input buttons */

    div[data-testid="stNumberInput"] button {
        background-color: #252733 !important;
        color: #ffffff !important;
        border: none !important;
    }


    /* =====================================================
       Buttons
       ===================================================== */

    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        min-height: 46px;
    }


    /* =====================================================
       Divider
       ===================================================== */

    hr {
        border-color: #dfe3eb !important;
    }


    /* =====================================================
       Metrics
       ===================================================== */

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e3e7ef;
        border-radius: 12px;
        padding: 15px;
    }


    div[data-testid="stMetricLabel"] {
        color: #5f6b7a !important;
    }


    div[data-testid="stMetricValue"] {
        color: #172033 !important;
    }


    /* =====================================================
       Alert Boxes
       ===================================================== */

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }


    /* =====================================================
       Expander
       ===================================================== */

    details {
        background-color: #ffffff !important;
        border: 1px solid #e3e7ef !important;
        border-radius: 10px !important;
    }


    /* =====================================================
       Dataframe
       ===================================================== */

    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }


    </style>
    """,
    unsafe_allow_html=True
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
        1
        for value in service_values
        if value == "Yes"
    )


    # Create customer dataframe
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


# =========================================================
# Column 1
# =========================================================

with col1:

    gender = st.selectbox(
        "Gender",
        [
            "Female",
            "Male"
        ]
    )


    SeniorCitizen = st.selectbox(
        "Senior Citizen",
        [
            0,
            1
        ],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


    Partner = st.selectbox(
        "Partner",
        [
            "Yes",
            "No"
        ]
    )


    Dependents = st.selectbox(
        "Dependents",
        [
            "Yes",
            "No"
        ]
    )


    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )


# =========================================================
# Column 2
# =========================================================

with col2:

    PhoneService = st.selectbox(
        "Phone Service",
        [
            "Yes",
            "No"
        ]
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


# =========================================================
# Column 3
# =========================================================

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
        value=float(MonthlyCharges * tenure),
        step=10.0
    )


# =========================================================
# Prediction Button
# =========================================================

st.divider()


predict_button = st.button(
    "🔮 Predict Customer Churn",
    type="primary",
    use_container_width=True
)


# =========================================================
# Prediction
# =========================================================

if predict_button:

    # -----------------------------------------------------
    # Create raw customer dataframe
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Model Prediction
    # -----------------------------------------------------

    prediction = best_model.predict(customer)[0]

    probability = best_model.predict_proba(
        customer
    )[0, 1]


    # =====================================================
    # Prediction Result
    # =====================================================

    st.subheader("📌 Prediction Result")


    result_col1, result_col2 = st.columns(2)


    # -----------------------------------------------------
    # Prediction Status
    # -----------------------------------------------------

    with result_col1:

        if prediction == 1:

            st.error(
                "⚠️ Customer is predicted to CHURN"
            )

        else:

            st.success(
                "✅ Customer is predicted to STAY"
            )


    # -----------------------------------------------------
    # Probability
    # -----------------------------------------------------

    with result_col2:

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )


    # =====================================================
    # SHAP Explanation
    # =====================================================

    st.subheader("🔍 SHAP Explanation")


    # -----------------------------------------------------
    # Transform raw customer
    # -----------------------------------------------------

    customer_transformed = preprocessor.transform(
        customer
    )


    # -----------------------------------------------------
    # SHAP values
    # -----------------------------------------------------

    customer_shap = explainer.shap_values(
        customer_transformed
    )


    # -----------------------------------------------------
    # Handle multi-output SHAP
    # -----------------------------------------------------

    customer_shap_churn = customer_shap[0][:, 1]


    # -----------------------------------------------------
    # Base value
    # -----------------------------------------------------

    base_value_churn = explainer.expected_value[1]


    # -----------------------------------------------------
    # Feature names
    # -----------------------------------------------------

    shap_feature_names = (
        preprocessor.get_feature_names_out()
    )


    # -----------------------------------------------------
    # SHAP Explanation Object
    # -----------------------------------------------------

    shap_explanation = shap.Explanation(

        values=customer_shap_churn,

        base_values=base_value_churn,

        data=customer_transformed[0],

        feature_names=shap_feature_names

    )


    # =====================================================
    # Waterfall Plot
    # =====================================================

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )


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

    with st.expander(
        "View Processed Customer Data"
    ):

        st.dataframe(
            customer,
            use_container_width=True
        )
