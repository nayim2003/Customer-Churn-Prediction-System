import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import textwrap
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ChurnIQ | Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# GLOBAL CSS
# =========================================================

st.markdown(
    textwrap.dedent(
        """
        <style>

        /* =================================================
           GLOBAL
           ================================================= */

        .stApp {
            background: #f5f7fb;
        }

        .block-container {
            max-width: 1380px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3, h4 {
            color: #172033 !important;
        }

        p {
            color: #526071;
        }

        hr {
            border: none;
            border-top: 1px solid #e4e8ef;
            margin: 1.5rem 0;
        }


        /* =================================================
           SIDEBAR
           ================================================= */

        section[data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid #202938;
        }

        section[data-testid="stSidebar"] * {
            color: #e5e7eb !important;
        }

        .sidebar-brand {
            padding: 10px 5px 25px 5px;
        }

        .sidebar-brand-title {
            font-size: 24px;
            font-weight: 800;
            color: #ffffff !important;
            margin-bottom: 4px;
        }

        .sidebar-brand-subtitle {
            font-size: 13px;
            color: #9ca3af !important;
        }

        .sidebar-card {
            background: #1b2433;
            border: 1px solid #2a3547;
            border-radius: 12px;
            padding: 15px;
            margin-top: 15px;
        }

        .sidebar-card-title {
            font-size: 13px;
            font-weight: 700;
            color: #ffffff !important;
            margin-bottom: 8px;
        }

        .sidebar-card-text {
            font-size: 12px;
            line-height: 1.6;
            color: #aeb8c7 !important;
        }


        /* =================================================
           HERO
           ================================================= */

        .hero {
            background: linear-gradient(
                135deg,
                #111827 0%,
                #1d3557 55%,
                #244b78 100%
            );
            border-radius: 18px;
            padding: 35px 40px;
            margin-bottom: 30px;
            box-shadow: 0 15px 35px rgba(17, 24, 39, 0.15);
        }

        .hero-badge {
            display: inline-block;
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.15);
            color: #dbeafe !important;
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 14px;
        }

        .hero-title {
            color: #ffffff !important;
            font-size: 34px;
            font-weight: 800;
            margin: 0;
        }

        .hero-subtitle {
            color: #cbd5e1 !important;
            font-size: 15px;
            margin-top: 10px;
            margin-bottom: 0;
        }


        /* =================================================
           SECTION HEADERS
           ================================================= */

        .section-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 28px 0 8px 0;
        }

        .section-icon {
            width: 34px;
            height: 34px;
            border-radius: 9px;
            background: #e8f0ff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 17px;
        }

        .section-title {
            font-size: 20px;
            font-weight: 750;
            color: #172033 !important;
        }

        .section-description {
            font-size: 13px;
            color: #697586 !important;
            margin-bottom: 18px;
        }


        /* =================================================
           INPUT LABELS
           ================================================= */

        div[data-testid="stWidgetLabel"] p {
            color: #273449 !important;
            font-weight: 650 !important;
            font-size: 13px !important;
            opacity: 1 !important;
        }

        div[data-testid="stSelectbox"] label p,
        div[data-testid="stNumberInput"] label p {
            color: #273449 !important;
            font-weight: 650 !important;
            opacity: 1 !important;
        }


        /* =================================================
           SELECTBOX
           ================================================= */

        div[data-baseweb="select"] > div {
            background: #ffffff !important;
            border: 1px solid #d8dee8 !important;
            border-radius: 8px !important;
            min-height: 42px;
        }

        div[data-baseweb="select"] > div:hover {
            border-color: #7098d8 !important;
        }

        div[data-baseweb="select"] span {
            color: #172033 !important;
        }


        /* =================================================
           NUMBER INPUT
           ================================================= */

        div[data-testid="stNumberInput"] input {
            background: #ffffff !important;
            color: #172033 !important;
            border: 1px solid #d8dee8 !important;
            border-radius: 8px !important;
            min-height: 42px;
        }

        div[data-testid="stNumberInput"] input:focus {
            border-color: #4f7fd6 !important;
            box-shadow: 0 0 0 1px #4f7fd6 !important;
        }


        /* =================================================
           PREDICTION BUTTON
           ================================================= */

        div.stButton > button {
            background: linear-gradient(
                135deg,
                #1d4ed8,
                #2563eb
            ) !important;

            color: #ffffff !important;
            border: none !important;
            border-radius: 10px !important;
            min-height: 50px !important;
            font-size: 15px !important;
            font-weight: 700 !important;
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.22);
            transition: all 0.2s ease;
        }

        div.stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 24px rgba(37, 99, 235, 0.28);
        }


        /* =================================================
           KPI CARDS
           ================================================= */

        .kpi-card {
            background: #ffffff;
            border: 1px solid #e2e7ef;
            border-radius: 14px;
            padding: 20px;
            min-height: 115px;
            box-shadow: 0 5px 15px rgba(15, 23, 42, 0.04);
        }

        .kpi-label {
            color: #697586 !important;
            font-size: 12px;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .kpi-value {
            color: #172033 !important;
            font-size: 26px;
            font-weight: 800;
            margin-top: 8px;
        }

        .kpi-small {
            color: #7b8796 !important;
            font-size: 12px;
            margin-top: 4px;
        }


        /* =================================================
           RISK CARD
           ================================================= */

        .risk-card {
            border-radius: 14px;
            padding: 20px 22px;
            margin-top: 18px;
            border: 1px solid #e2e7ef;
            background: #ffffff;
        }

        .risk-title {
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            color: #697586 !important;
        }

        .risk-value {
            font-size: 25px;
            font-weight: 800;
            margin-top: 5px;
        }

        .risk-high {
            color: #dc2626 !important;
        }

        .risk-low {
            color: #15803d !important;
        }


        /* =================================================
           PROGRESS BAR
           ================================================= */

        .progress-container {
            width: 100%;
            height: 10px;
            background: #e7ebf1;
            border-radius: 999px;
            overflow: hidden;
            margin-top: 10px;
        }

        .progress-bar {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(
                90deg,
                #22c55e,
                #f59e0b,
                #ef4444
            );
        }


        /* =================================================
           SHAP CARD
           ================================================= */

        .shap-card {
            background: #ffffff;
            border: 1px solid #e2e7ef;
            border-radius: 14px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(15, 23, 42, 0.04);
        }


        /* =================================================
           INFO CARD
           ================================================= */

        .info-card {
            background: #f8fafc;
            border: 1px solid #e4e8ef;
            border-radius: 12px;
            padding: 15px 18px;
            margin-top: 12px;
        }

        .info-title {
            font-weight: 700;
            color: #273449 !important;
            font-size: 13px;
        }

        .info-text {
            font-size: 12px;
            color: #697586 !important;
            line-height: 1.6;
            margin-top: 5px;
        }


        /* =================================================
           EXPANDER
           ================================================= */

        details {
            background: #ffffff !important;
            border: 1px solid #e2e7ef !important;
            border-radius: 12px !important;
        }


        /* =================================================
           FOOTER
           ================================================= */

        .footer {
            text-align: center;
            color: #8a94a3 !important;
            font-size: 12px;
            padding-top: 35px;
            padding-bottom: 10px;
        }

        </style>
        """
    ),
    unsafe_allow_html=True
)


# =========================================================
# MODEL PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "churn_pipeline.pkl"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:

    best_model = load_model()

    preprocessor = (
        best_model
        .named_steps["preprocessor"]
    )

    xgb_model = (
        best_model
        .named_steps["model"]
    )

except Exception as e:

    st.error(
        "Unable to load the churn prediction model."
    )

    st.code(str(e))

    st.stop()


# =========================================================
# SHAP EXPLAINER
# =========================================================

@st.cache_resource
def load_explainer(model):

    return shap.TreeExplainer(model)


try:

    explainer = load_explainer(
        xgb_model
    )

except Exception:

    explainer = None


# =========================================================
# FEATURE ENGINEERING
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

    AvgChargesPerMonth = (
        TotalCharges / (tenure + 1)
    )


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


    # Customer dataframe

    customer = pd.DataFrame(
        [{
            "gender": gender,

            "SeniorCitizen":
                SeniorCitizen,

            "Partner":
                Partner,

            "Dependents":
                Dependents,

            "tenure":
                tenure,

            "PhoneService":
                PhoneService,

            "MultipleLines":
                MultipleLines,

            "InternetService":
                InternetService,

            "OnlineSecurity":
                OnlineSecurity,

            "OnlineBackup":
                OnlineBackup,

            "DeviceProtection":
                DeviceProtection,

            "TechSupport":
                TechSupport,

            "StreamingTV":
                StreamingTV,

            "StreamingMovies":
                StreamingMovies,

            "Contract":
                Contract,

            "PaperlessBilling":
                PaperlessBilling,

            "PaymentMethod":
                PaymentMethod,

            "MonthlyCharges":
                MonthlyCharges,

            "TotalCharges":
                TotalCharges,

            "AvgChargesPerMonth":
                AvgChargesPerMonth,

            "TenureGroup":
                TenureGroup,

            "NumServices":
                NumServices
        }]
    )

    return customer


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        textwrap.dedent(
            """
            <div class="sidebar-brand">

                <div class="sidebar-brand-title">
                    📊 ChurnIQ
                </div>

                <div class="sidebar-brand-subtitle">
                    Customer Analytics Platform
                </div>

            </div>
            """
        ),
        unsafe_allow_html=True
    )


    st.markdown(
        textwrap.dedent(
            """
            <div class="sidebar-card">

                <div class="sidebar-card-title">
                    🤖 Model
                </div>

                <div class="sidebar-card-text">
                    XGBoost classification model with
                    engineered customer features.
                </div>

            </div>
            """
        ),
        unsafe_allow_html=True
    )


    st.markdown(
        textwrap.dedent(
            """
            <div class="sidebar-card">

                <div class="sidebar-card-title">
                    🔍 Explainability
                </div>

                <div class="sidebar-card-text">
                    SHAP-based feature contribution
                    analysis explains individual predictions.
                </div>

            </div>
            """
        ),
        unsafe_allow_html=True
    )


    st.markdown(
        textwrap.dedent(
            """
            <div class="sidebar-card">

                <div class="sidebar-card-title">
                    📋 How to use
                </div>

                <div class="sidebar-card-text">
                    1. Enter customer information.<br>
                    2. Review billing details.<br>
                    3. Click Analyze Customer Churn.<br>
                    4. Review risk and SHAP explanation.
                </div>

            </div>
            """
        ),
        unsafe_allow_html=True
    )


    st.markdown(
        textwrap.dedent(
            """
            <div class="sidebar-card">

                <div class="sidebar-card-title">
                    ⚠️ Note
                </div>

                <div class="sidebar-card-text">
                    Predictions are model-based estimates
                    and should be interpreted together
                    with business context.
                </div>

            </div>
            """
        ),
        unsafe_allow_html=True
    )


# =========================================================
# HERO
# =========================================================

st.markdown(
    textwrap.dedent(
        """
        <div class="hero">

            <div class="hero-badge">
                CUSTOMER ANALYTICS • MACHINE LEARNING
            </div>

            <div class="hero-title">
                Customer Churn Prediction
            </div>

            <div class="hero-subtitle">
                Predict customer churn probability and
                understand the key factors influencing
                each prediction.
            </div>

        </div>
        """
    ),
    unsafe_allow_html=True
)


# =========================================================
# CUSTOMER PROFILE HEADER
# =========================================================

st.markdown(
    textwrap.dedent(
        """
        <div class="section-header">

            <div class="section-icon">
                👤
            </div>

            <div class="section-title">
                Customer Profile
            </div>

        </div>

        <div class="section-description">
            Enter demographic, subscription and service information.
        </div>
        """
    ),
    unsafe_allow_html=True
)


# =========================================================
# CUSTOMER INPUTS
# =========================================================

col1, col2, col3 = st.columns(3)


# =========================================================
# COLUMN 1
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
        format_func=lambda x:
            "Yes" if x == 1 else "No"
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
        value=12,
        step=1
    )


# =========================================================
# COLUMN 2
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
# COLUMN 3
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
# BILLING HEADER
# =========================================================

st.markdown(
    textwrap.dedent(
        """
        <div class="section-header">

            <div class="section-icon">
                💳
            </div>

            <div class="section-title">
                Billing Information
            </div>

        </div>

        <div class="section-description">
            Enter the customer's monthly and accumulated billing information.
        </div>
        """
    ),
    unsafe_allow_html=True
)


# =========================================================
# BILLING INPUTS
# =========================================================

billing_col1, billing_col2 = st.columns(2)


with billing_col1:

    MonthlyCharges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )


with billing_col2:

    TotalCharges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=float(
            MonthlyCharges * tenure
        ),
        step=10.0
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True
)


predict_button = st.button(
    "🔮  Analyze Customer Churn",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    try:

        # -------------------------------------------------
        # CREATE CUSTOMER
        # -------------------------------------------------

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


        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction = best_model.predict(
            customer
        )[0]


        probability = best_model.predict_proba(
            customer
        )[0, 1]


        stay_probability = (
            1 - probability
        )


        # =================================================
        # PREDICTION SUMMARY
        # =================================================

        st.markdown(
            textwrap.dedent(
                """
                <div class="section-header">

                    <div class="section-icon">
                        📊
                    </div>

                    <div class="section-title">
                        Prediction Summary
                    </div>

                </div>
                """
            ),
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # KPI 1
        # -------------------------------------------------

        kpi1, kpi2, kpi3 = st.columns(3)


        with kpi1:

            if prediction == 1:

                prediction_text = "CHURN"
                prediction_icon = "⚠️"

            else:

                prediction_text = "STAY"
                prediction_icon = "✓"


            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="kpi-card">

                        <div class="kpi-label">
                            Prediction
                        </div>

                        <div class="kpi-value">
                            {prediction_icon} {prediction_text}
                        </div>

                        <div class="kpi-small">
                            Model classification
                        </div>

                    </div>
                    """
                ),
                unsafe_allow_html=True
            )


        # -------------------------------------------------
        # KPI 2
        # -------------------------------------------------

        with kpi2:

            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="kpi-card">

                        <div class="kpi-label">
                            Churn Probability
                        </div>

                        <div class="kpi-value">
                            {probability:.1%}
                        </div>

                        <div class="kpi-small">
                            Estimated probability
                        </div>

                    </div>
                    """
                ),
                unsafe_allow_html=True
            )


        # -------------------------------------------------
        # KPI 3
        # -------------------------------------------------

        with kpi3:

            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="kpi-card">

                        <div class="kpi-label">
                            Stay Probability
                        </div>

                        <div class="kpi-value">
                            {stay_probability:.1%}
                        </div>

                        <div class="kpi-small">
                            Estimated retention probability
                        </div>

                    </div>
                    """
                ),
                unsafe_allow_html=True
            )


        # =================================================
        # RISK ASSESSMENT
        # =================================================

        if probability >= 0.50:

            risk_text = "Higher Churn Risk"
            risk_class = "risk-high"

        else:

            risk_text = "Lower Churn Risk"
            risk_class = "risk-low"


        progress_width = (
            probability * 100
        )


        st.markdown(
            textwrap.dedent(
                f"""
                <div class="risk-card">

                    <div class="risk-title">
                        Risk Assessment
                    </div>

                    <div class="risk-value {risk_class}">
                        {risk_text}
                    </div>

                    <div class="kpi-small">
                        Churn probability: {probability:.1%}
                    </div>

                    <div class="progress-container">

                        <div
                            class="progress-bar"
                            style="width:{progress_width:.2f}%"
                        ></div>

                    </div>

                </div>
                """
            ),
            unsafe_allow_html=True
        )


        # =================================================
        # SHAP EXPLANATION HEADER
        # =================================================

        st.markdown(
            textwrap.dedent(
                """
                <div class="section-header">

                    <div class="section-icon">
                        🔍
                    </div>

                    <div class="section-title">
                        Prediction Explanation
                    </div>

                </div>

                <div class="section-description">
                    SHAP explains which features contributed
                    to this individual prediction.
                </div>
                """
            ),
            unsafe_allow_html=True
        )


        # =================================================
        # SHAP
        # =================================================

        if explainer is not None:

            customer_transformed = (
                preprocessor.transform(
                    customer
                )
            )


            customer_shap = (
                explainer.shap_values(
                    customer_transformed
                )
            )


            # -------------------------------------------------
            # HANDLE SHAP OUTPUT
            # -------------------------------------------------

            if isinstance(
                customer_shap,
                list
            ):

                if len(customer_shap) > 1:

                    shap_values = (
                        customer_shap[1][0]
                    )

                else:

                    shap_values = (
                        customer_shap[0][0]
                    )


            elif len(
                np.asarray(
                    customer_shap
                ).shape
            ) == 3:

                shap_values = (
                    customer_shap[0, :, 1]
                )

            else:

                shap_values = (
                    customer_shap[0]
                )


            shap_values = np.asarray(
                shap_values
            ).flatten()


            # -------------------------------------------------
            # FEATURE NAMES
            # -------------------------------------------------

            feature_names = (
                preprocessor
                .get_feature_names_out()
            )


            # -------------------------------------------------
            # SHAP DATAFRAME
            # -------------------------------------------------

            shap_df = pd.DataFrame(
                {
                    "Feature":
                        feature_names,

                    "SHAP Value":
                        shap_values
                }
            )


            shap_df["Impact"] = np.where(
                shap_df["SHAP Value"] > 0,
                "Increases Churn Risk",
                "Reduces Churn Risk"
            )


            shap_df["Absolute Impact"] = (
                shap_df["SHAP Value"]
                .abs()
            )


            shap_df = (
                shap_df
                .sort_values(
                    "Absolute Impact",
                    ascending=False
                )
                .head(12)
            )


            # -------------------------------------------------
            # PLOT DATA
            # -------------------------------------------------

            plot_df = (
                shap_df
                .sort_values(
                    "SHAP Value"
                )
            )


            # -------------------------------------------------
            # CREATE PLOT
            # -------------------------------------------------

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )


            bars = ax.barh(
                plot_df["Feature"],
                plot_df["SHAP Value"]
            )


            # -------------------------------------------------
            # BAR COLORS
            # -------------------------------------------------

            for bar, value in zip(
                bars,
                plot_df["SHAP Value"]
            ):

                if value >= 0:

                    bar.set_color(
                        "#dc2626"
                    )

                else:

                    bar.set_color(
                        "#2563eb"
                    )


            # -------------------------------------------------
            # ZERO LINE
            # -------------------------------------------------

            ax.axvline(
                0,
                linewidth=1.2,
                color="#475569"
            )


            # -------------------------------------------------
            # TITLE
            # -------------------------------------------------

            ax.set_title(
                "Top Factors Influencing Churn Prediction",
                fontsize=15,
                fontweight="bold",
                pad=15,
                color="#172033"
            )


            ax.set_xlabel(
                "SHAP Contribution",
                fontsize=10
            )


            # -------------------------------------------------
            # GRID
            # -------------------------------------------------

            ax.grid(
                axis="x",
                alpha=0.2,
                linestyle="--"
            )


            # -------------------------------------------------
            # SPINES
            # -------------------------------------------------

            ax.spines[
                "top"
            ].set_visible(False)

            ax.spines[
                "right"
            ].set_visible(False)

            ax.spines[
                "left"
            ].set_visible(False)


            # -------------------------------------------------
            # VALUE LABELS
            # -------------------------------------------------

            for bar, value in zip(
                bars,
                plot_df["SHAP Value"]
            ):

                if value >= 0:

                    x_position = value
                    ha = "left"

                else:

                    x_position = value
                    ha = "right"


                ax.text(

                    x_position,

                    bar.get_y()
                    + bar.get_height() / 2,

                    f"{value:+.3f}",

                    va="center",

                    ha=ha,

                    fontsize=9,

                    color="#334155"

                )


            plt.tight_layout()


            # -------------------------------------------------
            # DISPLAY SHAP
            # -------------------------------------------------

            st.markdown(
                '<div class="shap-card">',
                unsafe_allow_html=True
            )


            st.pyplot(
                fig,
                use_container_width=True
            )


            st.markdown(
                textwrap.dedent(
                    """
                    <div class="info-card">

                        <div class="info-title">
                            How to read this chart
                        </div>

                        <div class="info-text">
                            Red features push the prediction
                            toward higher churn risk.
                            Blue features push the prediction
                            toward lower churn risk.
                            Larger absolute SHAP values indicate
                            stronger influence on this prediction.
                        </div>

                    </div>
                    """
                ),
                unsafe_allow_html=True
            )


            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


            # =================================================
            # SHAP TABLE
            # =================================================

            with st.expander(
                "📋 View Detailed Feature Contributions"
            ):

                display_df = shap_df[
                    [
                        "Feature",
                        "SHAP Value",
                        "Impact"
                    ]
                ].copy()


                display_df[
                    "SHAP Value"
                ] = (
                    display_df[
                        "SHAP Value"
                    ].round(4)
                )


                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )


        else:

            st.warning(
                "SHAP explanation is currently unavailable."
            )


        # =================================================
        # ENGINEERED FEATURES
        # =================================================

        with st.expander(
            "🧮 View Engineered Customer Features"
        ):

            engineered_df = (
                customer
                .T
                .reset_index()
            )


            engineered_df.columns = [
                "Feature",
                "Value"
            ]


            st.dataframe(
                engineered_df,
                use_container_width=True,
                hide_index=True
            )


        # =================================================
        # MODEL INFORMATION
        # =================================================

        with st.expander(
            "🤖 View Model Information"
        ):

            model_col1, model_col2 = (
                st.columns(2)
            )


            with model_col1:

                st.markdown(
                    """
                    **Model**

                    XGBoost Classification
                    """
                )

                st.markdown(
                    """
                    **Preprocessing**

                    Numerical scaling + categorical encoding
                    """
                )


            with model_col2:

                st.markdown(
                    """
                    **Explainability**

                    SHAP TreeExplainer
                    """
                )

                st.markdown(
                    """
                    **Input**

                    22 raw/engineered customer features
                    """
                )


    except Exception as e:

        st.error(
            "An error occurred while generating the prediction."
        )


        with st.expander(
            "View technical details"
        ):

            st.code(
                str(e)
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    textwrap.dedent(
        """
        <div class="footer">

            ChurnIQ • Customer Churn Prediction System
            <br>
            Built with Python • Streamlit • XGBoost • SHAP

        </div>
        """
    ),
    unsafe_allow_html=True
)
