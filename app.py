import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ChurnIQ | Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROFESSIONAL CSS
# NOTE:
# No custom HTML <div> is used anywhere in the UI.
# =========================================================

st.markdown(
    """
    <style>

    /* ---------- APP ---------- */

    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: white !important;
    }


    /* ---------- HEADINGS ---------- */

    h1 {
        color: #172033 !important;
        font-weight: 800 !important;
    }

    h2 {
        color: #172033 !important;
        font-weight: 750 !important;
    }

    h3 {
        color: #172033 !important;
        font-weight: 700 !important;
    }


    /* ---------- INPUT LABELS ---------- */

    label p {
        color: #334155 !important;
        font-weight: 600 !important;
    }


    /* ---------- SELECTBOX ---------- */

    div[data-baseweb="select"] > div {
        background-color: white !important;
        border: 1px solid #d8dee8 !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="select"] span {
        color: #172033 !important;
    }


    /* ---------- NUMBER INPUT ---------- */

    div[data-testid="stNumberInput"] input {
        background-color: white !important;
        color: #172033 !important;
        border: 1px solid #d8dee8 !important;
        border-radius: 8px !important;
    }


    /* ---------- BUTTON ---------- */

    div.stButton > button {
        width: 100%;
        min-height: 50px;
        border-radius: 10px;
        border: none;
        font-weight: 700;
        font-size: 15px;
        background: linear-gradient(
            135deg,
            #1d4ed8,
            #2563eb
        );
        color: white;
    }

    div.stButton > button:hover {
        background: linear-gradient(
            135deg,
            #1e40af,
            #1d4ed8
        );
        color: white;
    }


    /* ---------- METRICS ---------- */

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }


    /* ---------- DATAFRAME ---------- */

    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }


    /* ---------- EXPANDER ---------- */

    details {
        background: white;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }


    /* ---------- DIVIDER ---------- */

    hr {
        border-color: #e2e8f0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PATHS
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
            f"Model not found: {MODEL_PATH}"
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
        "Unable to load the trained model."
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

    # Average charges

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

    services = [

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
        value == "Yes"
        for value in services
    )


    customer = pd.DataFrame(
        [{
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
        }]
    )

    return customer


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("📊 ChurnIQ")

    st.caption(
        "Customer Analytics & Churn Prediction"
    )

    st.divider()

    st.subheader("🤖 Model")

    st.write(
        "XGBoost classification model with "
        "engineered customer features."
    )

    st.divider()

    st.subheader("🔍 Explainability")

    st.write(
        "SHAP TreeExplainer is used to explain "
        "individual predictions."
    )

    st.divider()

    st.subheader("📋 How to use")

    st.markdown(
        """
        **1.** Enter customer information.

        **2.** Enter billing information.

        **3.** Click **Analyze Customer Churn**.

        **4.** Review probability and SHAP explanation.
        """
    )

    st.divider()

    st.info(
        "Predictions are model-based estimates "
        "and should be interpreted with business context."
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.title(
    "📊 Customer Churn Prediction"
)

st.write(
    "Predict customer churn probability and "
    "understand the key factors influencing "
    "each prediction."
)

st.divider()


# =========================================================
# CUSTOMER PROFILE
# =========================================================

st.header(
    "👤 Customer Profile"
)

st.caption(
    "Enter demographic, subscription and service information."
)


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
# BILLING
# =========================================================

st.divider()

st.header(
    "💳 Billing Information"
)

st.caption(
    "Enter the customer's monthly and accumulated charges."
)


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
        value=840.0,
        step=10.0
    )


# =========================================================
# ANALYZE BUTTON
# =========================================================

st.divider()

analyze = st.button(
    "🔮 Analyze Customer Churn",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if analyze:

    try:

        # -------------------------------------------------
        # CREATE CUSTOMER
        # -------------------------------------------------

        customer = create_customer(

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
        )


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        prediction = (
            best_model.predict(
                customer
            )[0]
        )


        probability = (
            best_model
            .predict_proba(
                customer
            )[0, 1]
        )


        stay_probability = (
            1 - probability
        )


        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.header(
            "📈 Prediction Result"
        )


        result1, result2, result3 = (
            st.columns(3)
        )


        with result1:

            if prediction == 1:

                st.metric(
                    "Prediction",
                    "⚠️ CHURN"
                )

            else:

                st.metric(
                    "Prediction",
                    "✅ STAY"
                )


        with result2:

            st.metric(
                "Churn Probability",
                f"{probability:.2%}"
            )


        with result3:

            st.metric(
                "Stay Probability",
                f"{stay_probability:.2%}"
            )


        # =================================================
        # RISK MESSAGE
        # =================================================

        st.subheader(
            "Risk Assessment"
        )


        if probability >= 0.50:

            st.error(
                f"⚠️ Higher churn risk — "
                f"estimated probability: "
                f"{probability:.2%}"
            )

        else:

            st.success(
                f"✅ Lower churn risk — "
                f"estimated probability: "
                f"{probability:.2%}"
            )


        st.progress(
            float(probability)
        )


        # =================================================
        # SHAP EXPLANATION
        # =================================================

        st.divider()

        st.header(
            "🔍 Prediction Explanation"
        )

        st.caption(
            "SHAP identifies which features contributed "
            "toward or against the churn prediction."
        )


        if explainer is not None:

            # ---------------------------------------------
            # TRANSFORM CUSTOMER
            # ---------------------------------------------

            customer_transformed = (
                preprocessor.transform(
                    customer
                )
            )


            # ---------------------------------------------
            # SHAP
            # ---------------------------------------------

            customer_shap = (
                explainer.shap_values(
                    customer_transformed
                )
            )


            shap_array = np.asarray(
                customer_shap
            )


            # ---------------------------------------------
            # HANDLE SHAP OUTPUT
            # ---------------------------------------------

            if isinstance(
                customer_shap,
                list
            ):

                if len(customer_shap) > 1:

                    shap_values = (
                        np.asarray(
                            customer_shap[1]
                        )[0]
                    )

                else:

                    shap_values = (
                        np.asarray(
                            customer_shap[0]
                        )[0]
                    )

            elif shap_array.ndim == 3:

                # (samples, features, classes)

                shap_values = (
                    shap_array[0, :, 1]
                )

            elif shap_array.ndim == 2:

                shap_values = (
                    shap_array[0]
                )

            else:

                shap_values = (
                    shap_array.flatten()
                )


            shap_values = np.asarray(
                shap_values
            ).flatten()


            # ---------------------------------------------
            # FEATURE NAMES
            # ---------------------------------------------

            feature_names = (
                preprocessor
                .get_feature_names_out()
            )


            # ---------------------------------------------
            # SAFETY CHECK
            # ---------------------------------------------

            min_length = min(
                len(feature_names),
                len(shap_values)
            )


            feature_names = (
                feature_names[
                    :min_length
                ]
            )


            shap_values = (
                shap_values[
                    :min_length
                ]
            )


            # ---------------------------------------------
            # DATAFRAME
            # ---------------------------------------------

            shap_df = pd.DataFrame(
                {
                    "Feature":
                        feature_names,

                    "SHAP Value":
                        shap_values
                }
            )


            shap_df[
                "Absolute Impact"
            ] = (
                shap_df[
                    "SHAP Value"
                ].abs()
            )


            shap_df = (
                shap_df
                .sort_values(
                    "Absolute Impact",
                    ascending=False
                )
                .head(12)
            )


            # ---------------------------------------------
            # PLOT
            # ---------------------------------------------

            plot_df = (
                shap_df
                .sort_values(
                    "SHAP Value"
                )
            )


            fig, ax = plt.subplots(
                figsize=(11, 6)
            )


            colors = [

                "#dc2626"
                if value > 0
                else "#2563eb"

                for value
                in plot_df["SHAP Value"]

            ]


            ax.barh(
                plot_df["Feature"],
                plot_df["SHAP Value"],
                color=colors
            )


            ax.axvline(
                0,
                color="#475569",
                linewidth=1
            )


            ax.set_title(
                "Top Factors Influencing Churn",
                fontsize=16,
                fontweight="bold",
                pad=15
            )


            ax.set_xlabel(
                "SHAP Contribution"
            )


            ax.grid(
                axis="x",
                linestyle="--",
                alpha=0.20
            )


            ax.spines[
                "top"
            ].set_visible(False)

            ax.spines[
                "right"
            ].set_visible(False)

            ax.spines[
                "left"
            ].set_visible(False)


            # ---------------------------------------------
            # VALUE LABELS
            # ---------------------------------------------

            max_abs = max(
                abs(
                    plot_df[
                        "SHAP Value"
                    ]
                )
            )


            for index, value in enumerate(
                plot_df["SHAP Value"]
            ):

                offset = (
                    max_abs * 0.02
                )


                if value >= 0:

                    ax.text(
                        value + offset,
                        index,
                        f"{value:+.3f}",
                        va="center",
                        fontsize=9
                    )

                else:

                    ax.text(
                        value - offset,
                        index,
                        f"{value:+.3f}",
                        va="center",
                        ha="right",
                        fontsize=9
                    )


            plt.tight_layout()


            st.pyplot(
                fig,
                use_container_width=True
            )


            plt.close(fig)


            # ---------------------------------------------
            # LEGEND
            # ---------------------------------------------

            st.info(
                "🔴 Positive SHAP values increase the "
                "model's churn output. "
                "🔵 Negative SHAP values decrease it. "
                "Larger absolute values indicate stronger "
                "model influence."
            )


            # =================================================
            # SHAP TABLE
            # =================================================

            with st.expander(
                "📋 View Detailed SHAP Contributions"
            ):

                table_df = shap_df[
                    [
                        "Feature",
                        "SHAP Value"
                    ]
                ].copy()


                table_df[
                    "Direction"
                ] = np.where(
                    table_df[
                        "SHAP Value"
                    ] > 0,
                    "Increases Churn",
                    "Reduces Churn"
                )


                table_df[
                    "SHAP Value"
                ] = table_df[
                    "SHAP Value"
                ].round(4)


                st.dataframe(
                    table_df,
                    use_container_width=True,
                    hide_index=True
                )


        else:

            st.warning(
                "SHAP explanation is unavailable."
            )


        # =================================================
        # CUSTOMER DATA
        # =================================================

        st.divider()

        with st.expander(
            "👤 View Customer Input & Engineered Features"
        ):

            st.dataframe(
                customer.T.rename(
                    columns={
                        0: "Value"
                    }
                ),
                use_container_width=True
            )


        # =================================================
        # MODEL INFORMATION
        # =================================================

        with st.expander(
            "🤖 Model Information"
        ):

            info1, info2 = st.columns(2)


            with info1:

                st.write(
                    "**Algorithm:** XGBoost"
                )

                st.write(
                    "**Preprocessing:** "
                    "ColumnTransformer"
                )

                st.write(
                    "**Feature Engineering:** "
                    "AvgChargesPerMonth, "
                    "TenureGroup, NumServices"
                )


            with info2:

                st.write(
                    "**Explainability:** SHAP"
                )

                st.write(
                    "**Prediction Type:** "
                    "Binary Classification"
                )

                st.write(
                    "**Output:** Churn Probability"
                )


    except Exception as e:

        st.error(
            "Something went wrong while "
            "generating the prediction."
        )

        with st.expander(
            "Technical Details"
        ):

            st.code(
                str(e)
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "ChurnIQ • Customer Churn Prediction System • "
    "Python • Streamlit • XGBoost • SHAP"
)
