import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
except Exception:
    shap = None
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROFESSIONAL DARK THEME
# =========================================================

st.markdown(
    """
    <style>

    /* ===============================
       GLOBAL
    =============================== */

    .stApp {
        background-color: #0b1220;
        color: #f8fafc;
    }

    .main {
        background-color: #0b1220;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* ===============================
       SIDEBAR
    =============================== */

    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #cbd5e1;
    }


    /* ===============================
       HEADINGS
    =============================== */

    h1 {
        color: #f8fafc !important;
        font-weight: 800 !important;
    }

    h2 {
        color: #f8fafc !important;
        font-weight: 750 !important;
    }

    h3 {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }


    /* ===============================
       NORMAL TEXT
    =============================== */

    p {
        color: #cbd5e1;
    }


    /* ===============================
       INPUT LABEL
    =============================== */

    label p {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }


    /* ===============================
       SELECTBOX
    =============================== */

    div[data-baseweb="select"] > div {
        background-color: #151e2e !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="select"] span {
        color: #f8fafc !important;
    }


    /* Dropdown menu */

    ul[role="listbox"] {
        background-color: #111827 !important;
    }

    li[role="option"] {
        background-color: #111827 !important;
        color: #f8fafc !important;
    }

    li[role="option"]:hover {
        background-color: #1e293b !important;
    }


    /* ===============================
       NUMBER INPUT
    =============================== */

    div[data-testid="stNumberInput"] input {
        background-color: #151e2e !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }


    /* ===============================
       BUTTON
    =============================== */

    div.stButton > button {
        background: linear-gradient(
            135deg,
            #2563eb,
            #1d4ed8
        ) !important;

        color: white !important;

        border: none !important;

        border-radius: 8px !important;

        height: 48px;

        font-weight: 700 !important;

        box-shadow:
            0 5px 20px rgba(
                37,
                99,
                235,
                0.25
            );
    }

    div.stButton > button:hover {
        background: linear-gradient(
            135deg,
            #3b82f6,
            #2563eb
        ) !important;
    }


    /* ===============================
       METRIC CARDS
    =============================== */

    div[data-testid="stMetric"] {
        background-color: #111827 !important;

        border: 1px solid #263244 !important;

        border-radius: 12px !important;

        padding: 18px !important;
    }

    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }


    /* ===============================
       EXPANDER
    =============================== */

    details {
        background-color: #111827 !important;

        border: 1px solid #263244 !important;

        border-radius: 10px !important;
    }

    details summary {
        color: #f8fafc !important;
    }


    /* ===============================
       DIVIDER
    =============================== */

    hr {
        border-color: #263244 !important;
    }


    /* ===============================
       DATAFRAME
    =============================== */

    div[data-testid="stDataFrame"] {
        border: 1px solid #334155;
        border-radius: 8px;
        overflow: hidden;
    }


    /* ===============================
       PROGRESS
    =============================== */

    div[data-testid="stProgress"] > div {
        background-color: #1e293b !important;
    }

    div[data-testid="stProgress"] > div > div {
        background-color: #2563eb !important;
    }

    </style>
    """,
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
            f"""
Model file not found.

Expected:
{MODEL_PATH}
"""
        )

    return joblib.load(
        MODEL_PATH
    )


try:

    best_model = load_model()

except Exception as e:

    st.error(
        "❌ Model could not be loaded."
    )

    st.code(
        str(e)
    )

    st.stop()


# =========================================================
# GET PIPELINE COMPONENTS
# =========================================================

try:

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
        "❌ Invalid model pipeline."
    )

    st.code(
        str(e)
    )

    st.stop()


# =========================================================
# SHAP EXPLAINER
# =========================================================

@st.cache_resource
def load_explainer(model):

    return shap.TreeExplainer(
        model
    )


try:

    explainer = load_explainer(
        xgb_model
    )

    shap_ready = True
    shap_init_error = None

except Exception as e:

    explainer = None
    shap_ready = False
    shap_init_error = str(e)


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

    # ---------------------------------------------
    # Average Charges Per Month
    # ---------------------------------------------

    AvgChargesPerMonth = (
        TotalCharges /
        (tenure + 1)
    )


    # ---------------------------------------------
    # Tenure Group
    # ---------------------------------------------

    if tenure <= 12:

        TenureGroup = "0-1yr"

    elif tenure <= 24:

        TenureGroup = "1-2yr"

    elif tenure <= 48:

        TenureGroup = "2-4yr"

    else:

        TenureGroup = "4-6yr"


    # ---------------------------------------------
    # Number of Services
    # ---------------------------------------------

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


    # ---------------------------------------------
    # DataFrame
    # ---------------------------------------------

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

    st.caption("✅ App version: v2")

    st.title(
        "📊 ChurnIQ"
    )

    st.caption(
        "Customer Analytics & "
        "Churn Prediction"
    )

    st.divider()

    st.subheader(
        "🤖 Model"
    )

    st.write(
        "XGBoost"
    )

    st.write(
        "Binary Classification"
    )

    st.divider()

    st.subheader(
        "🔍 Explainability"
    )

    if shap_ready:

        st.success(
            "SHAP Explainer Ready"
        )

    else:

        st.error(
            "SHAP initialization failed"
        )

    st.divider()

    st.subheader(
        "📋 How to Use"
    )

    st.markdown(
        """
        **01** Enter customer information.

        **02** Enter billing information.

        **03** Click **Predict Customer Churn**.

        **04** Review the prediction.

        **05** Review the SHAP explanation.
        """
    )

    st.divider()

    st.caption(
        "XGBoost • SHAP • Streamlit"
    )


# =========================================================
# HEADER
# =========================================================

st.title(
    "📊 Customer Churn Prediction"
)

st.write(
    "Predict customer churn probability "
    "and understand the model's decision drivers."
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
# PREDICT BUTTON
# =========================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Customer Churn",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    try:

        # =============================================
        # CREATE CUSTOMER
        # =============================================

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


        # =============================================
        # PREDICTION
        # =============================================

        prediction = int(
            best_model.predict(
                customer
            )[0]
        )


        probability = float(
            best_model.predict_proba(
                customer
            )[0, 1]
        )


        stay_probability = (
            1 - probability
        )


        # =============================================
        # RESULT
        # =============================================

        st.divider()

        st.header(
            "📈 Prediction Result"
        )


        result_col1, result_col2, result_col3 = (
            st.columns(3)
        )


        with result_col1:

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


        with result_col2:

            st.metric(
                "Churn Probability",
                f"{probability:.2%}"
            )


        with result_col3:

            st.metric(
                "Stay Probability",
                f"{stay_probability:.2%}"
            )


        # =============================================
        # RISK ASSESSMENT
        # =============================================

        st.subheader(
            "Risk Assessment"
        )


        if probability >= 0.50:

            st.warning(
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
            probability
        )


        # =============================================
        # SHAP EXPLANATION
        # =============================================

        st.divider()

        st.header(
            "🔍 Prediction Explanation"
        )

        st.caption(
            "SHAP explains how the transformed model "
            "features contributed to this customer's prediction."
        )


        # =============================================
        # TRANSFORM CUSTOMER
        # =============================================

        customer_transformed = (
            preprocessor.transform(
                customer
            )
        )


        # Convert sparse matrix to numpy

        if hasattr(
            customer_transformed,
            "toarray"
        ):

            customer_transformed = (
                customer_transformed.toarray()
            )


        customer_transformed = np.asarray(
            customer_transformed
        )


        # =============================================
        # FEATURE NAMES
        # =============================================

        shap_feature_names = (
            preprocessor
            .get_feature_names_out()
        )


        # =============================================
        # SHAP CALCULATION
        # =============================================

        raw_shap = None
        contrib_base = None

        if shap_ready:

            try:

                raw_shap = (
                    explainer.shap_values(
                        customer_transformed
                    )
                )

            except Exception:

                raw_shap = None


        if raw_shap is None:

            # Fallback: native XGBoost SHAP contributions
            # (works even when shap/xgboost versions mismatch)

            import xgboost as xgb

            contribs = (
                xgb_model
                .get_booster()
                .predict(
                    xgb.DMatrix(
                        customer_transformed
                    ),
                    pred_contribs=True,
                    validate_features=False
                )
            )

            raw_shap = contribs[:, :-1]

            contrib_base = float(
                contribs[0, -1]
            )


        # =============================================
        # ROBUST SHAP OUTPUT HANDLING
        # =============================================

        if isinstance(
            raw_shap,
            list
        ):

            # Old SHAP format:
            # [class_0_values, class_1_values]

            if len(raw_shap) > 1:

                shap_array = np.asarray(
                    raw_shap[1]
                )

            else:

                shap_array = np.asarray(
                    raw_shap[0]
                )


        else:

            shap_array = np.asarray(
                raw_shap
            )


        # ---------------------------------------------
        # Shape handling
        # ---------------------------------------------

        if shap_array.ndim == 3:

            # Possible shape:
            # samples × features × classes

            shap_values = (
                shap_array[
                    0,
                    :,
                    1
                ]
            )


        elif shap_array.ndim == 2:

            # Normal binary XGBoost SHAP:
            # samples × features

            shap_values = (
                shap_array[0]
            )


        elif shap_array.ndim == 1:

            shap_values = (
                shap_array
            )


        else:

            raise ValueError(
                "Unexpected SHAP output shape: "
                + str(
                    shap_array.shape
                )
            )


        shap_values = np.asarray(
            shap_values,
            dtype=float
        ).flatten()


        # =============================================
        # BASE VALUE
        # =============================================

        expected_value = (
            explainer.expected_value
            if explainer is not None
            else 0.0
        )


        if isinstance(
            expected_value,
            (list, np.ndarray)
        ):

            expected_array = np.asarray(
                expected_value
            ).flatten()


            if len(expected_array) > 1:

                base_value = float(
                    expected_array[1]
                )

            else:

                base_value = float(
                    expected_array[0]
                )

        else:

            base_value = float(
                expected_value
            )


        if contrib_base is not None:

            base_value = contrib_base


        # =============================================
        # SAFETY CHECK
        # =============================================

        if len(shap_values) != len(
            shap_feature_names
        ):

            raise ValueError(
                "SHAP feature count does not match "
                "the transformed feature count.\n\n"
                f"SHAP values: {len(shap_values)}\n"
                f"Feature names: {len(shap_feature_names)}"
            )


        # =============================================
        # SHAP EXPLANATION OBJECT
        # =============================================

        shap_explanation = None if shap is None else shap.Explanation(

            values=shap_values,

            base_values=base_value,

            data=customer_transformed[0],

            feature_names=shap_feature_names

        )


        # =============================================
        # SHAP CHART (native Streamlit/Altair - always renders)
        # =============================================

        import altair as alt

        chart_df = pd.DataFrame(
            {
                "Feature": [
                    str(f) for f in shap_feature_names
                ],
                "SHAP Value": shap_values
            }
        )

        chart_df["Abs"] = (
            chart_df["SHAP Value"].abs()
        )

        chart_df = (
            chart_df
            .sort_values(
                "Abs",
                ascending=False
            )
            .head(15)
        )

        chart_df["Direction"] = np.where(
            chart_df["SHAP Value"] > 0,
            "Toward Churn",
            "Away From Churn"
        )

        shap_chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "SHAP Value:Q",
                    title="SHAP value (impact on churn)"
                ),
                y=alt.Y(
                    "Feature:N",
                    sort=alt.EncodingSortField(
                        field="Abs",
                        order="descending"
                    ),
                    title=None
                ),
                color=alt.Color(
                    "Direction:N",
                    scale=alt.Scale(
                        domain=[
                            "Toward Churn",
                            "Away From Churn"
                        ],
                        range=[
                            "#ef4444",
                            "#3b82f6"
                        ]
                    ),
                    legend=alt.Legend(
                        title=None,
                        orient="bottom"
                    )
                ),
                tooltip=[
                    "Feature",
                    alt.Tooltip(
                        "SHAP Value:Q",
                        format=".4f"
                    ),
                    "Direction"
                ]
            )
            .properties(
                width="container",
                height=440,
                title="SHAP Explanation — Customer Churn Prediction"
            )
            .configure(
                background="#111827"
            )
            .configure_axis(
                labelColor="#cbd5e1",
                titleColor="#cbd5e1",
                gridColor="#263244",
                domainColor="#334155"
            )
            .configure_legend(
                labelColor="#cbd5e1"
            )
            .configure_title(
                color="#f8fafc",
                fontSize=16
            )
            .configure_view(
                strokeWidth=0
            )
        )

        st.altair_chart(
            shap_chart,
            theme=None
        )


        # =============================================
        # OPTIONAL: CLASSIC SHAP WATERFALL
        # =============================================

        with st.expander(
            "📉 View classic SHAP waterfall plot"
        ):

            try:

                plt.figure(
                    figsize=(10, 7)
                )

                shap.plots.waterfall(
                    shap_explanation,
                    max_display=15,
                    show=False
                )

                waterfall_fig = plt.gcf()

                st.pyplot(
                    waterfall_fig
                )

                plt.close(
                    waterfall_fig
                )

            except Exception as waterfall_error:

                st.warning(
                    "Waterfall plot could not be drawn "
                    "(the chart above is unaffected)."
                )

                st.code(
                    str(waterfall_error)
                )



        st.caption(
            "Positive SHAP contributions push the model "
            "toward churn, while negative contributions "
            "push it away from churn. The SHAP output "
            "represents model-output contribution, not "
            "a direct percentage probability contribution."
        )


        # =============================================
        # SHAP CONTRIBUTION TABLE
        # =============================================

        with st.expander(
            "📋 View SHAP Feature Contributions"
        ):

            shap_table = pd.DataFrame(
                {
                    "Feature":
                        shap_feature_names,

                    "SHAP Value":
                        shap_values
                }
            )


            shap_table[
                "Absolute Impact"
            ] = (
                shap_table[
                    "SHAP Value"
                ].abs()
            )


            shap_table = (
                shap_table
                .sort_values(
                    "Absolute Impact",
                    ascending=False
                )
                .head(15)
            )


            shap_table[
                "Direction"
            ] = np.where(
                shap_table[
                    "SHAP Value"
                ] > 0,

                "Toward Churn",

                "Away From Churn"
            )


            shap_table[
                "SHAP Value"
            ] = shap_table[
                "SHAP Value"
            ].round(4)


            shap_table[
                "Absolute Impact"
            ] = shap_table[
                "Absolute Impact"
            ].round(4)


            st.dataframe(
                shap_table[
                    [
                        "Feature",
                        "SHAP Value",
                        "Direction"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )


        # =============================================
        # CUSTOMER DATA
        # =============================================

        with st.expander(
            "👤 View Customer Input & Engineered Features"
        ):

            st.dataframe(
                customer.T,
                use_container_width=True
            )


        # =============================================
        # MODEL INFORMATION
        # =============================================

        with st.expander(
            "🤖 Model Information"
        ):

            info_col1, info_col2 = (
                st.columns(2)
            )


            with info_col1:

                st.write(
                    "**Algorithm:** XGBoost"
                )

                st.write(
                    "**Task:** Binary Classification"
                )

                st.write(
                    "**Preprocessing:** "
                    "ColumnTransformer"
                )


            with info_col2:

                st.write(
                    "**Explainability:** SHAP"
                )

                st.write(
                    "**Output:** Churn Probability"
                )

                st.write(
                    "**Engineered Features:** "
                    "AvgChargesPerMonth, "
                    "TenureGroup, NumServices"
                )


    except Exception as e:

        st.error(
            "❌ Prediction or SHAP explanation "
            "could not be generated."
        )


        st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "ChurnIQ • Customer Churn Prediction System • "
    "XGBoost • SHAP • Streamlit"
)
