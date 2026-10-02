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
# DARK PROFESSIONAL THEME
# FONT: TIMES NEW ROMAN
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
    ===================================================== */

    * {
        font-family: "Times New Roman", Times, serif !important;
    }

    html,
    body,
    [class*="css"] {
        font-family: "Times New Roman", Times, serif !important;
    }

    .stApp {
        background: #0b0f19;
        color: #f8fafc;
    }

    .main {
        background: #0b0f19;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }


    /* =====================================================
       MAIN TEXT
    ===================================================== */

    h1,
    h2,
    h3,
    h4,
    h5,
    h6 {
        font-family: "Times New Roman", Times, serif !important;
        color: #f8fafc !important;
        font-weight: 700 !important;
    }

    p,
    span,
    label,
    div {
        font-family: "Times New Roman", Times, serif !important;
    }

    p {
        color: #cbd5e1;
    }


    /* =====================================================
       SIDEBAR
    ===================================================== */

    section[data-testid="stSidebar"] {
        background: #070b13 !important;
        border-right: 1px solid #1e293b;
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }


    /* =====================================================
       INPUT LABELS
    ===================================================== */

    label p {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }


    /* =====================================================
       SELECTBOX
    ===================================================== */

    div[data-baseweb="select"] > div {
        background: #151b28 !important;
        border: 1px solid #334155 !important;
        border-radius: 9px !important;
        color: #ffffff !important;
    }

    div[data-baseweb="select"] span {
        color: #ffffff !important;
    }

    div[data-baseweb="select"] input {
        color: #ffffff !important;
    }


    /* Dropdown */

    ul[role="listbox"] {
        background: #111827 !important;
        border: 1px solid #334155 !important;
    }

    li[role="option"] {
        background: #111827 !important;
        color: #f8fafc !important;
    }

    li[role="option"]:hover {
        background: #1e293b !important;
    }


    /* =====================================================
       NUMBER INPUT
    ===================================================== */

    div[data-testid="stNumberInput"] {
        background: transparent !important;
    }

    div[data-testid="stNumberInput"] input {
        background: #151b28 !important;
        color: #ffffff !important;
        border: 1px solid #334155 !important;
        border-radius: 9px !important;
    }


    /* =====================================================
       BUTTON
    ===================================================== */

    div.stButton > button {
        width: 100%;
        min-height: 54px;

        border-radius: 10px;

        border: 1px solid #3b82f6;

        background: linear-gradient(
            135deg,
            #1d4ed8,
            #2563eb
        ) !important;

        color: white !important;

        font-family: "Times New Roman", Times, serif !important;

        font-size: 17px !important;
        font-weight: 700 !important;

        box-shadow:
            0 8px 25px rgba(
                37,
                99,
                235,
                0.25
            );
    }

    div.stButton > button:hover {
        background: linear-gradient(
            135deg,
            #2563eb,
            #3b82f6
        ) !important;

        border-color: #60a5fa;
    }


    /* =====================================================
       METRICS
    ===================================================== */

    div[data-testid="stMetric"] {
        background: #111827 !important;

        border: 1px solid #263244 !important;

        border-radius: 14px !important;

        padding: 20px !important;

        box-shadow:
            0 8px 25px rgba(
                0,
                0,
                0,
                0.25
            );
    }

    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-family: "Times New Roman", Times, serif !important;
    }

    div[data-testid="stMetricDelta"] {
        color: #cbd5e1 !important;
    }


    /* =====================================================
       ALERTS
    ===================================================== */

    div[data-testid="stAlert"] {
        border-radius: 10px !important;
    }


    /* =====================================================
       EXPANDER
    ===================================================== */

    details {
        background: #111827 !important;

        border: 1px solid #263244 !important;

        border-radius: 10px !important;
    }

    details summary {
        color: #f8fafc !important;
    }


    /* =====================================================
       DATAFRAME
    ===================================================== */

    div[data-testid="stDataFrame"] {
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }


    /* =====================================================
       DIVIDER
    ===================================================== */

    hr {
        border-color: #263244 !important;
    }


    /* =====================================================
       CAPTION
    ===================================================== */

    .stCaption {
        color: #94a3b8 !important;
    }


    /* =====================================================
       PROGRESS BAR
    ===================================================== */

    div[data-testid="stProgress"] > div {
        background: #1e293b !important;
    }

    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(
            90deg,
            #2563eb,
            #3b82f6
        ) !important;
    }


    /* =====================================================
       CHECKBOX / RADIO
    ===================================================== */

    div[data-testid="stCheckbox"] label,
    div[data-testid="stRadio"] label {
        color: #e2e8f0 !important;
    }


    /* =====================================================
       SCROLLBAR
    ===================================================== */

    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #0b0f19;
    }

    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
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
            f"""
Model file was not found.

Expected location:
{MODEL_PATH}
"""
        )

    return joblib.load(MODEL_PATH)


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
# PIPELINE COMPONENTS
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
        "❌ The saved pipeline does not contain "
        "the expected 'preprocessor' and 'model' steps."
    )

    st.code(
        str(e)
    )

    st.stop()


# =========================================================
# SHAP EXPLAINER
# =========================================================

@st.cache_resource
def create_shap_explainer(model):

    errors = []

    # ---------------------------------------------
    # METHOD 1
    # ---------------------------------------------

    try:

        return (
            shap.TreeExplainer(
                model
            ),
            "TreeExplainer"
        )

    except Exception as e:

        errors.append(
            f"TreeExplainer: {e}"
        )


    # ---------------------------------------------
    # METHOD 2
    # ---------------------------------------------

    try:

        if hasattr(
            model,
            "get_booster"
        ):

            booster = (
                model.get_booster()
            )

            return (
                shap.TreeExplainer(
                    booster
                ),
                "Booster TreeExplainer"
            )

    except Exception as e:

        errors.append(
            f"Booster TreeExplainer: {e}"
        )


    raise RuntimeError(
        "\n\n".join(errors)
    )


try:

    explainer, explainer_type = (
        create_shap_explainer(
            xgb_model
        )
    )

    shap_available = True

except Exception as e:

    explainer = None
    explainer_type = None
    shap_available = False

    shap_error = str(e)


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

    AvgChargesPerMonth = (
        TotalCharges /
        (tenure + 1)
    )


    if tenure <= 12:

        TenureGroup = "0-1yr"

    elif tenure <= 24:

        TenureGroup = "1-2yr"

    elif tenure <= 48:

        TenureGroup = "2-4yr"

    else:

        TenureGroup = "4-6yr"


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


    return pd.DataFrame(
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


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("📊 ChurnIQ")

    st.caption(
        "Customer Analytics & Churn Prediction"
    )

    st.divider()

    st.subheader(
        "🤖 Machine Learning"
    )

    st.write(
        "XGBoost classification model "
        "with engineered customer features."
    )

    st.divider()

    st.subheader(
        "🔍 Explainability"
    )

    if shap_available:

        st.success(
            f"SHAP active • {explainer_type}"
        )

    else:

        st.warning(
            "SHAP initialization failed."
        )

    st.divider()

    st.subheader(
        "📋 Workflow"
    )

    st.markdown(
        """
        **01 — Customer Profile**

        Enter demographic and service details.

        **02 — Billing**

        Enter monthly and total charges.

        **03 — Analyze**

        Generate churn probability.

        **04 — Explain**

        Review SHAP feature contributions.
        """
    )

    st.divider()

    st.caption(
        "Built with Python • Streamlit • "
        "XGBoost • SHAP"
    )


# =========================================================
# HERO
# =========================================================

st.title(
    "📊 Customer Churn Prediction"
)

st.write(
    "Predict churn probability and understand "
    "the model's key decision drivers."
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
# ANALYZE
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
        # CUSTOMER
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
        # PREDICT
        # -------------------------------------------------

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


        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.header(
            "📈 Prediction Result"
        )


        r1, r2, r3 = st.columns(3)


        with r1:

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


        with r2:

            st.metric(
                "Churn Probability",
                f"{probability:.2%}"
            )


        with r3:

            st.metric(
                "Stay Probability",
                f"{stay_probability:.2%}"
            )


        # =================================================
        # RISK
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
            probability
        )


        # =================================================
        # SHAP
        # =================================================

        st.divider()

        st.header(
            "🔍 Prediction Explanation"
        )

        st.caption(
            "The chart shows which transformed features "
            "pushed the model toward or away from churn."
        )


        # -------------------------------------------------
        # TRANSFORM
        # -------------------------------------------------

        customer_transformed = (
            preprocessor.transform(
                customer
            )
        )


        # Convert sparse matrix if necessary

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


        # -------------------------------------------------
        # SHAP CALCULATION
        # -------------------------------------------------

        shap_success = False

        if explainer is not None:

            try:

                raw_shap = (
                    explainer.shap_values(
                        customer_transformed
                    )
                )


                # =========================================
                # SHAP OUTPUT HANDLING
                # =========================================

                if isinstance(
                    raw_shap,
                    list
                ):

                    if len(raw_shap) > 1:

                        shap_values = np.asarray(
                            raw_shap[1]
                        )[0]

                    else:

                        shap_values = np.asarray(
                            raw_shap[0]
                        )[0]


                else:

                    raw_array = np.asarray(
                        raw_shap
                    )


                    if raw_array.ndim == 3:

                        # samples, features, classes

                        shap_values = (
                            raw_array[0, :, 1]
                        )


                    elif raw_array.ndim == 2:

                        shap_values = (
                            raw_array[0]
                        )


                    elif raw_array.ndim == 1:

                        shap_values = (
                            raw_array
                        )

                    else:

                        shap_values = (
                            raw_array.flatten()
                        )


                shap_values = np.asarray(
                    shap_values,
                    dtype=float
                ).flatten()


                shap_success = True


            except Exception as shap_calc_error:

                shap_success = False

                shap_error = (
                    "SHAP calculation error:\n\n"
                    + str(shap_calc_error)
                )


        # =================================================
        # SHAP GRAPH
        # =================================================

        if shap_success:

            # ---------------------------------------------
            # FEATURE NAMES
            # ---------------------------------------------

            feature_names = (
                preprocessor
                .get_feature_names_out()
            )


            # ---------------------------------------------
            # LENGTH SAFETY
            # ---------------------------------------------

            n = min(
                len(feature_names),
                len(shap_values)
            )


            feature_names = (
                feature_names[:n]
            )

            shap_values = (
                shap_values[:n]
            )


            # ---------------------------------------------
            # SHAP DATAFRAME
            # ---------------------------------------------

            shap_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "SHAP Value": shap_values
                }
            )


            shap_df[
                "Absolute Impact"
            ] = (
                shap_df[
                    "SHAP Value"
                ].abs()
            )


            # Top 12

            top_shap = (
                shap_df
                .sort_values(
                    "Absolute Impact",
                    ascending=False
                )
                .head(12)
                .copy()
            )


            # Sort for horizontal graph

            plot_df = (
                top_shap
                .sort_values(
                    "SHAP Value"
                )
            )


            # ---------------------------------------------
            # FIGURE
            # ---------------------------------------------

            fig, ax = plt.subplots(
                figsize=(13, 7)
            )


            fig.patch.set_facecolor(
                "#111827"
            )

            ax.set_facecolor(
                "#111827"
            )


            colors = [

                "#ef4444"
                if x > 0
                else "#38bdf8"

                for x
                in plot_df[
                    "SHAP Value"
                ]

            ]


            bars = ax.barh(
                plot_df[
                    "Feature"
                ],
                plot_df[
                    "SHAP Value"
                ],
                color=colors,
                edgecolor="none",
                height=0.65
            )


            # ---------------------------------------------
            # ZERO LINE
            # ---------------------------------------------

            ax.axvline(
                0,
                color="#94a3b8",
                linewidth=1.2
            )


            # ---------------------------------------------
            # GRID
            # ---------------------------------------------

            ax.grid(
                axis="x",
                color="#334155",
                linestyle="--",
                linewidth=0.7,
                alpha=0.6
            )


            # ---------------------------------------------
            # TITLE
            # ---------------------------------------------

            ax.set_title(
                "Top SHAP Contributions to Churn",
                color="#f8fafc",
                fontsize=18,
                fontweight="bold",
                pad=20,
                fontfamily="Times New Roman"
            )


            ax.set_xlabel(
                "SHAP Value  →  Impact on Churn",
                color="#cbd5e1",
                fontsize=12,
                fontfamily="Times New Roman"
            )


            # ---------------------------------------------
            # TICK COLORS
            # ---------------------------------------------

            ax.tick_params(
                axis="x",
                colors="#cbd5e1",
                labelsize=10
            )


            ax.tick_params(
                axis="y",
                colors="#e2e8f0",
                labelsize=10
            )


            for label in ax.get_xticklabels():

                label.set_fontfamily(
                    "Times New Roman"
                )


            for label in ax.get_yticklabels():

                label.set_fontfamily(
                    "Times New Roman"
                )


            # ---------------------------------------------
            # SPINES
            # ---------------------------------------------

            ax.spines[
                "top"
            ].set_visible(False)

            ax.spines[
                "right"
            ].set_visible(False)

            ax.spines[
                "left"
            ].set_visible(False)

            ax.spines[
                "bottom"
            ].set_color(
                "#334155"
            )


            # ---------------------------------------------
            # VALUE LABELS
            # ---------------------------------------------

            max_value = max(
                abs(
                    plot_df[
                        "SHAP Value"
                    ]
                )
            )


            if max_value == 0:

                max_value = 1


            offset = (
                max_value * 0.025
            )


            for bar, value in zip(
                bars,
                plot_df[
                    "SHAP Value"
                ]
            ):

                y = (
                    bar.get_y()
                    +
                    bar.get_height() / 2
                )


                if value >= 0:

                    ax.text(
                        value + offset,
                        y,
                        f"{value:+.3f}",
                        va="center",
                        ha="left",
                        color="#f8fafc",
                        fontsize=10,
                        fontweight="bold",
                        fontfamily="Times New Roman"
                    )

                else:

                    ax.text(
                        value - offset,
                        y,
                        f"{value:+.3f}",
                        va="center",
                        ha="right",
                        color="#f8fafc",
                        fontsize=10,
                        fontweight="bold",
                        fontfamily="Times New Roman"
                    )


            # ---------------------------------------------
            # LEGEND TEXT
            # ---------------------------------------------

            ax.text(
                0.01,
                -0.15,
                "🔴 Positive = pushes toward churn     "
                "🔵 Negative = pushes away from churn",
                transform=ax.transAxes,
                color="#94a3b8",
                fontsize=10,
                fontfamily="Times New Roman"
            )


            plt.tight_layout()


            # ---------------------------------------------
            # DISPLAY
            # ---------------------------------------------

            st.pyplot(
                fig,
                use_container_width=True
            )


            plt.close(fig)


            # =================================================
            # SHAP TABLE
            # =================================================

            with st.expander(
                "📋 Detailed SHAP Contributions"
            ):

                detail_df = top_shap[
                    [
                        "Feature",
                        "SHAP Value"
                    ]
                ].copy()


                detail_df[
                    "Direction"
                ] = np.where(
                    detail_df[
                        "SHAP Value"
                    ] > 0,
                    "Toward Churn",
                    "Away From Churn"
                )


                detail_df[
                    "SHAP Value"
                ] = detail_df[
                    "SHAP Value"
                ].round(4)


                st.dataframe(
                    detail_df,
                    use_container_width=True,
                    hide_index=True
                )


        # =================================================
        # SHAP FAILED
        # =================================================

        else:

            st.error(
                "❌ SHAP graph could not be generated."
            )

            with st.expander(
                "🔧 SHAP Technical Details"
            ):

                if "shap_error" in locals():

                    st.code(
                        shap_error
                    )

                else:

                    st.code(
                        "Unknown SHAP error."
                    )


            # ---------------------------------------------
            # FALLBACK FEATURE IMPORTANCE
            # ---------------------------------------------

            if hasattr(
                xgb_model,
                "feature_importances_"
            ):

                st.warning(
                    "Showing model feature importance "
                    "as a fallback. This is NOT SHAP."
                )


                feature_names = (
                    preprocessor
                    .get_feature_names_out()
                )


                importance = (
                    np.asarray(
                        xgb_model
                        .feature_importances_
                    )
                )


                n = min(
                    len(feature_names),
                    len(importance)
                )


                importance_df = pd.DataFrame(
                    {
                        "Feature":
                            feature_names[:n],

                        "Importance":
                            importance[:n]
                    }
                )


                importance_df = (
                    importance_df
                    .sort_values(
                        "Importance",
                        ascending=False
                    )
                    .head(12)
                )


                fig, ax = plt.subplots(
                    figsize=(12, 6)
                )


                fig.patch.set_facecolor(
                    "#111827"
                )

                ax.set_facecolor(
                    "#111827"
                )


                plot_importance = (
                    importance_df
                    .sort_values(
                        "Importance"
                    )
                )


                ax.barh(
                    plot_importance[
                        "Feature"
                    ],
                    plot_importance[
                        "Importance"
                    ],
                    color="#3b82f6"
                )


                ax.set_title(
                    "Model Feature Importance",
                    color="#f8fafc",
                    fontsize=17,
                    fontweight="bold",
                    fontfamily="Times New Roman"
                )


                ax.tick_params(
                    colors="#e2e8f0"
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

                ax.spines[
                    "bottom"
                ].set_color(
                    "#334155"
                )


                plt.tight_layout()


                st.pyplot(
                    fig,
                    use_container_width=True
                )


                plt.close(fig)


        # =================================================
        # CUSTOMER DATA
        # =================================================

        st.divider()

        with st.expander(
            "👤 Customer Input & Engineered Features"
        ):

            display_customer = (
                customer.T
                .rename(
                    columns={
                        0: "Value"
                    }
                )
            )


            st.dataframe(
                display_customer,
                use_container_width=True
            )


        # =================================================
        # MODEL INFO
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
                    "**Prediction:** Binary Classification"
                )

                st.write(
                    "**Output:** Churn Probability"
                )


    except Exception as e:

        st.error(
            "❌ An error occurred while "
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
    "XGBoost • SHAP • Streamlit"
)
