```python
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
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# DARK UI
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #0b1220;
        color: #f8fafc;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }

    section[data-testid="stSidebar"] * {
        color: #dbe4f0;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
    }

    p, .stCaption {
        color: #b8c4d6;
    }

    label p {
        color: #dbe4f0 !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="select"] > div {
        background: #151e2e !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="select"] span {
        color: #f8fafc !important;
    }

    div[data-testid="stNumberInput"] input {
        background: #151e2e !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
    }

    div.stButton > button {
        background: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        min-height: 46px;
        font-weight: 700 !important;
    }

    div.stButton > button:hover {
        background: #1d4ed8 !important;
    }

    div[data-testid="stMetric"] {
        background: #111827 !important;
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

    details {
        background: #111827 !important;
        border: 1px solid #263244 !important;
        border-radius: 10px !important;
    }

    details summary {
        color: #f8fafc !important;
    }

    hr {
        border-color: #263244 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.pkl"


# =========================================================
# LOAD FINAL DEPLOYMENT BUNDLE
# =========================================================

@st.cache_resource
def load_model_bundle():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}\n\n"
            "Put churn_pipeline.pkl inside the models folder."
        )

    bundle = joblib.load(MODEL_PATH)

    required_keys = {
        "model",
        "scaler",
        "encoder",
        "feature_names",
        "threshold",
    }

    missing_keys = required_keys - set(bundle.keys())

    if missing_keys:
        raise ValueError(
            "The saved deployment bundle is missing:\n"
            + ", ".join(sorted(missing_keys))
        )

    return bundle


try:
    model_bundle = load_model_bundle()

except Exception as exc:

    st.error("❌ Model could not be loaded.")

    st.code(str(exc))

    st.stop()


# =========================================================
# EXTRACT FINAL MODEL COMPONENTS
# =========================================================

xgb_model = model_bundle["model"]

scaler = model_bundle["scaler"]

encoder = model_bundle["encoder"]

feature_names = np.asarray(
    model_bundle["feature_names"]
).astype(str)

best_threshold = float(
    model_bundle["threshold"]
)


# =========================================================
# MODEL VALIDATION
# =========================================================

if not hasattr(xgb_model, "predict_proba"):

    st.error(
        "❌ Saved model does not support probability prediction."
    )

    st.stop()


# =========================================================
# SHAP
# =========================================================

@st.cache_resource
def load_shap_explainer(model):

    try:

        return shap.TreeExplainer(model)

    except Exception as tree_exc:

        try:

            return shap.Explainer(model)

        except Exception as generic_exc:

            raise RuntimeError(
                "TreeExplainer failed:\n"
                + str(tree_exc)
                + "\n\nGeneric SHAP Explainer also failed:\n"
                + str(generic_exc)
            ) from generic_exc


try:

    shap_explainer = load_shap_explainer(
        xgb_model
    )

    shap_available = True

    shap_init_error = None

except Exception as exc:

    shap_explainer = None

    shap_available = False

    shap_init_error = str(exc)


# =========================================================
# TRAINING FEATURE DEFINITIONS
# =========================================================

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AvgChargesPerMonth",
    "NumServices",
]

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "TenureGroup",
]


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
    TotalCharges,
):

    avg_charges_per_month = (
        TotalCharges / (tenure + 1)
    )

    if tenure <= 12:

        tenure_group = "0-1yr"

    elif tenure <= 24:

        tenure_group = "1-2yr"

    elif tenure <= 48:

        tenure_group = "2-4yr"

    else:

        tenure_group = "4-6yr"

    service_values = [
        PhoneService,
        MultipleLines,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection,
        TechSupport,
        StreamingTV,
        StreamingMovies,
    ]

    num_services = sum(
        1
        for value in service_values
        if value == "Yes"
    )

    return pd.DataFrame(
        [
            {
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
                "AvgChargesPerMonth": avg_charges_per_month,
                "TenureGroup": tenure_group,
                "NumServices": num_services,
            }
        ]
    )


# =========================================================
# PREPROCESS RAW CUSTOMER
# =========================================================

def transform_customer(customer):

    numeric_data = customer[
        numeric_features
    ]

    categorical_data = customer[
        categorical_features
    ]

    scaled_numeric = scaler.transform(
        numeric_data
    )

    encoded_categorical = encoder.transform(
        categorical_data
    )

    if hasattr(
        encoded_categorical,
        "toarray"
    ):

        encoded_categorical = (
            encoded_categorical.toarray()
        )

    transformed = np.hstack(
        [
            scaled_numeric,
            encoded_categorical,
        ]
    )

    transformed = np.asarray(
        transformed,
        dtype=float,
    )

    if transformed.ndim == 1:

        transformed = transformed.reshape(
            1, -1
        )

    return transformed


# =========================================================
# PREDICTION
# =========================================================

def predict_customer(customer):

    transformed = transform_customer(
        customer
    )

    probability = float(
        xgb_model.predict_proba(
            transformed
        )[0, 1]
    )

    prediction = int(
        probability >= best_threshold
    )

    return (
        prediction,
        probability,
        transformed,
    )


# =========================================================
# SHAP VALUE EXTRACTION
# =========================================================

def extract_shap_values(explanation):

    values = np.asarray(
        explanation.values
    )

    # (samples, features, classes)
    if values.ndim == 3:

        if values.shape[-1] >= 2:

            return values[0, :, 1]

        return values[0, :, 0]

    # (samples, features)
    if values.ndim == 2:

        return values[0]

    # (features,)
    if values.ndim == 1:

        return values

    raise ValueError(
        f"Unsupported SHAP output shape: "
        f"{values.shape}"
    )


# =========================================================
# CONTRIBUTION CHART
# =========================================================

def make_contribution_chart(
    values,
    feature_names,
    title,
    x_label,
    max_display=12,
):

    values = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    names = np.asarray(
        feature_names
    ).astype(str).reshape(-1)

    if len(values) != len(names):

        raise ValueError(
            f"Value count ({len(values)}) "
            f"does not match feature count "
            f"({len(names)})."
        )

    frame = pd.DataFrame(
        {
            "feature": names,
            "value": values,
        }
    )

    frame["abs_value"] = (
        frame["value"].abs()
    )

    frame = (
        frame
        .sort_values(
            "abs_value",
            ascending=False,
        )
        .head(max_display)
        .sort_values("value")
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    fig.patch.set_facecolor(
        "#111827"
    )

    ax.set_facecolor(
        "#111827"
    )

    colors = [
        "#60a5fa"
        if value < 0
        else "#f87171"
        for value in frame["value"]
    ]

    ax.barh(
        frame["feature"],
        frame["value"],
        color=colors,
        height=0.62,
    )

    ax.axvline(
        0,
        color="#94a3b8",
        linewidth=1.2,
    )

    ax.grid(
        axis="x",
        alpha=0.15,
        linewidth=0.8,
    )

    ax.set_axisbelow(True)

    ax.set_title(
        title,
        color="#f8fafc",
        fontsize=17,
        fontweight="bold",
        pad=16,
    )

    ax.set_xlabel(
        x_label,
        color="#cbd5e1",
        fontsize=10,
    )

    ax.tick_params(
        axis="x",
        colors="#cbd5e1",
        labelsize=9,
    )

    ax.tick_params(
        axis="y",
        colors="#f8fafc",
        labelsize=9,
    )

    for spine in ax.spines.values():

        spine.set_visible(False)

    max_abs = max(
        float(
            frame["abs_value"].max()
        ),
        1e-9,
    )

    for y, value in enumerate(
        frame["value"]
    ):

        offset = max_abs * 0.025

        if value >= 0:

            x = value + offset

            ha = "left"

        else:

            x = value - offset

            ha = "right"

        ax.text(
            x,
            y,
            f"{value:+.3f}",
            va="center",
            ha=ha,
            color="#e2e8f0",
            fontsize=8,
        )

    plt.tight_layout()

    return fig, frame


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

    st.write("XGBoost")

    st.write(
        "Binary Classification"
    )

    st.write(
        f"Decision Threshold: "
        f"{best_threshold:.3f}"
    )

    st.divider()

    st.subheader(
        "🔍 Explainability"
    )

    if shap_available:

        st.success(
            "SHAP Ready"
        )

    else:

        st.error(
            "SHAP unavailable"
        )

        st.caption(
            shap_init_error
        )

    st.divider()

    st.subheader(
        "📋 Workflow"
    )

    st.markdown(
        """
        **01** Enter customer information.

        **02** Enter billing information.

        **03** Click **Analyze Customer Churn**.

        **04** Review churn probability.

        **05** Review SHAP drivers.
        """
    )


# =========================================================
# HEADER
# =========================================================

st.title(
    "📊 Customer Churn Prediction"
)

st.write(
    "Predict churn probability and understand "
    "why the final XGBoost model made its prediction."
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


with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"],
    )

    SeniorCitizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No",
    )

    Partner = st.selectbox(
        "Partner",
        ["Yes", "No"],
    )

    Dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"],
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12,
        step=1,
    )


with col2:

    PhoneService = st.selectbox(
        "Phone Service",
        ["Yes", "No"],
    )

    MultipleLines = st.selectbox(
        "Multiple Lines",
        [
            "Yes",
            "No",
            "No phone service",
        ],
    )

    InternetService = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No",
        ],
    )

    OnlineSecurity = st.selectbox(
        "Online Security",
        [
            "Yes",
            "No",
            "No internet service",
        ],
    )

    OnlineBackup = st.selectbox(
        "Online Backup",
        [
            "Yes",
            "No",
            "No internet service",
        ],
    )

    DeviceProtection = st.selectbox(
        "Device Protection",
        [
            "Yes",
            "No",
            "No internet service",
        ],
    )


with col3:

    TechSupport = st.selectbox(
        "Tech Support",
        [
            "Yes",
            "No",
            "No internet service",
        ],
    )

    StreamingTV = st.selectbox(
        "Streaming TV",
        [
            "Yes",
            "No",
            "No internet service",
        ],
    )

    StreamingMovies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No",
            "No internet service",
        ],
    )

    Contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year",
        ],
    )

    PaperlessBilling = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"],
    )

    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
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
        step=1.0,
    )


with billing_col2:

    TotalCharges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0,
        step=10.0,
    )


# =========================================================
# PREDICT
# =========================================================

st.divider()

predict_button = st.button(
    "🔮 Analyze Customer Churn",
    type="primary",
    use_container_width=True,
)


if predict_button:

    try:

        # -------------------------------------------------
        # CREATE RAW CUSTOMER
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
            TotalCharges=TotalCharges,
        )


        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction, probability, transformed = (
            predict_customer(customer)
        )

        stay_probability = (
            1.0 - probability
        )


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.divider()

        st.header(
            "📈 Prediction Result"
        )

        r1, r2, r3 = st.columns(3)


        with r1:

            st.metric(
                "Prediction",
                (
                    "⚠️ CHURN"
                    if prediction == 1
                    else "✅ STAY"
                ),
            )


        with r2:

            st.metric(
                "Churn Probability",
                f"{probability:.2%}",
            )


        with r3:

            st.metric(
                "Stay Probability",
                f"{stay_probability:.2%}",
            )


        # -------------------------------------------------
        # RISK ASSESSMENT
        # -------------------------------------------------

        st.subheader(
            "Risk Assessment"
        )

        if probability >= best_threshold:

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
            min(
                max(probability, 0.0),
                1.0,
            )
        )

        st.caption(
            f"Prediction threshold: "
            f"{best_threshold:.3f}"
        )


        # -------------------------------------------------
        # MODEL EXPLANATION
        # -------------------------------------------------

        st.divider()

        st.header(
            "🔍 Prediction Explanation"
        )

        st.caption(
            "The explanation is generated from the "
            "final tuned XGBoost model used for prediction."
        )


        try:

            # -------------------------------------------------
            # FEATURE NAMES
            # -------------------------------------------------

            if len(feature_names) != transformed.shape[1]:

                raise ValueError(
                    "Feature count mismatch.\n"
                    f"Model input features: "
                    f"{transformed.shape[1]}\n"
                    f"Saved feature names: "
                    f"{len(feature_names)}"
                )


            # -------------------------------------------------
            # SHAP
            # -------------------------------------------------

            shap_success = False

            runtime_shap_error = None

            shap_values = None


            if shap_available:

                try:

                    explanation = (
                        shap_explainer(
                            transformed
                        )
                    )

                    shap_values = (
                        extract_shap_values(
                            explanation
                        )
                    )

                    shap_values = np.asarray(
                        shap_values,
                        dtype=float,
                    ).reshape(-1)


                    if len(shap_values) != len(
                        feature_names
                    ):

                        raise ValueError(
                            "SHAP values and feature "
                            "names have different lengths."
                        )


                    if not np.all(
                        np.isfinite(
                            shap_values
                        )
                    ):

                        raise ValueError(
                            "SHAP returned "
                            "non-finite values."
                        )


                    shap_success = True


                except Exception as exc:

                    runtime_shap_error = str(
                        exc
                    )


            # -------------------------------------------------
            # SHAP GRAPH
            # -------------------------------------------------

            if shap_success:

                fig, contribution_frame = (
                    make_contribution_chart(
                        shap_values,
                        feature_names,
                        "SHAP Feature Contributions",
                        (
                            "SHAP value | "
                            "positive → higher churn output"
                        ),
                        max_display=12,
                    )
                )


                st.success(
                    "✅ SHAP explanation generated "
                    "from the final tuned XGBoost model."
                )


                st.pyplot(
                    fig,
                    use_container_width=True,
                )

                plt.close(fig)


                st.caption(
                    "Positive SHAP values push the "
                    "model toward higher churn output; "
                    "negative values push it away from churn."
                )


                with st.expander(
                    "📋 View SHAP Feature Contributions"
                ):

                    display_frame = (
                        contribution_frame.copy()
                    )

                    display_frame[
                        "Direction"
                    ] = np.where(
                        display_frame["value"] > 0,
                        "Toward Churn",
                        "Away From Churn",
                    )


                    display_frame["value"] = (
                        display_frame["value"]
                        .round(4)
                    )


                    st.dataframe(
                        display_frame[
                            [
                                "feature",
                                "value",
                                "Direction",
                            ]
                        ].rename(
                            columns={
                                "feature": "Feature",
                                "value": "SHAP Value",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )


            # -------------------------------------------------
            # SHAP ERROR
            # -------------------------------------------------

            else:

                st.error(
                    "❌ SHAP explanation could not "
                    "be generated."
                )

                with st.expander(
                    "🔧 SHAP Technical Details"
                ):

                    st.code(
                        runtime_shap_error
                        or shap_init_error
                        or "Unknown SHAP error."
                    )


        except Exception as explanation_error:

            st.error(
                "❌ The prediction worked, but the "
                "explanation could not be generated."
            )

            with st.expander(
                "🔧 Explanation Technical Details"
            ):

                st.code(
                    str(explanation_error)
                )


        # -------------------------------------------------
        # CUSTOMER DATA
        # -------------------------------------------------

        with st.expander(
            "👤 View Customer Input & Engineered Features"
        ):

            st.dataframe(
                customer.T,
                use_container_width=True,
            )


        # -------------------------------------------------
        # MODEL INFORMATION
        # -------------------------------------------------

        with st.expander(
            "🤖 Model Information"
        ):

            c1, c2 = st.columns(2)


            with c1:

                st.write(
                    "**Algorithm:** XGBoost"
                )

                st.write(
                    "**Task:** Binary Classification"
                )

                st.write(
                    "**Preprocessing:** "
                    "StandardScaler + OneHotEncoder"
                )


            with c2:

                st.write(
                    "**Explainability:** SHAP"
                )

                st.write(
                    "**Output:** Churn Probability"
                )

                st.write(
                    f"**Decision Threshold:** "
                    f"{best_threshold:.3f}"
                )

                st.write(
                    "**Engineered Features:** "
                    "AvgChargesPerMonth, "
                    "TenureGroup, NumServices"
                )


    except Exception as exc:

        st.error(
            "❌ Prediction could not be generated."
        )

        with st.expander(
            "🔧 Technical Details"
        ):

            st.code(
                str(exc)
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "ChurnIQ • Customer Churn Prediction System • "
    "Final Tuned XGBoost • SHAP • Streamlit"
)
```
