import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ChurnIQ | Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PROFESSIONAL DARK DASHBOARD UI
# Font colors are intentionally not changed for the metric
# cards; only backgrounds, borders and containers are styled.
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top right, #182338 0%, transparent 32%),
            linear-gradient(135deg, #0b1220 0%, #101827 55%, #0b1220 100%);
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
    }

    section[data-testid="stSidebar"] {
        background: #0d1625;
        border-right: 1px solid #243247;
    }

    section[data-testid="stSidebar"] * {
        color: #dbe4f0;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
        letter-spacing: -0.02em;
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
        border-radius: 9px !important;
    }

    div[data-baseweb="select"] span {
        color: #f8fafc !important;
    }

    div[data-testid="stNumberInput"] input {
        background: #151e2e !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 9px !important;
    }

    div[data-testid="stTextInput"] input {
        background: #151e2e !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 9px !important;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        min-height: 46px;
        font-weight: 700 !important;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #1d4ed8 !important;
    }

    /* Fix the white metric cards from the screenshot.
       Existing font colors are not overridden here. */
    div[data-testid="stMetric"] {
        background: #151e2e !important;
        border: 1px solid #2b3a4f !important;
        border-radius: 14px !important;
        padding: 18px !important;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.18) !important;
    }

    div[data-testid="stMetricValue"] {
        background: transparent !important;
    }

    div[data-testid="stPlotlyChart"] {
        background: #111a29 !important;
        border: 1px solid #2b3a4f !important;
        border-radius: 14px !important;
        padding: 8px !important;
    }

    div[data-testid="stExpander"] {
        background: #111a29 !important;
        border: 1px solid #2b3a4f !important;
        border-radius: 14px !important;
    }

    div[data-testid="stFileUploader"] {
        background: #151e2e !important;
        border: 1px dashed #46566c !important;
        border-radius: 14px !important;
        padding: 12px !important;
    }

    .dashboard-card {
        background: #111a29;
        border: 1px solid #2b3a4f;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }

    .hero-card {
        background:
            linear-gradient(135deg, rgba(37,99,235,.18), rgba(14,165,233,.05)),
            #111a29;
        border: 1px solid #2b3a4f;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 18px;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 0.88rem;
    }

    hr {
        border-color: #263449 !important;
    }

    [data-testid="stProgress"] > div > div {
        background: #2563eb !important;
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
# MODEL / DEPLOYMENT BUNDLE
#
# The notebook's final deployment bundle contains:
# model, scaler, encoder, feature_names, threshold
#
# The final model is the Balanced Tuned XGBoost model.
# =========================================================

@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle not found:\n{MODEL_PATH}\n\n"
            "Place the notebook-generated models/churn_pipeline.pkl "
            "inside the models folder beside app.py."
        )

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict):
        raise TypeError(
            "The saved churn_pipeline.pkl is not the Phase-20 "
            "deployment bundle. Expected a dictionary containing "
            "model, scaler, encoder, feature_names and threshold."
        )

    required = {
        "model",
        "scaler",
        "encoder",
        "feature_names",
        "threshold",
    }

    missing = required.difference(bundle.keys())

    if missing:
        raise KeyError(
            "Deployment bundle is missing: "
            + ", ".join(sorted(missing))
        )

    return bundle


try:
    bundle = load_bundle()

    model = bundle["model"]
    scaler = bundle["scaler"]
    encoder = bundle["encoder"]
    feature_names = list(bundle["feature_names"])
    threshold = float(bundle["threshold"])

except Exception as exc:
    st.error("❌ Final model bundle could not be loaded.")
    st.code(str(exc))
    st.stop()


# =========================================================
# NOTEBOOK-EXACT FEATURE DEFINITIONS
#
# Feature engineering was already performed in the notebook.
# The app only creates these engineered columns when a raw
# customer does not already contain them.
# =========================================================

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AvgChargesPerMonth",
    "NumServices",
]

CATEGORICAL_FEATURES = [
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

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

RAW_REQUIRED_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
]


# =========================================================
# TRANSFORMATION
#
# This follows the notebook's transform_customer() logic.
# It does NOT ask the user to manually enter TenureGroup.
# =========================================================

def transform_customer(
    customer_df,
    customer_scaler=None,
    customer_encoder=None,
):
    customer_df = customer_df.copy()

    if customer_scaler is None:
        customer_scaler = scaler

    if customer_encoder is None:
        customer_encoder = encoder

    # Remove target / identifier columns if they are present.
    customer_df = customer_df.drop(
        columns=["Churn", "customerID"],
        errors="ignore",
    )

    missing_raw = [
        col for col in RAW_REQUIRED_COLUMNS
        if col not in customer_df.columns
    ]

    if missing_raw:
        raise ValueError(
            "Missing required customer columns:\n"
            + ", ".join(missing_raw)
        )

    # Create only missing engineered features.
    if "AvgChargesPerMonth" not in customer_df.columns:
        customer_df["AvgChargesPerMonth"] = (
            customer_df["TotalCharges"]
            / (customer_df["tenure"] + 1)
        )

    if "TenureGroup" not in customer_df.columns:
        customer_df["TenureGroup"] = pd.cut(
            customer_df["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
        ).astype(str)

    if "NumServices" not in customer_df.columns:
        customer_df["NumServices"] = (
            customer_df[SERVICE_COLUMNS]
            .eq("Yes")
            .sum(axis=1)
        )

    # Match the notebook's preprocessing order.
    customer_num_scaled = customer_scaler.transform(
        customer_df[NUMERIC_FEATURES]
    )

    customer_cat_encoded = customer_encoder.transform(
        customer_df[CATEGORICAL_FEATURES]
    )

    if hasattr(customer_cat_encoded, "toarray"):
        customer_cat_encoded = customer_cat_encoded.toarray()

    transformed_data = np.hstack(
        [
            np.asarray(customer_num_scaled),
            np.asarray(customer_cat_encoded),
        ]
    )

    transformed_data = np.asarray(
        transformed_data,
        dtype=float,
    )

    if transformed_data.shape[1] != len(feature_names):
        raise ValueError(
            "Transformed feature count does not match the saved model.\n"
            f"Transformed: {transformed_data.shape[1]}\n"
            f"Expected: {len(feature_names)}"
        )

    return transformed_data


# =========================================================
# PREDICTION
# =========================================================

def predict_customer(customer_df):
    transformed = transform_customer(customer_df)

    probability = float(
        model.predict_proba(transformed)[0, 1]
    )

    prediction = int(
        probability >= threshold
    )

    return {
        "prediction": prediction,
        "label": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": probability,
        "threshold": threshold,
        "transformed": transformed,
    }


# =========================================================
# SHAP
#
# Always explains the SAME FINAL XGBoost model loaded from
# churn_pipeline.pkl. Random Forest is never used here.
# =========================================================

@st.cache_resource
def load_shap_explainer():
    if not SHAP_AVAILABLE:
        raise ImportError(
            "SHAP is not installed. Install it with: pip install shap"
        )

    return shap.TreeExplainer(model)


def get_customer_shap_explanation(transformed):
    explainer = load_shap_explainer()

    raw_values = np.asarray(
        explainer.shap_values(transformed)
    )

    # XGBoost binary classification normally returns
    # (samples, features). Handle class dimension as well
    # for SHAP versions that return it.
    if raw_values.ndim == 3:
        values = raw_values[0, :, -1]
    elif raw_values.ndim == 2:
        values = raw_values[0]
    else:
        values = raw_values.reshape(-1)

    base_value = explainer.expected_value

    if np.ndim(base_value) > 0:
        base_value = np.asarray(
            base_value
        ).reshape(-1)[-1]

    base_value = float(base_value)

    if len(values) != len(feature_names):
        raise ValueError(
            "SHAP feature count mismatch.\n"
            f"SHAP: {len(values)}\n"
            f"Features: {len(feature_names)}"
        )

    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=transformed[0],
        feature_names=feature_names,
    )

    return explanation


# =========================================================
# GRAPH HELPERS
# =========================================================

def render_probability_threshold_chart(probability):
    if not PLOTLY_AVAILABLE:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(
            ["Churn probability"],
            [probability],
        )
        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=2,
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Churn Probability vs Decision Threshold")
        ax.grid(axis="y", alpha=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=["Current Customer"],
            y=[probability],
            text=[f"{probability:.1%}"],
            textposition="auto",
            marker_color="#38bdf8",
            name="Churn Probability",
        )
    )

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#f97316",
        line_width=2,
        annotation_text=f"Decision threshold = {threshold:.2f}",
        annotation_position="top right",
    )

    fig.update_layout(
        height=350,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(
            range=[0, 1],
            tickformat=".0%",
            title="Churn probability",
        ),
        xaxis_title="",
        showlegend=False,
        margin=dict(l=30, r=30, t=45, b=30),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def render_shap_waterfall(transformed):
    """Render a readable customer-level SHAP waterfall for the final model."""
    if not SHAP_AVAILABLE:
        st.warning(
            "SHAP is not installed. Install `shap` to enable "
            "the customer-level waterfall explanation."
        )
        return

    explanation = get_customer_shap_explanation(transformed)
    values = np.asarray(explanation.values, dtype=float).reshape(-1)
    names = np.asarray(feature_names).astype(str).reshape(-1)

    if len(values) != len(names):
        raise ValueError("SHAP values and feature names have different lengths.")

    # Keep the largest absolute contributors individually visible.
    max_display = 12
    order = np.argsort(np.abs(values))[::-1]
    keep = order[:max_display]
    other = float(values[order[max_display:]].sum()) if len(values) > max_display else 0.0

    display_names = names[keep].tolist()
    display_values = values[keep].tolist()

    if len(values) > max_display:
        display_names.append(f"Other {len(values) - max_display} features")
        display_values.append(other)

    # Reverse for top-to-bottom readability.
    display_names = display_names[::-1]
    display_values = display_values[::-1]

    base = float(explanation.base_values)
    cumulative = [base]
    for value in display_values:
        cumulative.append(cumulative[-1] + value)

    # Plotly waterfall is much easier to read in the dark dashboard than
    # SHAP's default matplotlib waterfall, especially for feature labels.
    if PLOTLY_AVAILABLE:
        measure = ["absolute"] + ["relative"] * len(display_values) + ["total"]
        x_values = [base] + display_values + [0.0]
        labels = ["Base value"] + display_names + ["Model output"]

        fig = go.Figure(
            go.Waterfall(
                name="SHAP",
                orientation="v",
                measure=measure,
                x=labels,
                y=x_values,
                text=[f"{base:+.2f}"]
                + [f"{v:+.2f}" for v in display_values]
                + [f"{cumulative[-1]:+.2f}"],
                textposition="outside",
                connector={"line": {"color": "#64748b", "width": 1}},
                increasing={"marker": {"color": "#fb7185"}},
                decreasing={"marker": {"color": "#38bdf8"}},
                totals={"marker": {"color": "#a78bfa"}},
                hovertemplate="%{x}<br>Contribution: %{y:+.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            height=570,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=55, r=35, t=30, b=150),
            showlegend=False,
            xaxis=dict(
                title="",
                tickangle=-35,
                tickfont=dict(size=11, color="#e2e8f0"),
                automargin=True,
            ),
            yaxis=dict(
                title="SHAP contribution / model output",
                tickfont=dict(size=10, color="#cbd5e1"),
                gridcolor="rgba(148,163,184,0.16)",
                zerolinecolor="#64748b",
            ),
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Red = pushes toward higher churn output • Blue = pushes toward lower churn output. "
            "The largest contributors are shown individually; remaining features are grouped."
        )
        return

    # Matplotlib fallback with explicit dark text/background styling.
    plt.figure(figsize=(11, 7))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.patch.set_facecolor("#111a29")
    for ax in fig.axes:
        ax.set_facecolor("#111a29")
        ax.tick_params(colors="#e2e8f0", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        for text in ax.texts:
            text.set_color("#e2e8f0")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "SHAP values explain the final Balanced Tuned XGBoost model output."
    )


def render_model_feature_importance():
    importances = getattr(
        model,
        "feature_importances_",
        None,
    )

    if importances is None:
        st.info(
            "The loaded final model does not expose "
            "feature_importances_."
        )
        return

    importances = np.asarray(
        importances,
        dtype=float,
    )

    if len(importances) != len(feature_names):
        st.warning(
            "Model feature importance count does not match "
            "the saved feature names."
        )
        return

    frame = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances,
        }
    )

    frame = (
        frame
        .sort_values("Importance", ascending=True)
        .tail(15)
    )

    if PLOTLY_AVAILABLE:
        fig = px.bar(
            frame,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 Model Feature Importances",
        )

        fig.update_traces(
            marker_color="#38bdf8"
        )

        fig.update_layout(
            height=480,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=55, b=25),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    else:
        frame = frame.set_index("Feature")
        st.bar_chart(
            frame,
            y="Importance",
            height=430,
        )


def render_batch_churn_donut(output_df):
    counts = (
        output_df["predicted_churn"]
        .value_counts()
        .reindex(
            ["No Churn", "Churn"],
            fill_value=0,
        )
    )

    if PLOTLY_AVAILABLE:
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=counts.index,
                    values=counts.values,
                    hole=0.64,
                    textinfo="label+percent",
                    marker=dict(
                        colors=[
                            "#38bdf8",
                            "#fb7185",
                        ]
                    ),
                )
            ]
        )

        fig.update_layout(
            height=390,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=25, b=20),
            legend_title_text="",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )
    else:
        st.bar_chart(
            counts,
            height=350,
        )


def render_probability_distribution(output_df):
    if PLOTLY_AVAILABLE:
        fig = px.histogram(
            output_df,
            x="churn_probability",
            nbins=20,
            title="Distribution of Predicted Churn Probability",
        )

        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="#f97316",
            annotation_text=f"Threshold = {threshold:.2f}",
            annotation_position="top right",
        )

        fig.update_layout(
            height=390,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                tickformat=".0%",
                title="Churn probability",
            ),
            yaxis_title="Customers",
            margin=dict(l=25, r=25, t=55, b=30),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.hist(
            output_df["churn_probability"],
            bins=20,
        )
        ax.axvline(
            threshold,
            linestyle="--",
            linewidth=2,
        )
        ax.set_xlabel("Churn probability")
        ax.set_ylabel("Customers")
        ax.set_title("Churn Probability Distribution")
        ax.grid(axis="y", alpha=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def render_confusion_matrix(output_df):
    if "actual_churn" not in output_df.columns:
        st.info(
            "Confusion matrix is available when the uploaded "
            "CSV contains the actual `Churn` column."
        )
        return

    actual = output_df["actual_churn"]
    valid = actual.notna()

    if valid.sum() == 0:
        st.info(
            "No supported actual churn labels were found."
        )
        return

    actual_values = actual[valid].astype(int).to_numpy()
    predicted_values = (
        output_df.loc[valid, "churn_probability"]
        >= threshold
    ).astype(int).to_numpy()

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(
        actual_values,
        predicted_values,
        labels=[0, 1],
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual: No Churn",
            "Actual: Churn",
        ],
        columns=[
            "Predicted: No Churn",
            "Predicted: Churn",
        ],
    )

    if PLOTLY_AVAILABLE:
        fig = px.imshow(
            cm_df,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            labels={
                "x": "Predicted",
                "y": "Actual",
                "color": "Customers",
            },
        )

        fig.update_layout(
            height=390,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=30, r=30, t=30, b=30),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )
    else:
        st.dataframe(
            cm_df,
            use_container_width=True,
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

    st.subheader("🤖 Final Model")
    st.write("Balanced Tuned XGBoost")
    st.write("Binary Classification")

    st.divider()

    st.subheader("🎯 Decision Threshold")
    st.metric(
        "Selected Threshold",
        f"{threshold:.2f}",
    )

    st.caption(
        "Threshold selected in the notebook using "
        "5-fold out-of-fold F1 analysis."
    )

    st.divider()

    st.subheader("🔍 Explainability")

    if SHAP_AVAILABLE:
        st.success(
            "SHAP Ready"
        )
    else:
        st.warning(
            "SHAP package unavailable"
        )

    st.caption(
        "SHAP always explains the final XGBoost model "
        "loaded from the deployment bundle."
    )

    st.divider()

    st.subheader("📋 Workflow")
    st.markdown(
        """
        **01** Enter customer information  
        **02** Analyze churn probability  
        **03** Review model decision  
        **04** Review SHAP waterfall  
        **05** Review feature importance  
        **06** Upload a CSV for batch analytics
        """
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin-bottom:6px;">📊 Customer Churn Prediction</h1>
        <div class="small-muted">
            Final Balanced Tuned XGBoost • Notebook-aligned preprocessing •
            SHAP explainability • Decision threshold 0.45
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CUSTOMER INPUT
# =========================================================

st.header("👤 Customer Profile")
st.caption(
    "Enter demographic, subscription and service information. "
    "TenureGroup and other engineered features are generated "
    "automatically exactly as defined in the notebook."
)

with st.form("customer_form"):

    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )

        SeniorCitizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: (
                "Yes" if x == 1 else "No"
            ),
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

        PhoneService = st.selectbox(
            "Phone Service",
            ["Yes", "No"],
        )

    with c2:
        MultipleLines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"],
        )

        InternetService = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
        )

        OnlineSecurity = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"],
        )

        OnlineBackup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"],
        )

        DeviceProtection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"],
        )

        TechSupport = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"],
        )

    with c3:
        StreamingTV = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"],
        )

        StreamingMovies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"],
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

    st.divider()

    st.subheader("💳 Billing Information")

    b1, b2 = st.columns(2)

    with b1:
        MonthlyCharges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0,
        )

    with b2:
        TotalCharges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0,
            step=10.0,
        )

    analyze = st.form_submit_button(
        "🔮 Predict Churn Risk",
        use_container_width=True,
    )


# =========================================================
# SINGLE CUSTOMER RESULT
# =========================================================

if analyze:

    new_customer = pd.DataFrame(
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
            }
        ]
    )

    try:
        result = predict_customer(
            new_customer
        )

        probability = result[
            "churn_probability"
        ]

        prediction = result[
            "prediction"
        ]

        transformed_customer = result[
            "transformed"
        ]

        st.divider()
        st.header("📈 Prediction Result")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                "Churn Probability",
                f"{probability:.1%}",
            )

        with r2:
            st.metric(
                "Model Decision",
                (
                    "Churn Flag"
                    if prediction == 1
                    else "No Churn Flag"
                ),
            )

        with r3:
            if probability >= 0.70:
                risk_band = "Higher"
            elif probability >= threshold:
                risk_band = "Elevated"
            else:
                risk_band = "Lower"

            st.metric(
                "Risk Band",
                risk_band,
            )

        st.subheader(
            "Estimated churn probability"
        )

        st.progress(
            min(
                max(probability, 0.0),
                1.0,
            )
        )

        st.caption(
            f"Decision threshold: {threshold:.2f}. "
            "Risk band is a presentation label; "
            "the probability is the model output."
        )

        if probability >= threshold:
            st.warning(
                "⚠️ This profile is at or above the model's "
                "churn decision threshold."
            )
        else:
            st.success(
                "✅ This profile is below the model's "
                "churn decision threshold."
            )

        # -------------------------------------------------
        # GRAPH 1
        # -------------------------------------------------

        st.divider()
        st.subheader(
            "1️⃣ Probability vs Decision Threshold"
        )

        render_probability_threshold_chart(
            probability
        )

        # -------------------------------------------------
        # GRAPH 2 — SHAP WATERFALL
        # -------------------------------------------------

        st.divider()
        st.subheader(
            "2️⃣ Customer-Level SHAP Waterfall"
        )

        render_shap_waterfall(
            transformed_customer
        )

        # -------------------------------------------------
        # GRAPH 3 — MODEL FEATURE IMPORTANCE
        # -------------------------------------------------

        st.divider()
        st.subheader(
            "3️⃣ Global Model Feature Importance"
        )

        st.caption(
            "This chart uses feature_importances_ from the "
            "same final Balanced Tuned XGBoost model."
        )

        render_model_feature_importance()

        # -------------------------------------------------
        # CUSTOMER DATA
        # -------------------------------------------------

        with st.expander(
            "👤 View Customer Data & Engineered Features"
        ):
            engineered_customer = new_customer.copy()

            engineered_customer[
                "AvgChargesPerMonth"
            ] = (
                engineered_customer["TotalCharges"]
                / (engineered_customer["tenure"] + 1)
            )

            engineered_customer[
                "TenureGroup"
            ] = pd.cut(
                engineered_customer["tenure"],
                bins=[-1, 12, 24, 48, 72],
                labels=[
                    "0-1yr",
                    "1-2yr",
                    "2-4yr",
                    "4-6yr",
                ],
            ).astype(str)

            engineered_customer[
                "NumServices"
            ] = (
                engineered_customer[
                    SERVICE_COLUMNS
                ]
                .eq("Yes")
                .sum(axis=1)
            )

            st.dataframe(
                engineered_customer.T,
                use_container_width=True,
                hide_index=True,
            )

        with st.expander(
            "🤖 Final Model Information"
        ):
            info1, info2 = st.columns(2)

            with info1:
                st.write(
                    "**Algorithm:** Balanced Tuned XGBoost"
                )
                st.write(
                    "**Task:** Binary Classification"
                )
                st.write(
                    "**Preprocessing:** StandardScaler + OneHotEncoder"
                )

            with info2:
                st.write(
                    "**Explainability:** SHAP"
                )
                st.write(
                    f"**Threshold:** {threshold:.2f}"
                )
                st.write(
                    f"**Transformed features:** {len(feature_names)}"
                )

    except Exception as exc:

        st.error(
            "❌ Prediction could not be generated."
        )

        with st.expander(
            "🔧 Technical Details"
        ):
            st.code(str(exc))


# =========================================================
# BATCH ANALYTICS
# =========================================================

st.divider()
st.header("📂 Batch Customer Analytics")

st.caption(
    "Upload a CSV containing the same raw customer fields used "
    "by the notebook. An optional `Churn` column enables the "
    "confusion matrix."
)

uploaded_file = st.file_uploader(
    "Upload customer CSV",
    type=["csv"],
)

if uploaded_file is not None:

    try:
        batch_df = pd.read_csv(
            uploaded_file
        )

        st.write(
            f"Loaded **{len(batch_df):,} customers**."
        )

        batch_transformed = transform_customer(
            batch_df
        )

        batch_probability = (
            model.predict_proba(
                batch_transformed
            )[:, 1]
        )

        batch_prediction = (
            batch_probability >= threshold
        ).astype(int)

        output_df = batch_df.copy()

        output_df[
            "churn_probability"
        ] = batch_probability

        output_df[
            "predicted_churn"
        ] = np.where(
            batch_prediction == 1,
            "Churn",
            "No Churn",
        )

        if "Churn" in batch_df.columns:

            actual_raw = (
                batch_df["Churn"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            actual_map = {
                "yes": 1,
                "no": 0,
                "churn": 1,
                "no churn": 0,
                "1": 1,
                "0": 0,
                "true": 1,
                "false": 0,
            }

            output_df[
                "actual_churn"
            ] = actual_raw.map(
                actual_map
            )

        # -------------------------------------------------
        # BATCH METRICS
        # -------------------------------------------------

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric(
                "Customers",
                f"{len(output_df):,}",
            )

        with m2:
            st.metric(
                "Predicted Churn",
                f"{int(batch_prediction.sum()):,}",
            )

        with m3:
            st.metric(
                "Predicted No Churn",
                f"{int((batch_prediction == 0).sum()):,}",
            )

        with m4:
            st.metric(
                "Average Churn Probability",
                f"{batch_probability.mean():.1%}",
            )

        # -------------------------------------------------
        # GRAPHS 4 + 5 — BATCH DASHBOARD
        # -------------------------------------------------

        st.divider()
        st.subheader("📊 Batch Risk Dashboard")
        st.caption(
            "Two complementary views of the uploaded customer population: "
            "predicted churn mix and the distribution of churn probabilities."
        )

        g1, g2 = st.columns(2, gap="large")

        with g1:
            st.markdown("### 4️⃣ Churn vs Non-churn Donut Chart")
            render_batch_churn_donut(output_df)

        with g2:
            st.markdown("### 5️⃣ Churn Probability Histogram")
            render_probability_distribution(output_df)

        # -------------------------------------------------
        # CONFUSION MATRIX
        # -------------------------------------------------

        if "actual_churn" in output_df.columns:

            st.divider()
            st.subheader(
                "6️⃣ Actual vs Predicted — Confusion Matrix"
            )

            render_confusion_matrix(
                output_df
            )

        # -------------------------------------------------
        # BATCH OUTPUT
        # -------------------------------------------------

        with st.expander(
            "📋 View Prediction Table"
        ):
            st.dataframe(
                output_df,
                use_container_width=True,
                hide_index=True,
            )

        csv_output = output_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Prediction Results",
            data=csv_output,
            file_name="churn_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except Exception as exc:

        st.error(
            "❌ Batch prediction could not be generated."
        )

        with st.expander(
            "🔧 Batch Technical Details"
        ):
            st.code(str(exc))


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "ChurnIQ • Customer Churn Prediction System • "
    "Balanced Tuned XGBoost • SHAP • Streamlit"
)
