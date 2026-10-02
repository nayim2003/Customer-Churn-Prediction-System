from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_pipeline.pkl"


def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def get_churn_explanation(raw_df: pd.DataFrame):
    pipeline = load_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    transformed = preprocessor.transform(raw_df)
    feature_names = list(preprocessor.get_feature_names_out())

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(transformed)

    if isinstance(values, list):
        shap_values = np.asarray(values[1][0])
        base_value = float(explainer.expected_value[1])
    else:
        arr = np.asarray(values)
        if arr.ndim == 3:
            shap_values = arr[0, :, 1]
            base_value = float(np.asarray(explainer.expected_value)[1])
        elif arr.ndim == 2:
            shap_values = arr[0]
            base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        else:
            raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")

    table = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Contribution": shap_values,
    })
    table["Absolute Contribution"] = table["SHAP Contribution"].abs()
    table["Direction"] = np.where(
        table["SHAP Contribution"] > 0,
        "Toward churn",
        "Away from churn",
    )
    table = table.sort_values(
        "Absolute Contribution", ascending=False
    ).reset_index(drop=True)

    return {
        "shap_values": shap_values,
        "base_value": base_value,
        "feature_names": feature_names,
        "transformed": transformed,
        "table": table,
    }
