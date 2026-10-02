from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_pipeline.pkl"

def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def engineer_customer_features(raw: dict) -> pd.DataFrame:
    tenure = int(raw["tenure"])
    total_charges = float(raw["TotalCharges"])

    if tenure <= 12:
        tenure_group = "0-1yr"
    elif tenure <= 24:
        tenure_group = "1-2yr"
    elif tenure <= 48:
        tenure_group = "2-4yr"
    else:
        tenure_group = "4-6yr"

    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    num_services = sum(raw[col] == "Yes" for col in service_cols)

    data = dict(raw)
    data["AvgChargesPerMonth"] = total_charges / (tenure + 1)
    data["TenureGroup"] = tenure_group
    data["NumServices"] = num_services

    return pd.DataFrame([data])

def predict_customer(raw: dict) -> dict:
    pipeline = load_pipeline()
    raw_df = engineer_customer_features(raw)
    prediction = int(pipeline.predict(raw_df)[0])
    probability = float(pipeline.predict_proba(raw_df)[0, 1])

    return {
        "prediction": prediction,
        "label": "Churn" if prediction == 1 else "No Churn",
        "probability": probability,
        "raw_df": raw_df,
    }
