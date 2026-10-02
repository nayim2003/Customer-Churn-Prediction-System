"""Feature engineering functions aligned with the project notebook."""
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["AvgChargesPerMonth"] = data["TotalCharges"] / (data["tenure"] + 1)
    data["TenureGroup"] = pd.cut(
        data["tenure"], bins=[-1, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
    )
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    data["NumServices"] = data[service_cols].apply(lambda x: (x == "Yes").sum(), axis=1)
    return data
