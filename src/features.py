"""
Feature engineering.

This dataset is a single snapshot (one row per customer, no dates), so
there is no time-series leakage risk the way there would be for a
forecasting problem. The leakage risk here is different and more subtle:
any transformation that *learns a parameter from data* (one-hot encoding
categories, scaling) must be fit on TRAIN only and applied unchanged to
TEST. That is handled in `modeling.py` via sklearn `Pipeline`s, not here --
everything in this file is a deterministic, row-wise function of a single
customer's own raw fields, so it is safe to apply before the split.
"""
import numpy as np
import pandas as pd

from src import config


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # How many of the 6 optional add-on services this customer has --
    # a direct proxy for how "embedded"/sticky the customer is. A
    # customer with zero add-ons has little switching cost; one with all
    # six has built real habit and setup cost around the service.
    df["num_services"] = (df[config.SERVICE_COLUMNS] == "Yes").sum(axis=1)

    # Binned tenure. Churn risk is famously front-loaded (highest in the
    # first year), so the RAW tenure number is less informative to a
    # linear model than "which lifecycle stage is this customer in" --
    # the risk-vs-tenure relationship is strongly nonlinear.
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=config.TENURE_BUCKET_EDGES, labels=config.TENURE_BUCKET_LABELS
    )

    # Contract type is the single strongest known churn driver in this
    # dataset (month-to-month customers can leave with zero penalty);
    # pull it out as an explicit boolean so its effect is easy to read
    # off a linear model's coefficient directly.
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    df["has_internet"] = (df["InternetService"] != "No").astype(int)

    # Average historical monthly spend implied by total billing, vs. the
    # CURRENT monthly rate. A customer whose current rate is well above
    # their historical average (e.g. a promo just expired) is a
    # bill-shock churn risk that MonthlyCharges alone doesn't capture.
    implied_avg_monthly = df["TotalCharges"] / df["tenure"].replace(0, 1)
    ratio = df["MonthlyCharges"] / implied_avg_monthly.replace(0, np.nan)
    df["charge_increase_ratio"] = ratio.fillna(1.0)

    return df


BINARY_YESNO_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]

CATEGORICAL_COLS = [
    "gender", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod", "tenure_bucket",
]

NUMERIC_COLS = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "num_services", "is_month_to_month", "has_internet", "charge_increase_ratio",
]


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the simple Yes/No columns to 0/1; leave multi-category
    columns as strings for the modeling Pipeline's OneHotEncoder to
    handle (fit-on-train-only, see modeling.py)."""
    df = df.copy()
    for col in BINARY_YESNO_COLS:
        df[col] = (df[col] == "Yes").astype(int)
    return df


FEATURE_COLUMNS = BINARY_YESNO_COLS + CATEGORICAL_COLS + NUMERIC_COLS


if __name__ == "__main__":
    from src.data_pipeline import load_raw, clean

    df = clean(load_raw())
    df = engineer_features(df)
    df = prepare_model_frame(df)
    print(df[FEATURE_COLUMNS].head())
    print(df[FEATURE_COLUMNS].isna().sum().loc[lambda s: s > 0])
