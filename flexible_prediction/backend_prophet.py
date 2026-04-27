import pandas as pd
import numpy as np
from prophet import Prophet

from prophet_feature_engineering import create_prophet_features


# =====================================================
# 1. TRAIN MODEL (RUN ON STARTUP)
# =====================================================
def train_model():

    df = pd.read_csv("../Datasets/new_transaction_data.csv")
    df["date"] = pd.to_datetime(df["date"])

    # -----------------------------
    # BASE DATASET
    # -----------------------------
    df_prophet = pd.DataFrame()
    df_prophet["ds"] = df["date"]
    df_prophet["y"] = df["total_txn_current"]

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------
    features = create_prophet_features(df)

    df_prophet = pd.concat(
        [df_prophet.reset_index(drop=True),
         features.reset_index(drop=True)],
        axis=1
    )

    # Fill NaNs (critical for Prophet stability)
    df_prophet = df_prophet.fillna(0)

    # -----------------------------
    # PROPHET MODEL
    # -----------------------------
    model = Prophet(

        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,

        changepoint_prior_scale=0.15,
        changepoint_range=0.85,
        n_changepoints=40,

        growth="linear",

        seasonality_mode="additive",
        seasonality_prior_scale=5.0,

        holidays_prior_scale=12.0,

        interval_width=0.9
    )

    # -----------------------------
    # REGRESSORS (ONLY FUTURE-SAFE)
    # -----------------------------
    regressors = [
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "is_salary_period",
        "is_mid_month",
        "is_end_month_spike",
        "is_payday",
        "campaign_window",

        # SHOCK PROXIES (SAFE FOR FUTURE)
        "high_risk_period",
        "stress_window"
    ]

    regressors = [r for r in regressors if r in df_prophet.columns]

    print("Regressors used:", regressors)

    for r in regressors:
        model.add_regressor(r)

    model.fit(df_prophet)

    return model, regressors


# GLOBAL MODEL
MODEL, REGRESSORS = train_model()


# =====================================================
# 2. FUTURE FEATURE ENGINEERING (SAFE)
# =====================================================
def build_future_features(future_df: pd.DataFrame):

    df = future_df.copy()

    # rename for feature function compatibility
    df = df.rename(columns={"ds": "date"})

    features = create_prophet_features(df)

    future_df = future_df.reset_index(drop=True)
    features = features.reset_index(drop=True)

    combined = pd.concat([future_df, features], axis=1)

    # ensure all regressors exist
    for col in REGRESSORS:
        if col not in combined.columns:
            combined[col] = 0

    return combined


# =====================================================
# 3. PREDICTION FUNCTION (USED BY DASH)
# =====================================================
def predict_transaction(start_date, end_date):

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    future_dates = pd.date_range(start_date, end_date)

    future = pd.DataFrame({"ds": future_dates})

    # build regressors (future-safe only)
    future = build_future_features(future)

    # prediction
    forecast = MODEL.predict(future)

    result = forecast[["ds", "yhat"]].copy()
    result = result.rename(columns={"yhat": "y"})

    return result