import pandas as pd
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

    # -----------------------------
    # PROPhet MODEL
    # -----------------------------
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1
    )

    # -----------------------------
    # REGRESSORS (STRICT LIST)
    # -----------------------------
    regressors = [
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "is_salary_period",
        "salary_peak",
        "salary_mid",
        "salary_late",
        "is_mid_month",
        "is_end_month_spike",
        "is_payday",
        "is_payday_15",
        "is_payday_30",
        "weekend_payday",
        "campaign_window",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "dom_sin",
        "dom_cos"
    ]

    # enforce only existing columns
    regressors = [r for r in regressors if r in df_prophet.columns]

    for r in regressors:
        model.add_regressor(r)

    # remove any NaNs (critical for Prophet stability)
    df_prophet = df_prophet.fillna(0)

    model.fit(df_prophet)

    return model, regressors


# GLOBAL MODEL
MODEL, REGRESSORS = train_model()


# =====================================================
# 2. FUTURE FEATURE ENGINEERING (FIXED)
# =====================================================
def build_future_features(future_df: pd.DataFrame):

    df = future_df.copy()

    # IMPORTANT: create required date column format
    df = df.rename(columns={"ds": "date"})

    features = create_prophet_features(df)

    future_df = future_df.reset_index(drop=True)
    features = features.reset_index(drop=True)

    combined = pd.concat([future_df, features], axis=1)

    # enforce missing regressor safety
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

    # build regressors
    future = build_future_features(future)

    # prediction
    forecast = MODEL.predict(future)

    result = forecast[["ds", "yhat"]].copy()
    result = result.rename(columns={"yhat": "y"})

    return result