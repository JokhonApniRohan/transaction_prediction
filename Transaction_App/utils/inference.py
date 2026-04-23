import numpy as np
import pandas as pd
import joblib

from feature_engineering import create_bd_date_features
from pre_processing import _preprocess, get_feature_columns


# =====================================================
# LOAD MODEL
# =====================================================
def load_model(path="Model/best_model.pkl"):
    data = joblib.load(path)
    return data["model"], data["features"], data["scaler"]


# =====================================================
# INFERENCE - EXACT REPLICA OF TRAINING PIPELINE
# =====================================================
def predict_next_day(df, model, scaler, features):
    """
    Predict next day transaction using the same pipeline as training.
    
    Pipeline matches: driver_code.py -> fit_transform_preprocess() -> transform_preprocess()
    
    NOTE: Tree-based models (CatBoost, XGBoost, LightGBM) are scale-invariant, 
    so we don't need to scale for inference. The scaler is saved but not needed.
    """

    # =============================
    # COPY AND SORT BY DATE
    # =============================
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Store the last row info BEFORE any processing
    current_value = df.iloc[-2]["total_txn_current"]
    current_date = df.iloc[-2]["date"]

    # =============================
    # FEATURE ENGINEERING (SAME AS TRAINING)
    # =============================
    df = create_bd_date_features(
        df,
        date_col="date",
        base_series=df["total_txn_current"]
    )

    # =============================
    # DROP LEAKAGE COLUMNS (SAME AS TRAINING)
    # These were dropped in driver_code.py before preprocessing
    # =============================
    drop_cols = ["date", "total_txn_nextday"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Drop NaN rows (same as training)
    df = df.dropna()

    # =============================
    # TAKE LAST ROW (after feature engineering, before preprocessing)
    # =============================
    X = df.iloc[-2:].copy()

    # =============================
    # PREPROCESS (EXACT SAME AS: transform_preprocess from pre_processing.py)
    # =============================
    X = _preprocess(X)

    # =============================
    # SELECT FEATURES FOR MODEL
    # Model was trained on these exact features after preprocessing
    # "features" = list(X_train.columns) from the model pickle
    # =============================
    if set(features).issubset(set(X.columns)):
        X_model = X[features].copy()
    else:
        # Handle case where some features are missing
        missing = set(features) - set(X.columns)
        print(f"⚠️ Warning: Missing features in inference data: {missing}")
        X_model = X.reindex(columns=features, fill_value=0)

    # =============================
    # PREDICT DELTA (next_day - current_day)
    # Note: Tree-based models don't require scaled data
    # =============================
    delta = model.predict(X_model)[0]

    # =============================
    # CALCULATE NEXT DAY PREDICTION
    # =============================
    next_day = current_value + delta

    return {
        "date": str(current_date),
        "current": float(current_value),
        "delta": float(delta),
        # "next_day": float(next_day)
    }