import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# -----------------------------
# CONFIG
# -----------------------------
LOG_COLS = [
    "txn_count",
    "count_camp",
    "lag_1",
    "lag_7",
    "lag_30",
    "rolling_7",
    "rolling_30"
]


# -----------------------------
# FIT TRANSFORM (TRAINING)
# -----------------------------
def fit_transform_preprocess(df: pd.DataFrame):
    """
    Fit preprocessing pipeline on training data.
    Returns processed dataframe + fitted scaler.
    """

    df = _preprocess(df)

    scaler = StandardScaler()
    df[LOG_COLS] = scaler.fit_transform(df[LOG_COLS])

    return df, scaler


# -----------------------------
# TRANSFORM (TEST / INFERENCE)
# -----------------------------
def transform_preprocess(df: pd.DataFrame, scaler: StandardScaler):
    """
    Apply preprocessing using fitted scaler.
    """

    df = _preprocess(df)
    df[LOG_COLS] = scaler.transform(df[LOG_COLS])

    return df


# -----------------------------
# CORE PREPROCESSING
# -----------------------------
def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Log transform skewed features
    for col in LOG_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # Handle missing values
    df = df.fillna(-1)

    return df