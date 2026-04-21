import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# -----------------------------
# CONFIG
# -----------------------------
FEATURE_COLS = [
    "txn_count",
    "count_camp",
    "lag_1",
    "lag_7",
    "lag_30",
    "rolling_7",
    "rolling_30",
    "diff_1",
    "trend_7",
    "growth_7"
]


# -----------------------------
# FIT TRANSFORM (TRAINING)
# -----------------------------
def fit_transform_preprocess(df: pd.DataFrame):
    df = _preprocess(df)

    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    return df, scaler


# -----------------------------
# TRANSFORM (TEST / INFERENCE)
# -----------------------------
def transform_preprocess(df: pd.DataFrame, scaler: StandardScaler):
    df = _preprocess(df)

    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])

    return df


# -----------------------------
# CORE PREPROCESSING
# -----------------------------
def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # =====================================================
    # 1. TREND FEATURES (RAW SPACE)
    # =====================================================

    # Short-term trend
    df["diff_1"] = df["lag_1"] - df.get("lag_2", df["lag_1"])

    # Weekly trend
    df["trend_7"] = df["lag_1"] - df["lag_7"]

    # Growth rate (SAFE VERSION)
    df["growth_7"] = (
        (df["lag_1"] - df["lag_7"]) /
        (np.abs(df["lag_7"]) + 1)
    )

    # =====================================================
    # 2. LOG TRANSFORM SKEWED FEATURES
    # =====================================================
    log_cols = [
        "txn_count",
        "count_camp",
        "lag_1",
        "lag_7",
        "lag_30",
        "rolling_7",
        "rolling_30"
    ]

    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # =====================================================
    # 3. MISSING VALUE HANDLING
    # =====================================================
    df = df.fillna(-1)

    return df