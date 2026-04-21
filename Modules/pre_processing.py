import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# -----------------------------
# DYNAMIC FEATURE SELECTOR
# -----------------------------
def get_feature_columns(df: pd.DataFrame, target_col: str = "target"):
    """
    Select only valid numeric, non-binary, non-target features.
    """

    feature_cols = []

    for col in df.columns:

        # Skip target
        if col == target_col:
            continue

        # Must be numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Skip binary columns (0/1 only)
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 2:
            if set(unique_vals).issubset({0, 1}):
                continue

        # Skip constant columns
        if df[col].nunique() <= 1:
            continue

        feature_cols.append(col)

    return feature_cols


# -----------------------------
# FIT TRANSFORM (TRAINING)
# -----------------------------
def fit_transform_preprocess(df: pd.DataFrame, target_col: str = "target"):
    df = _preprocess(df)

    feature_cols = get_feature_columns(df, target_col)

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df, scaler, feature_cols


# -----------------------------
# TRANSFORM (TEST / INFERENCE)
# -----------------------------
def transform_preprocess(df: pd.DataFrame, scaler: StandardScaler, feature_cols, target_col: str = "target"):
    df = _preprocess(df)

    df[feature_cols] = scaler.transform(df[feature_cols])

    return df


# -----------------------------
# CORE PREPROCESSING
# -----------------------------
def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # =====================================================
    # 1. TREND FEATURES
    # =====================================================

    df["diff_1"] = df["lag_1"] - df.get("lag_2", df["lag_1"])

    df["trend_7"] = df["lag_1"] - df["lag_7"]

    df["growth_7"] = (
        (df["lag_1"] - df["lag_7"]) /
        (np.abs(df["lag_7"]) + 1)
    )

    # =====================================================
    # 2. LOG TRANSFORM
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
    # 3. MISSING VALUES
    # =====================================================
    df = df.fillna(-1)

    return df