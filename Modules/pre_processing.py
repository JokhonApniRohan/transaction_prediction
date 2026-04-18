import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
def fit_transform_preprocess(df: pd.DataFrame, show_heatmap: bool = True):
    """
    Fit preprocessing pipeline on training data.
    Returns processed dataframe + fitted scaler.
    """

    df = _preprocess(df)

    scaler = StandardScaler()
    df[LOG_COLS] = scaler.fit_transform(df[LOG_COLS])

    if show_heatmap:
        _plot_correlation_heatmap(df)

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
    df = df.fillna(0)

    return df


# -----------------------------
# CORRELATION HEATMAP
# -----------------------------
def _plot_correlation_heatmap(df: pd.DataFrame):
    """
    Plot correlation heatmap with target column at the end.
    """

    df = df.copy()

    # Move target column to the end for better visualization
    if "total_txn" in df.columns:
        cols = [c for c in df.columns if c != "total_txn"] + ["total_txn"]
        df = df[cols]

    # Correlation matrix
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        linewidths=0.5
    )

    plt.title("Feature Correlation Heatmap (Target at End)")
    plt.show()