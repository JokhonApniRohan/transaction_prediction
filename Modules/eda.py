import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# -----------------------------
# BASIC INFO
# -----------------------------
def basic_info(df: pd.DataFrame):
    print("\n📌 SHAPE:", df.shape)
    print("\n📌 DATA TYPES:\n", df.dtypes)
    print("\n📌 MISSING VALUES:\n", df.isnull().sum())
    print("\n📌 DESCRIBE:\n", df.describe())


# -----------------------------
# TARGET DISTRIBUTION
# -----------------------------
def plot_target_distribution(df: pd.DataFrame, target_col: str):
    plt.figure(figsize=(8, 5))
    sns.histplot(df[target_col], kde=True)
    plt.title(f"Distribution of {target_col}")
    plt.show()


# -----------------------------
# FEATURE DISTRIBUTIONS
# -----------------------------
def plot_feature_distributions(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"{col} Distribution")
        plt.show()


# -----------------------------
# CORRELATION HEATMAP
# -----------------------------
def plot_correlation_heatmap(df: pd.DataFrame, target_col: str = None):
    df = df.copy()

    if target_col and target_col in df.columns:
        cols = [c for c in df.columns if c != target_col] + [target_col]
        df = df[cols]

    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()


# -----------------------------
# LAG RELATIONSHIP
# -----------------------------
def plot_lag_relationship(df: pd.DataFrame, target_col: str, lag_col: str):
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=df[lag_col], y=df[target_col])
    plt.title(f"{lag_col} vs {target_col}")
    plt.show()


# -----------------------------
# BOXPLOT (OUTLIERS)
# -----------------------------
def plot_boxplots(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"{col} Boxplot")
        plt.show()


# -----------------------------
# FULL EDA PIPELINE
# -----------------------------
def run_full_eda(df: pd.DataFrame, target_col: str, date_col: str = None):
    basic_info(df)

    plot_target_distribution(df, target_col)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df, target_col)

    # Example lag checks (if exist)
    for lag in ["lag_1", "lag_7", "lag_30"]:
        if lag in df.columns:
            plot_lag_relationship(df, target_col, lag)

    plot_boxplots(df)