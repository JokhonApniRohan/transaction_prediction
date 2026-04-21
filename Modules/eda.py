import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap


# -----------------------------
# BASIC INFO (KEEP MINIMAL)
# -----------------------------
def basic_info(df: pd.DataFrame):
    print("\n📌 SHAPE:", df.shape)
    print("\n📌 MISSING VALUES:\n", df.isnull().sum())


# -----------------------------
# CORRELATION HEATMAP
# -----------------------------
def plot_correlation_heatmap(df: pd.DataFrame, target_col: str = None):
    df = df.copy()

    df = df.select_dtypes(include=np.number)

    if target_col and target_col in df.columns:
        cols = [c for c in df.columns if c != target_col] + [target_col]
        df = df[cols]

    corr = df.corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

    return corr


# -----------------------------
# UNIQUE PAIRWISE CORRELATION
# -----------------------------
def print_unique_correlations(df: pd.DataFrame, threshold: float = 0.0):
    """
    Prints each feature-pair correlation ONLY ONCE (no duplicates)
    """

    # df = df.select_dtypes(include=np.number)
    # corr = df.corr()

    # printed_pairs = set()

    # print("\n📊 UNIQUE FEATURE CORRELATIONS:\n")

    # for i in range(len(corr.columns)):
    #     for j in range(i + 1, len(corr.columns)):  # ensures no duplicates
    #         f1 = corr.columns[i]
    #         f2 = corr.columns[j]
    #         value = corr.iloc[i, j]

    #         if abs(value) >= threshold:
    #             print(f"{f1} ↔ {f2} : {value:.4f}")
    #             printed_pairs.add((f1, f2))


# -----------------------------
# SHAP VALUES (LIGHTGBM)
# -----------------------------
def plot_shap_values(model, X: pd.DataFrame):
    """
    SHAP explanation for feature importance
    """

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    print("\n📊 Generating SHAP summary plot...")

    shap.summary_plot(shap_values, X)


# -----------------------------
# FULL EDA PIPELINE (FOCUSED)
# -----------------------------
def run_full_eda(
    df: pd.DataFrame,
    target_col: str,
    model=None,
    shap_data: pd.DataFrame = None
):
    """
    Focused EDA for ML feature selection
    """

    basic_info(df)

    # Correlation heatmap
    corr = plot_correlation_heatmap(df, target_col)

    # Unique correlations
    print_unique_correlations(df, threshold=0.0)

    # SHAP (only if model provided)
    if model is not None and shap_data is not None:
        plot_shap_values(model, shap_data)