import pandas as pd
from scipy.stats import skew
import csv

def load_and_inspect_data(file_path):
    """
    Load dataset and perform inspection + skewness analysis (robust version).
    """

    # -----------------------------
    # LOAD DATASET (FIXED)
    # -----------------------------
    print("Loading dataset...")

    try:
        df = pd.read_csv(
            file_path,
            sep=',',                 # adjust if needed
            on_bad_lines='skip',     # skip problematic rows
            low_memory=False,        # better dtype handling
            encoding='utf-8'         # change to 'latin1' if needed
        )
    except Exception as e:
        print("Error loading file:", e)
        return None

    print(f"✓ Dataset loaded successfully!")
    print(f"\nDataset Shape: {df.shape}")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")

    # -----------------------------
    # BASIC INFO
    # -----------------------------
    print("\n" + "="*80)
    print("DATASET OVERVIEW")
    print("="*80)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing Values:")
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })

    print(
        missing_data[missing_data['Missing_Count'] > 0]
        .sort_values('Missing_Percentage', ascending=False)
    )

    print("\nBasic Statistics:")
    print(df.describe())

    # -----------------------------
    # COLUMN TYPE ANALYSIS
    # -----------------------------
    print("\n" + "="*80)
    print("COLUMN TYPE ANALYSIS")
    print("="*80)

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    binary_numeric_cols = []
    true_numeric_cols = []

    for col in numeric_cols:
        unique_vals = df[col].dropna().unique()

        if len(unique_vals) == 2:
            binary_numeric_cols.append(col)
        else:
            true_numeric_cols.append(col)

    print("\n📌 CATEGORICAL COLUMNS:")
    for col in categorical_cols:
        print(f"- {col}: {df[col].nunique(dropna=True)} categories")

    print("\n📌 BINARY NUMERIC COLUMNS:")
    for col in binary_numeric_cols:
        print(f"- {col}")

    print("\n📌 NUMERIC COLUMNS:")
    for col in true_numeric_cols:
        print(f"- {col}")

    # -----------------------------
    # SKEWNESS ANALYSIS
    # -----------------------------
    print("\n" + "="*80)
    print("SKEWNESS ANALYSIS")
    print("="*80)

    skew_data = []

    for col in true_numeric_cols:
        col_data = df[col].dropna()

        if len(col_data) == 0:
            continue

        skewness = skew(col_data)

        if abs(skewness) < 0.5:
            dist_type = "Normal"
        elif skewness > 0.5:
            dist_type = "Right Skewed"
        else:
            dist_type = "Left Skewed"

        skew_data.append([col, round(skewness, 2), dist_type])

    skew_df = pd.DataFrame(skew_data, columns=["Column", "Skewness", "Distribution"])
    skew_df = skew_df.sort_values(by="Skewness", key=abs, ascending=False)

    print(skew_df)

    return df