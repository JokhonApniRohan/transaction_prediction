import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from load_and_explore_dataset import load_and_inspect_data
from feature_engineering import create_bd_date_features
from pre_processing import fit_transform_preprocess, transform_preprocess
from light_gbm import train_lightgbm, get_feature_importance
from xgboost_model import train_xgboost, get_xgb_feature_importance
from catboost_model import train_catboost, get_catboost_feature_importance


# -----------------------------
# TIME SPLIT
# -----------------------------
def time_based_split(df, date_col="date", train_ratio=0.8):
    df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():

    file_path = r"D:\Projects\Transaction Prediction\Datasets\new_transaction_data.csv"
    df = load_and_inspect_data(file_path)

    if df is None:
        print("❌ Failed to load dataset")
        return

    print("\n✅ Dataset loaded successfully")

    # =========================
    # TARGET SETUP (CORRECT)
    # =========================
    raw_target = "total_txn_nextday"
    model_target = "target"

    # 🔥 FIX: use next-day value directly (NOT delta)
    df[model_target] = (df[raw_target])

    # =========================
    # SPLIT FIRST (NO LEAKAGE)
    # =========================
    train_df, test_df = time_based_split(df, "date")

    # =========================
    # FEATURE ENGINEERING
    # =========================
    train_df = create_bd_date_features(
        train_df,
        date_col="date",
        base_series=train_df["total_txn_current"]
    )

    test_df = create_bd_date_features(
        test_df,
        date_col="date",
        base_series=test_df["total_txn_current"]
    )

    print("\n✅ Feature engineering completed")

    # =========================
    # DROP LEAKAGE COLS
    # =========================
    drop_cols = ["date", raw_target]

    train_df = train_df.drop(columns=drop_cols).dropna()
    test_df = test_df.drop(columns=drop_cols).dropna()

    # =========================
    # SPLIT X / Y
    # =========================
    y_train = train_df[model_target]
    y_test = test_df[model_target]

    X_train_raw = train_df.drop(columns=[model_target])
    X_test_raw = test_df.drop(columns=[model_target])

    # =========================
    # PREPROCESS
    # =========================
    X_train, scaler, feature_cols = fit_transform_preprocess(X_train_raw)

    X_test = transform_preprocess(
        X_test_raw,
        scaler,
        feature_cols
    )

    # =========================
    # SAFETY CHECK
    # =========================
    assert list(X_train.columns) == list(X_test.columns), "Feature mismatch!"

    # =========================
    # TRAIN MODELS
    # =========================
    print("\n🚀 Training LightGBM...")
    lgb_model, lgb_preds, lgb_metrics = train_lightgbm(X_train, y_train, X_test, y_test)

    print("\n🚀 Training XGBoost...")
    xgb_model, xgb_preds, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)

    print("\n🚀 Training CatBoost...")
    cat_model, cat_preds, cat_metrics = train_catboost(X_train, y_train, X_test, y_test)

    # =========================
    # MODEL COMPARISON
    # =========================
    print("\n📊 MODEL COMPARISON")
    print("LightGBM:", lgb_metrics)
    print("XGBoost:", xgb_metrics)
    print("CatBoost:", cat_metrics)

    models = {
        "LightGBM": (lgb_model, lgb_metrics, get_feature_importance),
        "XGBoost": (xgb_model, xgb_metrics, get_xgb_feature_importance),
        "CatBoost": (cat_model, cat_metrics, get_catboost_feature_importance),
    }

    best_name = min(models, key=lambda x: models[x][1]["RMSE"])
    best_model, best_metrics, importance_fn = models[best_name]

    print(f"\n🏆 Best Model: {best_name}")

    importance_df = importance_fn(best_model, X_train.columns)

    # =========================
    # 📊 DISTRIBUTION CHECK
    # =========================
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(np.expm1(y_train), bins=50)
    plt.title("Original Target")

    plt.subplot(1, 2, 2)
    plt.hist(np.expm1(lgb_preds), bins=50)
    plt.title("Predictions")

    plt.show()

    print("\n📊 Pipeline completed successfully!")


if __name__ == "__main__":
    main()