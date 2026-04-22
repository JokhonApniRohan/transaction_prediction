import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from load_and_explore_dataset import load_and_inspect_data
from feature_engineering import create_bd_date_features
from pre_processing import fit_transform_preprocess, transform_preprocess
from light_gbm import train_lightgbm, get_feature_importance
from xgboost_model import train_xgboost, get_xgb_feature_importance
from catboost_model import train_catboost, get_catboost_feature_importance


# =====================================================
# PROJECT PATH (PORTABLE)
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "Modules"
MODEL_DIR.mkdir(exist_ok=True)


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

    file_path = BASE_DIR / "Datasets" / "new_transaction_data.csv"
    df = load_and_inspect_data(file_path)

    if df is None:
        print("❌ Failed to load dataset")
        return

    print("\n✅ Dataset loaded successfully")

    # =========================
    # TARGET
    # =========================
    raw_target = "total_txn_nextday"
    model_target = "target"

    # =========================
    # SPLIT FIRST (NO LEAKAGE)
    # =========================
    train_df, test_df = time_based_split(df, "date")

    # =========================
    # 📊 OUTLIER HANDLING (TRAIN ONLY)
    # =========================
    print("\n📊 IQR OUTLIER HANDLING")

    Q1 = train_df[raw_target].quantile(0.25)
    Q3 = train_df[raw_target].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"Lower bound: {lower_bound}")
    print(f"Upper bound: {upper_bound}")

    train_df[raw_target] = train_df[raw_target].clip(lower_bound, upper_bound)
    test_df[raw_target] = test_df[raw_target].clip(lower_bound, upper_bound)

    print("✅ Outliers clipped using IQR")

    # =========================
    # TARGET COLUMN
    # =========================
    train_df[model_target] = train_df[raw_target]
    test_df[model_target] = test_df[raw_target]

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

    # =========================
    # SAVE MODEL (PORTABLE)
    # =========================
    model_path = "./Model/best_model.pkl"

    joblib.dump({
        "model": best_model,
        "features": list(X_train.columns),
        "metrics": best_metrics,
        "scaler": scaler
    }, model_path)

    print(f"\n💾 Model saved at: {model_path}")

    importance_df = importance_fn(best_model, X_train.columns)

    # =========================
    # PLOT
    # =========================
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(y_train, bins=50)
    plt.title("Target Distribution (Train)")

    plt.subplot(1, 2, 2)
    plt.hist(lgb_preds, bins=50)
    plt.title("Predictions (LightGBM)")

    plt.show()

    print("\n📊 Pipeline completed successfully!")


if __name__ == "__main__":
    main()