from load_and_explore_dataset import load_and_inspect_data
from date_info_generation import create_bd_date_features
from pre_processing import fit_transform_preprocess, transform_preprocess
from light_gbm import train_lightgbm, get_feature_importance
from xgboost_model import train_xgboost, get_xgb_feature_importance
from catboost_model import train_catboost, get_catboost_feature_importance
from eda import run_full_eda


# -----------------------------
# TIME-BASED SPLIT
# -----------------------------
def time_based_split(df, date_col="date", train_ratio=0.8):
    df = df.sort_values(date_col).reset_index(drop=True)

    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print("\n📅 Time-based split done")
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    return train_df, test_df


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
    print("Original shape:", df.shape)

    target_col = "total_txn_nextday"

    # RAW EDA
    print("\n📊 Running RAW EDA...")
    run_full_eda(df, target_col=target_col)

    # SPLIT
    train_df, test_df = time_based_split(df, date_col="date")

    # FEATURE ENGINEERING
    train_df = create_bd_date_features(train_df, date_col="date")
    test_df = create_bd_date_features(test_df, date_col="date")

    print("\n✅ Feature engineering completed")

    # FEATURE EDA
    print("\n📊 Running FEATURE EDA...")
    run_full_eda(train_df, target_col=target_col)

    # PREPROCESSING
    train_df, scaler = fit_transform_preprocess(train_df)
    test_df = transform_preprocess(test_df, scaler)

    print("\n✅ Preprocessing completed")

    # DROP DATE for ML models
    train_df_ml = train_df.drop(columns=["date"])
    test_df_ml = test_df.drop(columns=["date"])

    # =========================
    # TRAIN ML MODELS
    # =========================
    print("\n🚀 Training LightGBM...")
    lgb_model, lgb_preds, lgb_metrics = train_lightgbm(train_df_ml, test_df_ml)

    print("\n🚀 Training XGBoost...")
    xgb_model, xgb_preds, xgb_metrics = train_xgboost(train_df_ml, test_df_ml)

    print("\n🚀 Training CatBoost...")
    cat_model, cat_preds, cat_metrics = train_catboost(train_df_ml, test_df_ml)

    # =========================
    # MODEL COMPARISON
    # =========================
    print("\n📊 MODEL COMPARISON")

    print("\nLightGBM:", lgb_metrics)
    print("XGBoost:", xgb_metrics)
    print("CatBoost:", cat_metrics)

    models = {
        "LightGBM": (lgb_model, lgb_metrics, get_feature_importance),
        "XGBoost": (xgb_model, xgb_metrics, get_xgb_feature_importance),
        "CatBoost": (cat_model, cat_metrics, get_catboost_feature_importance),
    }

    best_name = min(models, key=lambda x: models[x][1]["RMSE"])
    best_model, best_metrics, importance_fn = models[best_name]

    print(f"\n🏆 Best ML Model: {best_name}")

    # =========================
    # SHAP ANALYSIS
    # =========================
    print("\n📊 Running SHAP analysis...")

    X_train = train_df_ml.drop(columns=[target_col])
    X_sample = X_train.sample(min(2000, len(X_train)), random_state=42)

    run_full_eda(
        train_df_ml,
        target_col=target_col,
        model=best_model,
        shap_data=X_sample
    )

    # FEATURE IMPORTANCE
    importance_df = importance_fn(best_model, X_train.columns)

    print("\n📊 Pipeline completed successfully!")


if __name__ == "__main__":
    main()