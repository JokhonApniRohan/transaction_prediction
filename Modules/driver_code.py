from load_and_explore_dataset import load_and_inspect_data
from date_info_generation import create_bd_date_features
from pre_processing import fit_transform_preprocess, transform_preprocess
from light_gbm import train_lightgbm, get_feature_importance
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

    # =========================
    # Step 1: Load dataset
    # =========================
    file_path = r"D:\Projects\Transaction Prediction\Datasets\new_transaction_data.csv"
    df = load_and_inspect_data(file_path)

    if df is None:
        print("❌ Failed to load dataset")
        return

    print("\n✅ Dataset loaded successfully")
    print("Original shape:", df.shape)

    # =========================
    # 🔥 Step 1.5: RAW EDA
    # =========================
    print("\n📊 Running RAW EDA...")
    run_full_eda(df, target_col="total_txn_nextday", date_col="date")

    # =========================
    # Step 2: Time-based split
    # =========================
    train_df, test_df = time_based_split(df, date_col="date")

    # =========================
    # Step 3: Feature Engineering
    # =========================
    train_df = create_bd_date_features(train_df, date_col="date")
    test_df = create_bd_date_features(test_df, date_col="date")

    print("\n✅ Feature engineering completed")
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    # =========================
    # 🔥 Step 3.5: POST-FEATURE EDA (OPTIONAL)
    # =========================
    print("\n📊 Running FEATURE-LEVEL EDA...")
    run_full_eda(train_df, target_col="total_txn_nextday", date_col="date")

    # =========================
    # Step 4: Preprocessing
    # =========================
    train_df, scaler = fit_transform_preprocess(train_df)
    test_df = transform_preprocess(test_df, scaler)

    print("\n✅ Preprocessing completed")
    print("Final train shape:", train_df.shape)
    print("Final test shape:", test_df.shape)

    # =========================
    # Step 5: Drop date
    # =========================
    train_df = train_df.drop(columns=["date"])
    test_df = test_df.drop(columns=["date"])

    # =========================
    # Step 6: Model Training
    # =========================
    model, preds, metrics = train_lightgbm(train_df, test_df)

    # =========================
    # Step 7: Feature Importance
    # =========================
    X_train = train_df.drop(columns=["total_txn_nextday"])
    importance_df = get_feature_importance(model, X_train.columns)

    print("\n📊 Pipeline completed successfully!")


if __name__ == "__main__":
    main()