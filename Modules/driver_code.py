from load_and_explore_dataset import load_and_inspect_data
from date_info_generation import create_bd_date_features
from pre_processing import fit_transform_preprocess, transform_preprocess


def time_based_split(df, date_col="date", train_ratio=0.8):
    df = df.sort_values(date_col).reset_index(drop=True)

    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print("\n📅 Time-based split done")
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    return train_df, test_df


def main():

    # =========================
    # Step 1: Load dataset
    # =========================
    file_path = r"D:\Projects\Transaction Prediction\Datasets\Transaction Data.csv"
    df = load_and_inspect_data(file_path)

    if df is None:
        print("❌ Failed to load dataset")
        return

    print("\n✅ Dataset loaded successfully")
    print("Original shape:", df.shape)

    # =========================
    # Step 2: Sort + Split (IMPORTANT: BEFORE FEATURE ENGINEERING)
    # =========================
    train_df, test_df = time_based_split(df, date_col="date")

    # =========================
    # Step 3: Feature Engineering (separately applied)
    # =========================
    train_df = create_bd_date_features(train_df, date_col="date")
    test_df = create_bd_date_features(test_df, date_col="date")

    print("\n✅ Feature engineering completed")
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    # =========================
    # Step 4: Preprocessing (FIT on train, TRANSFORM on test)
    # =========================
    train_df, scaler = fit_transform_preprocess(train_df, show_heatmap=True)
    test_df = transform_preprocess(test_df, scaler)

    print("\n✅ Preprocessing completed successfully")
    print("Final train shape:", train_df.shape)
    print("Final test shape:", test_df.shape)

    # =========================
    # Step 5: Drop date safely AFTER all processing
    # =========================
    train_df = train_df.drop(columns=["date"])
    test_df = test_df.drop(columns=["date"])

    print("\n📊 Ready for model training!")

    print("\nTrain columns sample:")
    print(train_df.columns.tolist()[-10:])


if __name__ == "__main__":
    main()