import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# TRAIN CATBOOST MODEL
# -----------------------------
def train_catboost(train_df: pd.DataFrame,
                   test_df: pd.DataFrame,
                   target_col: str = "total_txn_nextday"):
    """
    Train CatBoost model and evaluate performance
    """

    # -----------------------------
    # SPLIT FEATURES & TARGET
    # -----------------------------
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # -----------------------------
    # MODEL
    # -----------------------------
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        random_seed=42,
        verbose=200
    )

    # -----------------------------
    # TRAIN
    # -----------------------------
    model.fit(X_train, y_train)

    # -----------------------------
    # PREDICT
    # -----------------------------
    preds = model.predict(X_test)

    # -----------------------------
    # METRICS
    # -----------------------------
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-9))) * 100

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }

    # -----------------------------
    # PRINT RESULTS
    # -----------------------------
    print("\n📊 CATBOOST PERFORMANCE")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
    print(f"MAPE : {mape:.2f}%")

    return model, preds, metrics


# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
def get_catboost_feature_importance(model, feature_names):
    """
    Get feature importance from CatBoost
    """

    importance = model.get_feature_importance()

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    print("\n🔥 CATBOOST FEATURE IMPORTANCE:")
    print(importance_df.head(20))

    return importance_df