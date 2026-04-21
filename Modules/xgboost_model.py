import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# TRAIN XGBOOST MODEL
# -----------------------------
def train_xgboost(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = "total_txn_nextday"):
    """
    Train XGBoost model and evaluate performance
    """

    # Split features & target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # -----------------------------
    # MODEL
    # -----------------------------
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    # -----------------------------
    # METRICS
    # -----------------------------
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # Avoid division by zero in MAPE
    mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-9))) * 100

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }

    # Print results
    print("\n📊 XGBOOST PERFORMANCE")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
    print(f"MAPE : {mape:.2f}%")

    return model, preds, metrics


# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
def get_xgb_feature_importance(model, feature_names):
    """
    Get feature importance from XGBoost
    """

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    print("\n🔥 XGBOOST FEATURE IMPORTANCE:")
    print(importance_df.head(20))

    return importance_df