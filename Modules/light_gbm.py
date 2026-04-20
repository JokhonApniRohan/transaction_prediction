import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# TRAIN LIGHTGBM MODEL
# -----------------------------
def train_lightgbm(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = "total_txn_nextday"):
    """
    Train LightGBM model and evaluate on test data.

    Returns:
    - model
    - predictions
    - metrics dict
    """

    # -------------------------
    # Split features & target
    # -------------------------
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # -------------------------
    # Initialize model
    # -------------------------
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42
    )

    # -------------------------
    # Train model
    # -------------------------
    model.fit(X_train, y_train)

    # -------------------------
    # Predict
    # -------------------------
    y_pred = model.predict(X_test)

    # -------------------------
    # Evaluate
    # -------------------------
    metrics = evaluate_model(y_test, y_pred)

    return model, y_pred, metrics


# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
def evaluate_model(y_true, y_pred):
    """
    Calculate evaluation metrics including MAPE.
    """

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # -------------------------
    # MAPE (handle division safely)
    # -------------------------
    epsilon = 1e-10  # to avoid division by zero
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    print("\n📊 MODEL PERFORMANCE")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
    print(f"MAPE : {mape:.2f}%")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }


# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
def get_feature_importance(model, feature_names):
    """
    Returns sorted feature importance dataframe.
    """

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\n🔥 TOP FEATURE IMPORTANCE:")
    print(importance_df.head(20))

    return importance_df