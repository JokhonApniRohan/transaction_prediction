import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# TRAIN XGBOOST
# -----------------------------
def train_xgboost(X_train, y_train, X_test, y_test):

    model = XGBRegressor(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # -----------------------------
    # PREDICT
    # -----------------------------
    y_pred = model.predict(X_test)

    # -----------------------------
    # NO TRANSFORM (RAW SCALE)
    # -----------------------------
    y_test_orig = y_test.values if hasattr(y_test, "values") else y_test
    y_pred_orig = y_pred

    # -----------------------------
    # METRICS
    # -----------------------------
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    r2 = r2_score(y_test_orig, y_pred_orig)

    epsilon = 1e-9
    mape = np.mean(np.abs((y_test_orig - y_pred_orig) /
                          (y_test_orig + epsilon))) * 100

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }

    # -----------------------------
    # CLEAN PRINT
    # -----------------------------
    print("\n" + "=" * 50)
    print("📊 XGBOOST PERFORMANCE (RAW SCALE)")
    print("=" * 50)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
    print(f"MAPE : {mape:.2f}%")
    print("=" * 50)

    return model, y_pred_orig, metrics


# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
def get_xgb_feature_importance(model, feature_names):

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\n🔥 XGBOOST FEATURE IMPORTANCE:")
    print(importance_df.head(20))

    return importance_df