import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================
# GLOBAL SETTING (IMPORTANT)
# =============================
USE_LOG_TRANSFORM = True


# -----------------------------
# TRAIN LIGHTGBM MODEL
# -----------------------------
def train_lightgbm(X_train, y_train, X_test, y_test):

    model = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # -----------------------------
    # SAFE INVERSE TRANSFORM
    # -----------------------------
    if USE_LOG_TRANSFORM:
        y_test_orig =(y_test)
        y_pred_orig = (y_pred)
    else:
        y_test_orig = y_test
        y_pred_orig = y_pred

    metrics = evaluate_model(y_test_orig, y_pred_orig)

    return model, y_pred_orig, metrics


# -----------------------------
# EVALUATION
# -----------------------------
def evaluate_model(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    mape = np.mean(np.abs((y_true - y_pred) /
                          (y_true + 1e-9))) * 100

    print("\n📊 LIGHTGBM PERFORMANCE")
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

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\n🔥 LIGHTGBM FEATURE IMPORTANCE:")
    print(importance_df.head(20))

    return importance_df