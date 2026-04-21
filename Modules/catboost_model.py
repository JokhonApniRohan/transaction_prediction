import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


USE_LOG_TRANSFORM = True


# -----------------------------
# TRAIN CATBOOST
# -----------------------------
def train_catboost(X_train, y_train, X_test, y_test):

    model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.03,
        depth=8,
        loss_function="RMSE",
        random_seed=42,
        verbose=200
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # -----------------------------
    # SAFE INVERSE TRANSFORM
    # -----------------------------
    if USE_LOG_TRANSFORM:
        y_test_orig = (y_test)
        y_pred_orig = (y_pred)
    else:
        y_test_orig = y_test
        y_pred_orig = y_pred

    metrics = {
        "MAE": mean_absolute_error(y_test_orig, y_pred_orig),
        "RMSE": np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
        "R2": r2_score(y_test_orig, y_pred_orig),
        "MAPE": np.mean(np.abs((y_test_orig - y_pred_orig) /
                              (y_test_orig + 1e-9))) * 100
    }

    print("\n📊 CATBOOST PERFORMANCE")
    print(metrics)

    return model, y_pred_orig, metrics


# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
def get_catboost_feature_importance(model, feature_names):

    importance = model.get_feature_importance()

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    print("\n🔥 CATBOOST FEATURE IMPORTANCE:")
    print(importance_df.head(20))

    return importance_df