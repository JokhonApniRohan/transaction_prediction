import pandas as pd
import numpy as np


def create_bd_date_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Bangladesh-context feature engineering for transaction prediction models.
    Includes salary cycles, weekend structure, and cyclical encoding.
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # =============================
    # BASIC DATE FEATURES
    # =============================
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["month"] = df[date_col].dt.month
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["day_of_month"] = df[date_col].dt.day

    # Bangladesh weekend: Friday (4), Saturday (5)
    df["is_weekend"] = df["day_of_week"].isin([4, 5]).astype(int)

    # =============================
    # MONTH STRUCTURE FEATURES
    # =============================
    df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
    df["is_month_end"] = df[date_col].dt.is_month_end.astype(int)

    # =============================
    # 🇧🇩 SALARY / INCOME CYCLE (UPDATED FOR BANGLADESH)
    # =============================

    # Primary salary window (UPDATED: 1–10 days)
    df["is_salary_period"] = df["day_of_month"].between(1, 10).astype(int)

    # Salary intensity breakdown (more granular signal)
    df["salary_peak"] = df["day_of_month"].between(1, 3).astype(int)
    df["salary_mid"] = df["day_of_month"].between(4, 7).astype(int)
    df["salary_late"] = df["day_of_month"].between(8, 10).astype(int)

    # Mid-month financial behavior
    df["is_mid_month"] = df["day_of_month"].between(11, 18).astype(int)

    # End-of-month liquidity cycle
    df["is_end_month_spike"] = df["day_of_month"].between(25, 31).astype(int)

    # Traditional payday anchors
    df["is_payday_1"] = (df["day_of_month"] <= 3).astype(int)
    df["is_payday_15"] = (df["day_of_month"] == 15).astype(int)
    df["is_payday_30"] = (df["day_of_month"] >= 28).astype(int)

    # Combined payday signal
    df["is_payday"] = (
        df["is_salary_period"] |
        df["is_payday_15"] |
        df["is_payday_30"]
    ).astype(int)

    # =============================
    # BEHAVIORAL INTERACTIONS
    # =============================

    # Weekend + salary interaction (VERY STRONG SIGNAL in BD)
    df["weekend_payday"] = (df["is_weekend"] & df["is_payday"]).astype(int)

    # Salary + campaign overlap effect
    df["salary_campaign_window"] = (df["is_salary_period"] | df["is_campaign"]).astype(int)

    # =============================
    # CYCLICAL ENCODING (IMPORTANT FOR ML MODELS)
    # =============================

    # Day of week
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Day of month
    df["dom_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
    df["dom_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31)

    return df