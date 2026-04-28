import pandas as pd
import numpy as np
from hijri_converter import convert


def create_prophet_features(
    df: pd.DataFrame,
    date_col: str = "date",
    campaign_col: str = "campaign_count"
) -> pd.DataFrame:
    """
    Prophet-safe feature engineering module.

    Rules:
    - ONLY include features that are known in future OR rule-based
    - NO lag / rolling / target-derived features
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # =====================================================
    # HIJRI CONVERSION (REQUIRED FOR FESTIVAL FLAG)
    # =====================================================
    def get_festival_flag(date):
        hijri = convert.Gregorian(date.year, date.month, date.day).to_hijri()

        h_month = hijri.month
        h_day = hijri.day

        # Eid-ul-Fitr → 1 Shawwal
        # Flag: 10 days before + day 1
        is_eid_fitr_window = (
            (h_month == 10 and h_day == 1) or  # Eid day
            (h_month == 9 and h_day >= 20)     # last ~10 days of Ramadan
        )

        # Eid-ul-Adha → 10 Dhul Hijjah
        # Your requirement: previous 10 days of Dhul Hijjah
        is_eid_adha_window = (
            (h_month == 12 and h_day <= 10)
        )

        return int(is_eid_fitr_window or is_eid_adha_window)

    df["festival_flag"] = df[date_col].apply(get_festival_flag)

    # =====================================================
    # BASIC CALENDAR FEATURES
    # =====================================================
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["month"] = df[date_col].dt.month
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["day_of_month"] = df[date_col].dt.day

    df["is_weekend"] = df["day_of_week"].isin([4, 5]).astype(int)

    df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
    df["is_month_end"] = df[date_col].dt.is_month_end.astype(int)

    # =====================================================
    # 🇧🇩 SALARY / BEHAVIOR FEATURES
    # =====================================================
    df["is_salary_period"] = df["day_of_month"].between(1, 10).astype(int)

    df["salary_peak"] = df["day_of_month"].between(1, 3).astype(int)
    df["salary_mid"] = df["day_of_month"].between(4, 7).astype(int)
    df["salary_late"] = df["day_of_month"].between(8, 10).astype(int)

    df["is_mid_month"] = df["day_of_month"].between(11, 18).astype(int)
    df["is_end_month_spike"] = df["day_of_month"].between(25, 31).astype(int)

    # Payday signals
    df["is_payday_1"] = (df["day_of_month"] <= 3).astype(int)
    df["is_payday_15"] = (df["day_of_month"] == 15).astype(int)
    df["is_payday_30"] = (df["day_of_month"] >= 28).astype(int)

    df["is_payday"] = (
        df["is_salary_period"] |
        df["is_payday_15"] |
        df["is_payday_30"]
    ).astype(int)

    # =====================================================
    # INTERACTIONS
    # =====================================================
    df["weekend_payday"] = (df["is_weekend"] & df["is_payday"]).astype(int)

    # =====================================================
    # CAMPAIGN FEATURE
    # =====================================================
    if campaign_col in df.columns:
        df[campaign_col] = df[campaign_col].fillna(0).astype(int)
        df["campaign_window"] = df[campaign_col]

    # =====================================================
    # CYCLICAL ENCODING
    # =====================================================
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["dom_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
    df["dom_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31)

    # =====================================================
    # CLEANUP
    # =====================================================
    df = df.drop(columns=[date_col])

    return df