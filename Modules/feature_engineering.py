import pandas as pd
import numpy as np


def create_bd_date_features(
    df: pd.DataFrame,
    date_col: str = "date",
    base_series: pd.Series = None
) -> pd.DataFrame:

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # =====================================================
    # BASIC DATE FEATURES
    # =====================================================
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["month"] = df[date_col].dt.month
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["day_of_month"] = df[date_col].dt.day

    df["is_weekend"] = df["day_of_week"].isin([4, 5]).astype(int)

    df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
    df["is_month_end"] = df[date_col].dt.is_month_end.astype(int)

    # =====================================================
    # 🇧🇩 SALARY FEATURES
    # =====================================================
    df["is_salary_period"] = df["day_of_month"].between(1, 10).astype(int)
    df["salary_peak"] = df["day_of_month"].between(1, 3).astype(int)
    df["salary_mid"] = df["day_of_month"].between(4, 7).astype(int)
    df["salary_late"] = df["day_of_month"].between(8, 10).astype(int)

    df["is_mid_month"] = df["day_of_month"].between(11, 18).astype(int)
    df["is_end_month_spike"] = df["day_of_month"].between(25, 31).astype(int)

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

    df["salary_campaign_window"] = (
        df["is_salary_period"] | df.get("is_campaign", 0)
    ).astype(int)

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
    # 📊 TARGET-BASED FEATURES
    # =====================================================
    if base_series is not None:

        # =====================
        # 🔥 DELTA TARGET (NEW)
        # =====================
        df["delta"] = base_series - base_series.shift(1)

        # =====================
        # LAG SAFE BASE
        # =====================
        shifted = base_series.shift(1)

        # Rolling stats
        df["rolling_mean_7"] = shifted.rolling(7).mean()
        df["rolling_mean_30"] = shifted.rolling(30).mean()

        df["rolling_std_7"] = shifted.rolling(7).std()
        df["rolling_std_30"] = shifted.rolling(30).std()

        df["rolling_max_7"] = shifted.rolling(7).max()
        df["rolling_max_30"] = shifted.rolling(30).max()

        df["rolling_min_7"] = shifted.rolling(7).min()
        df["rolling_min_30"] = shifted.rolling(30).min()

        # =====================================================
        # 🧠 REGIME FEATURES
        # =====================================================
        rolling_mean = shifted.rolling(30).mean()
        rolling_std = shifted.rolling(30).std().replace(0, np.nan)

        z_score = (shifted - rolling_mean) / rolling_std

        df["regime_z_score"] = z_score

        df["regime_low"] = (z_score < -1).astype(int)
        df["regime_normal"] = ((z_score >= -1) & (z_score <= 1)).astype(int)
        df["regime_high"] = (z_score > 1).astype(int)
        df["regime_spike"] = (z_score > 2).astype(int)

        # Volatility
        vol_7 = shifted.rolling(7).std()
        vol_30 = shifted.rolling(30).std()

        df["volatility_7"] = vol_7
        df["volatility_30"] = vol_30

        df["volatility_ratio"] = vol_7 / (vol_30 + 1e-9)
        df["regime_volatile"] = (df["volatility_ratio"] > 1.2).astype(int)

    return df