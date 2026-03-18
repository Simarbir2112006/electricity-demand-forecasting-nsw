import os
import numpy as np
import pandas as pd
import holidays
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

def build_features(df_raw):
    df = df_raw.copy()
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])
    df = df.drop(columns=[c for c in ["REGION","PERIODTYPE"] if c in df.columns])
    df = df.sort_values("SETTLEMENTDATE").set_index("SETTLEMENTDATE")
    df = df.resample('30min').mean()

    try:
        nsw = holidays.Australia(state="NSW", years=range(df.index.year.min(), df.index.year.max() + 2))
        df["is_holiday"] = [1 if d in nsw else 0 for d in df.index.date]
    except Exception:
        df["is_holiday"] = (df.index.weekday >= 5).astype(int)

    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    df["month"] = df.index.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    df["lag_1"] = df["TOTALDEMAND"].shift(1)
    df["lag_2"] = df["TOTALDEMAND"].shift(2)
    df["lag_48"] = df["TOTALDEMAND"].shift(48)
    df["lag_336"] = df["TOTALDEMAND"].shift(336)
    df["rolling_48"] = df["TOTALDEMAND"].shift(1).rolling(window=48).mean()
    df["rolling_336"] = df["TOTALDEMAND"].shift(1).rolling(window=336).mean()

    df["day_type"] = df.index.dayofweek.map({0:0, 1:0, 2:0, 3:0, 4:0, 5:1, 6:2})
    df["hour_day_interaction"] = df["hour"].astype(str) + "_" + df["day_type"].astype(str)
    df["hour_day_interaction"] = LabelEncoder().fit_transform(df["hour_day_interaction"])

    daily_fourier = CalendarFourier(freq="D", order=4)
    yearly_fourier = CalendarFourier(freq="YE", order=2)
    dp = DeterministicProcess(
        index=df.index,
        constant=True,
        order=1,
        seasonal=False,
        period=48,
        additional_terms=[daily_fourier, yearly_fourier],
        drop=True
    )
    dp_feats = dp.in_sample().add_prefix("dp_")
    df = df.join(dp_feats)
    df = df.dropna()
    return df, dp