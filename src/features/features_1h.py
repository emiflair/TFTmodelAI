"""Hourly context feature engineering utilities."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def build_hourly_context(df_15m: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    hourly, new_cols = _compute_hourly_features(df_15m.copy())
    truncated, _ = _compute_hourly_features(df_15m.iloc[:-4].copy())
    shared_index = truncated.index.intersection(hourly.index[:-1])
    for col in new_cols:
        if not hourly.loc[shared_index, col].equals(truncated.loc[shared_index, col]):
            raise AssertionError(f"Look-ahead detected in hourly feature {col}")
    return hourly, new_cols


def attach_completed_hourly_context(df_15m: pd.DataFrame, hourly_context: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df_15m.copy()
    context_hour = df["timestamp"].dt.floor("h") - pd.Timedelta(hours=1)
    df["context_hour"] = context_hour
    merged = df.merge(hourly_context, left_on="context_hour", right_index=True, how="left", suffixes=("", ""))
    new_cols = [col for col in hourly_context.columns if col not in df_15m.columns]
    return merged.drop(columns=["context_hour"]), new_cols


def _compute_hourly_features(df_15m: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    if "timestamp" not in df_15m.columns:
        raise ValueError("DataFrame must include timestamp column")
    df = df_15m.copy()
    df["hour"] = df["timestamp"].dt.floor("h")
    counts = df.groupby("hour")["timestamp"].count()
    full_hours = counts[counts >= 4].index
    df = df[df["hour"].isin(full_hours)]

    hourly = df.groupby("hour").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        tick_volume=("tick_volume", "sum"),
        spread=("spread", "mean"),
    )

    hourly["true_range_1h"] = _true_range(hourly)
    hourly["atr14_1h"] = hourly["true_range_1h"].rolling(window=14, min_periods=1).mean()
    hourly["ema20_1h"] = hourly["close"].ewm(span=20, adjust=False).mean()
    hourly["ema50_1h"] = hourly["close"].ewm(span=50, adjust=False).mean()
    hourly["ema50_slope_1h"] = hourly["ema50_1h"].diff()
    hourly["bool_ema20_gt_ema50_1h"] = (hourly["ema20_1h"] > hourly["ema50_1h"]).astype(int)
    hourly["rsi14_1h"] = _rsi(hourly["close"], period=14)
    hourly["bb_width_1h"] = _bollinger_width(hourly["close"], window=20)
    hourly["dist_close_ema20_1h"] = (hourly["close"] - hourly["ema20_1h"]) / hourly["atr14_1h"].replace(0, np.nan)
    atr_mean = hourly["atr14_1h"].rolling(window=100, min_periods=20).mean()
    atr_std = hourly["atr14_1h"].rolling(window=100, min_periods=20).std().replace(0, np.nan)
    hourly["vol_z_1h"] = (hourly["atr14_1h"] - atr_mean) / atr_std

    rolling_high_24h = hourly["high"].rolling(window=24, min_periods=1).max()
    rolling_low_24h = hourly["low"].rolling(window=24, min_periods=1).min()
    hourly["dist_to_24h_high_1h"] = (rolling_high_24h - hourly["close"]) / hourly["atr14_1h"].replace(0, np.nan)
    hourly["dist_to_24h_low_1h"] = (hourly["close"] - rolling_low_24h) / hourly["atr14_1h"].replace(0, np.nan)

    hourly.replace([np.inf, -np.inf], np.nan, inplace=True)
    base_cols = ["open", "high", "low", "close", "tick_volume", "spread"]
    hourly_context = hourly.drop(columns=base_cols)

    return hourly_context, [
        "true_range_1h",
        "atr14_1h",
        "ema20_1h",
        "ema50_1h",
        "ema50_slope_1h",
        "bool_ema20_gt_ema50_1h",
        "rsi14_1h",
        "bb_width_1h",
        "dist_close_ema20_1h",
        "vol_z_1h",
        "dist_to_24h_high_1h",
        "dist_to_24h_low_1h",
    ]


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=period, adjust=False).mean()
    roll_down = down.ewm(span=period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _bollinger_width(series: pd.Series, window: int) -> pd.Series:
    ma = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std()
    return (std * 4) / ma.replace(0, np.nan)
