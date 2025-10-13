"""Compute 15-minute feature set for TFT model input."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def add_15m_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    enriched, new_cols = _compute_15m_features(df.copy())
    truncated, _ = _compute_15m_features(df.iloc[:-1].copy())
    for col in new_cols:
        if not enriched.iloc[:-1][col].equals(truncated[col]):
            raise AssertionError(f"Look-ahead detected in feature {col}")
    return enriched, new_cols


def _compute_15m_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    new_cols: List[str] = []

    df["return_1"] = df["close"].pct_change()
    df["true_range"] = _true_range(df)
    df["atr14"] = df["true_range"].rolling(window=14, min_periods=1).mean()
    df["atr14_norm"] = df["atr14"] / df["close"].replace(0, np.nan)
    atr_mean = df["atr14"].rolling(window=128, min_periods=32).mean()
    atr_std = df["atr14"].rolling(window=128, min_periods=32).std().replace(0, np.nan)
    df["atr14_z"] = (df["atr14"] - atr_mean) / atr_std
    new_cols.extend(["return_1", "true_range", "atr14", "atr14_norm", "atr14_z"])

    df["rsi14"] = _rsi(df["close"], period=14)
    df["roc5"] = df["close"].pct_change(periods=5)
    df["roc10"] = df["close"].pct_change(periods=10)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["ema20_slope"] = df["ema20"].diff()
    df["dist_close_ema20"] = (df["close"] - df["ema20"]) / df["atr14"].replace(0, np.nan)
    df["dist_close_ema50"] = (df["close"] - df["ema50"]) / df["atr14"].replace(0, np.nan)
    new_cols.extend([
        "rsi14",
        "roc5",
        "roc10",
        "ema20",
        "ema50",
        "ema200",
        "ema20_slope",
        "dist_close_ema20",
        "dist_close_ema50",
    ])

    df["stdev20"] = df["return_1"].rolling(window=20, min_periods=1).std()
    bb_mid = df["close"].rolling(window=20, min_periods=1).mean()
    bb_std = df["close"].rolling(window=20, min_periods=1).std()
    df["bb_width20"] = (bb_std * 4) / bb_mid.replace(0, np.nan)
    df["compression_ratio"] = (df["high"] - df["low"]) / df["atr14"].replace(0, np.nan)
    new_cols.extend(["stdev20", "bb_width20", "compression_ratio"])

    df["rolling_high20"] = df["high"].rolling(window=20, min_periods=1).max()
    df["rolling_low20"] = df["low"].rolling(window=20, min_periods=1).min()
    df["rolling_high50"] = df["high"].rolling(window=50, min_periods=1).max()
    df["rolling_low50"] = df["low"].rolling(window=50, min_periods=1).min()
    nearest_high = df[["rolling_high20", "rolling_high50"]].min(axis=1)
    nearest_low = df[["rolling_low20", "rolling_low50"]].max(axis=1)
    df["dist_to_nearest_extreme"] = np.minimum(
        np.abs(df["close"] - nearest_high),
        np.abs(df["close"] - nearest_low),
    ) / df["atr14"].replace(0, np.nan)
    new_cols.extend([
        "rolling_high20",
        "rolling_low20",
        "rolling_high50",
        "rolling_low50",
        "dist_to_nearest_extreme",
    ])

    df["mean_tick_volume20"] = df["tick_volume"].rolling(window=20, min_periods=1).mean()
    df["vol_per_atr"] = df["tick_volume"] / df["atr14"].replace(0, np.nan)
    new_cols.extend(["tick_volume", "mean_tick_volume20", "vol_per_atr"])

    hours = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    dow = df["timestamp"].dt.weekday
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    session_cols = _session_one_hot(hours)
    for name, values in session_cols.items():
        df[name] = values
    new_cols.extend([
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        *session_cols.keys(),
    ])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df, new_cols


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


def _session_one_hot(hours: pd.Series) -> dict[str, pd.Series]:
    asia = ((hours >= 0) & (hours < 8)).astype(int)
    london = ((hours >= 8) & (hours < 16)).astype(int)
    ny = ((hours >= 13) & (hours < 22)).astype(int)
    return {
        "session_asia": asia,
        "session_london": london,
        "session_ny": ny,
    }
