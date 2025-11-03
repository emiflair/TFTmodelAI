"""Data ingestion and cleaning utilities for the XAUUSD 15-minute dataset."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import DataConfig


@dataclass
class DataLoadSummary:
    initial_rows: int
    final_rows: int
    gaps_forward_filled: int
    gaps_dropped: int
    winsorized_columns: Tuple[str, ...]


def load_raw_csv(path: str, tz: str = "UTC") -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [col.strip().lower() for col in df.columns]

    column_map = {
        "time (utc)": "timestamp",
        "time": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "tick_volume",
        "volume (tick)": "tick_volume",
    }
    df.rename(columns={col: column_map.get(col, col) for col in df.columns}, inplace=True)

    if "timestamp" not in df.columns:
        raise ValueError("CSV must contain a timestamp column")

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(str).str.strip(),
                                      format="%Y.%m.%d %H:%M:%S",
                                      errors="coerce",
                                      utc=True)

    numeric_cols = [
        col
        for col in ("open", "high", "low", "close", "tick_volume")
        if col in df.columns
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["timestamp", *numeric_cols], inplace=True)

    df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def remove_weekends(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["timestamp"].dt.weekday < 5
    return df.loc[mask].copy()


def enforce_strict_grid(df: pd.DataFrame, freq: str) -> Tuple[pd.DataFrame, int, int]:
    df = df.set_index("timestamp").sort_index()
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz=df.index.tz)
    df = df.reindex(full_index)
    gaps = df.isna().any(axis=1)
    return df.reset_index().rename(columns={"index": "timestamp"}), int(gaps.sum()), int(len(full_index))


def forward_fill_small_gaps(df: pd.DataFrame, max_gap: int) -> Tuple[pd.DataFrame, int]:
    is_nan = df.isna().any(axis=1)
    gap_sizes = _compute_gap_lengths(is_nan)
    forward_fill_count = 0
    for start, length in gap_sizes:
        if length <= max_gap:
            df.iloc[start : start + length] = df.iloc[start - 1]
            forward_fill_count += length
    return df, forward_fill_count


def drop_large_gap_blocks(df: pd.DataFrame, max_gap: int) -> Tuple[pd.DataFrame, int]:
    is_nan = df.isna().any(axis=1)
    gap_sizes = _compute_gap_lengths(is_nan)
    drop_count = 0
    drop_idx = []
    for start, length in gap_sizes:
        if length > max_gap:
            drop_idx.extend(range(start, start + length))
            drop_count += length
    if drop_idx:
        df = df.drop(index=drop_idx)
    df.reset_index(drop=True, inplace=True)
    return df, drop_count


def _compute_gap_lengths(is_nan: pd.Series) -> Tuple[Tuple[int, int], ...]:
    gaps = []
    in_gap = False
    start = 0
    for idx, flag in enumerate(is_nan.values):
        if flag and not in_gap:
            in_gap = True
            start = idx
        elif not flag and in_gap:
            gaps.append((start, idx - start))
            in_gap = False
    if in_gap:
        gaps.append((start, len(is_nan) - start))
    return tuple(gaps)


def winsorize_columns(df: pd.DataFrame, columns: Tuple[str, ...], pct: float) -> None:
    for col in columns:
        lower = df[col].quantile(pct)
        upper = df[col].quantile(1 - pct)
        df[col] = df[col].clip(lower, upper)


def compute_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    if {"bid", "ask"}.issubset(df.columns):
        df["mid"] = (df["bid"] + df["ask"]) / 2
    if "spread" in df.columns:
        median = df["spread"].median()
        iqr = df["spread"].quantile(0.75) - df["spread"].quantile(0.25)
        if iqr == 0:
            iqr = 1e-8
        df["spread_z"] = (df["spread"] - median) / iqr
    else:
        df["spread_z"] = 0.0
        df["spread"] = 0.0
    return df


def load_and_clean_data(config: DataConfig) -> Tuple[pd.DataFrame, DataLoadSummary]:
    df = load_raw_csv(str(config.csv_path), tz=config.timezone)
    df = remove_weekends(df)
    df, _, _ = enforce_strict_grid(df, config.frequency)
    df, ffilled = forward_fill_small_gaps(df, config.max_forward_fill_bars)
    df, dropped = drop_large_gap_blocks(df, config.drop_threshold_bars - 1)
    df.dropna(inplace=True)

    df = compute_spread_features(df)

    df["return_1"] = df["close"].pct_change()
    if "high" in df.columns and "low" in df.columns and "close" in df.columns:
        df["atr_proxy"] = (df["high"] - df["low"]).rolling(window=14, min_periods=1).mean()
    winsorize_columns(df, ("return_1", "atr_proxy"), config.winsorize_pct)

    df.drop(columns=["return_1", "atr_proxy"], inplace=True)
    summary = DataLoadSummary(
        initial_rows=len(df),
        final_rows=len(df),
        gaps_forward_filled=ffilled,
        gaps_dropped=dropped,
        winsorized_columns=("return_1", "atr_proxy"),
    )
    return df, summary
