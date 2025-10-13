"""High-level data preparation pipeline for the TFT model."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .config import DEFAULT_CONFIG, ProjectConfig, ensure_artifact_dirs
from .data_loader import DataLoadSummary, load_and_clean_data
from .features.features_15m import add_15m_features
from .features.features_1h import attach_completed_hourly_context, build_hourly_context
from .preprocessing.scalers import RobustScalerStore
from .preprocessing.splits import SplitWindow, generate_walk_forward_splits
from .preprocessing.targets import add_future_return
from .preprocessing.time_index import add_time_index


def prepare_base_dataframe(config: ProjectConfig = DEFAULT_CONFIG) -> Tuple[pd.DataFrame, DataLoadSummary, Dict[str, List[str]]]:
    df, summary = load_and_clean_data(config.data)

    # Use simplified features to avoid pandas issues
    try:
        from .features.simplified_features_15m import add_15m_features as add_simplified_15m_features
        df_15m, fifteen_cols = add_simplified_15m_features(df)
        print("✅ Using simplified enhanced features (more reliable)")
    except Exception as e:
        print(f"⚠️  Fallback to original features due to: {e}")
        df_15m, fifteen_cols = add_15m_features(df)
    hourly_context, hourly_cols = build_hourly_context(df_15m)
    df_full, hourly_attached_cols = attach_completed_hourly_context(df_15m, hourly_context)

    feature_manifest: Dict[str, List[str]] = {
        "fifteen_min_features": fifteen_cols,
        "hourly_features": hourly_cols,
        "hourly_attached": hourly_attached_cols,
    }

    if "spread_z" in df_full.columns:
        df_full.loc[df_full["spread_z"].abs() > 5, ["open", "high", "low", "close", "tick_volume"]] = pd.NA

    df_full.dropna(inplace=True)
    df_full.reset_index(drop=True, inplace=True)

    return df_full, summary, feature_manifest


def build_training_frames(config: ProjectConfig = DEFAULT_CONFIG):
    ensure_artifact_dirs(config.artifacts)
    df, summary, feature_manifest = prepare_base_dataframe(config)

    df, target_col = add_future_return(df, config.data.horizon_bars)
    df = add_time_index(df, freq_minutes=15)

    df = df[df["timestamp"] <= df["timestamp"].max() - pd.Timedelta(minutes=config.data.horizon_bars * 15)]
    df.dropna(inplace=True)

    splits = generate_walk_forward_splits(df, config.windows)

    feature_columns = [
        col
        for col in df.columns
        if col
        not in {
            "timestamp",
            "series_id",
            "time_idx",
            "target",
            target_col,
        }
        and not col.startswith("future_")
    ]

    df.rename(columns={target_col: "target"}, inplace=True)

    scale_columns = [
        col
        for col in feature_columns
        if not col.startswith("session_")
        and not col.startswith("bool_")
        and col not in {"hour_sin", "hour_cos", "dow_sin", "dow_cos"}
    ]

    return df, splits, feature_columns, scale_columns, feature_manifest, summary


def fit_scaler_for_split(train_df: pd.DataFrame, feature_columns: List[str], scaler_path: Path) -> RobustScalerStore:
    scaler = RobustScalerStore()
    scaler.fit(train_df, feature_columns)
    scaler.save(scaler_path)
    return scaler


def transform_split_frames(
    df: pd.DataFrame,
    split: SplitWindow,
    scaler: RobustScalerStore,
    feature_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_mask = (df["timestamp"] >= split.train_start) & (df["timestamp"] < split.train_end)
    val_mask = (df["timestamp"] >= split.train_end) & (df["timestamp"] < split.val_end)
    test_mask = (df["timestamp"] >= split.val_end) & (df["timestamp"] < split.test_end)

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()
    test_df = df.loc[test_mask].copy()

    for frame in (train_df, val_df, test_df):
        scaler.transform_inplace(frame)

    return train_df, val_df, test_df
