"""Inference helpers for EURUSD TFT model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pytorch_forecasting as pf
from pytorch_forecasting import TemporalFusionTransformer

from ..config import DEFAULT_CONFIG, ProjectConfig
from ..features.features_15m import add_15m_features
from ..features.features_1h import attach_completed_hourly_context, build_hourly_context
from ..preprocessing.scalers import RobustScalerStore
from ..preprocessing.time_index import add_time_index
from ..utils.seeding import set_global_seeds


def _load_manifest(path: Path) -> Dict[str, List[str]]:
    payload = json.loads(path.read_text())
    required = ["feature_columns", "scaled_columns", "known_future_reals", "unknown_reals"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Manifest missing keys: {missing}")
    return payload


def _validate_input(df: pd.DataFrame, config: ProjectConfig) -> Optional[str]:
    required_cols = {"timestamp", "open", "high", "low", "close", "tick_volume"}
    missing = required_cols.difference(df.columns)
    if missing:
        return f"Missing columns: {sorted(missing)}"

    df = df.sort_values("timestamp")
    deltas = df["timestamp"].diff().dropna().dt.total_seconds() / 60
    if not np.allclose(deltas, 15, atol=1e-6):
        return "Timestamps must be aligned to 15-minute grid"

    if len(df) < config.data.lookback_bars:
        return f"Need at least {config.data.lookback_bars} rows"

    return None


def _prepare_features(df: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    features, _ = add_15m_features(df)
    hourly_ctx, _ = build_hourly_context(features)
    features, _ = attach_completed_hourly_context(features, hourly_ctx)
    features = features.dropna().reset_index(drop=True)
    features = add_time_index(features, freq_minutes=15)
    features["target"] = np.nan
    features["series_id"] = "EURUSD"
    return features


def _guardrails(feature_df: pd.DataFrame) -> Optional[str]:
    latest = feature_df.iloc[-1]
    if "spread_z" in feature_df.columns and abs(latest["spread_z"]) > 5:
        return "Spread z-score exceeds threshold"
    if "atr14_z" in feature_df.columns and abs(latest["atr14_z"]) > 5:
        return "ATR z-score exceeds threshold"
    return None


def predict_r3_quantiles(
    latest_bars_df: pd.DataFrame,
    checkpoint_path: Path,
    scaler_path: Path,
    manifest_path: Path,
    quantiles: Optional[Sequence[float]] = None,
    config: ProjectConfig = DEFAULT_CONFIG,
) -> Dict[str, Optional[float]]:
    """Return quantile predictions for the next 3-bar return."""

    df = latest_bars_df.copy()
    if "Time (UTC)" in df.columns and "timestamp" not in df.columns:
        df.rename(columns={"Time (UTC)": "timestamp"}, inplace=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(config.data.timezone)
    df.sort_values("timestamp", inplace=True)

    validation_error = _validate_input(df, config)
    if validation_error:
        return {"q10": None, "q50": None, "q90": None, "ok": False, "reason": validation_error}

    try:
        manifest = _load_manifest(manifest_path)
    except Exception as exc:  # pragma: no cover - defensive guard
        return {"q10": None, "q50": None, "q90": None, "ok": False, "reason": f"Manifest error: {exc}"}

    set_global_seeds(config.seed.global_seed)

    feature_df = _prepare_features(df, config)
    if feature_df.empty:
        return {"q10": None, "q50": None, "q90": None, "ok": False, "reason": "Insufficient feature rows"}

    feature_columns = manifest["feature_columns"]
    missing_features = [col for col in feature_columns if col not in feature_df.columns]
    if missing_features:
        return {
            "q10": None,
            "q50": None,
            "q90": None,
            "ok": False,
            "reason": f"Missing expected features: {sorted(missing_features)}",
        }

    scaler = RobustScalerStore.load(scaler_path)
    missing_scaled = [col for col in scaler.feature_list() if col not in feature_df.columns]
    if missing_scaled:
        return {
            "q10": None,
            "q50": None,
            "q90": None,
            "ok": False,
            "reason": f"Missing scaled columns: {sorted(missing_scaled)}",
        }
    scaler.transform_inplace(feature_df)

    try:
        model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path, map_location="cpu")
        model.freeze()
    except Exception as exc:  # pragma: no cover - defensive guard
        return {"q10": None, "q50": None, "q90": None, "ok": False, "reason": f"Model load error: {exc}"}

    dataset_params = getattr(model, "dataset_parameters", {}) or {}
    if not dataset_params:
        return {
            "q10": None,
            "q50": None,
            "q90": None,
            "ok": False,
            "reason": "Checkpoint missing dataset parameters for inference",
        }

    time_idx_field = dataset_params.get("time_idx", "time_idx")
    if time_idx_field not in feature_df.columns:
        return {
            "q10": None,
            "q50": None,
            "q90": None,
            "ok": False,
            "reason": f"Missing required time index column '{time_idx_field}'",
        }

    max_encoder_length = dataset_params.get("max_encoder_length") or model.hparams.get("max_encoder_length")
    max_prediction_length = dataset_params.get("max_prediction_length") or model.hparams.get("max_prediction_length")
    if max_encoder_length is None:
        max_encoder_length = max(config.data.lookback_bars - config.data.horizon_bars, 1)
    if max_prediction_length is None:
        max_prediction_length = max(config.data.horizon_bars, 1)

    total_required = int(max_encoder_length + max_prediction_length)
    if len(feature_df) < total_required:
        return {
            "q10": None,
            "q50": None,
            "q90": None,
            "ok": False,
            "reason": f"Need at least {total_required} feature rows for inference",
        }

    feature_df = feature_df.iloc[-total_required:].copy()
    feature_df.sort_values(time_idx_field, inplace=True)
    feature_df.reset_index(drop=True, inplace=True)
    feature_df["series_id"] = "EURUSD"

    guardrail_reason = _guardrails(feature_df)
    if guardrail_reason:
        return {"q10": None, "q50": None, "q90": None, "ok": False, "reason": guardrail_reason}

    expected_real_cols = set(dataset_params.get("time_varying_known_reals", [])) | set(
        dataset_params.get("time_varying_unknown_reals", [])
    )
    expected_real_cols.add(time_idx_field)
    missing_dataset_cols = sorted(expected_real_cols.difference(feature_df.columns))
    if missing_dataset_cols:
        return {
            "q10": None,
            "q50": None,
            "q90": None,
            "ok": False,
            "reason": f"Missing expected dataset columns: {missing_dataset_cols}",
        }

    requested_quantiles = list(quantiles or config.quantiles.quantiles)

    dataset = pf.TimeSeriesDataSet.from_dataset_parameters(
        model.dataset_parameters,
        feature_df,
        predict=True,
        stop_randomization=True,
    )
    dataloader = dataset.to_dataloader(train=False, batch_size=1)

    try:
        predictions = model.predict(
            dataloader,
            mode="quantiles",
            quantiles=requested_quantiles,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        return {"q10": None, "q50": None, "q90": None, "ok": False, "reason": f"Prediction error: {exc}"}

    predictions = predictions.detach().cpu().numpy()
    if predictions.ndim == 3:
        predictions = predictions[:, :, 0]

    latest_idx = -1
    quantile_map = {q: predictions[latest_idx, i] for i, q in enumerate(requested_quantiles)}

    return {
        "q10": quantile_map.get(0.1),
        "q50": quantile_map.get(0.5),
        "q90": quantile_map.get(0.9),
        "ok": True,
        "reason": None,
    }
