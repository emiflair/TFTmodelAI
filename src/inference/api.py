"""Inference helpers for XAUUSD TFT model."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pytorch_forecasting as pf
from pytorch_forecasting import TemporalFusionTransformer

from ..config import DEFAULT_CONFIG, ProjectConfig
from ..features.features_15m import add_15m_features
from ..features.features_1h import attach_completed_hourly_context, build_hourly_context

logger = logging.getLogger(__name__)
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
    # Check that each timestamp is aligned to 15-minute boundary (0, 15, 30, 45 minutes)
    # Don't check intervals since market data can have weekend/holiday gaps
    minutes = df["timestamp"].dt.minute
    if not (minutes.isin([0, 15, 30, 45])).all():
        return "Timestamps must be aligned to 15-minute boundaries (0, 15, 30, 45 minutes)"

    if len(df) < config.data.lookback_bars:
        return f"Need at least {config.data.lookback_bars} rows"

    return None


def _prepare_features(df: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    features, _ = add_15m_features(df)
    hourly_ctx, _ = build_hourly_context(features)
    features, _ = attach_completed_hourly_context(features, hourly_ctx)
    features = features.dropna().reset_index(drop=True)
    features = add_time_index(features, freq_minutes=15)
    # Use dummy target for prediction (will not be used, but dataset requires non-NaN)
    features["target"] = 0.0
    features["series_id"] = "XAUUSD"
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
    feature_df["series_id"] = "XAUUSD"

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

    # Simplified prediction approach - use most recent continuous sequence
    try:
        # Get dataset parameters from model
        params = model.dataset_parameters
        
        # Get the most recent N rows where N = encoder_length + prediction_length
        max_encoder = params.get("max_encoder_length", 128)
        max_pred = params.get("max_prediction_length", 1)
        required_length = max_encoder + max_pred
        
        # Take the last required_length rows
        if len(feature_df) < required_length:
            return {
                "q10": None, 
                "q50": None, 
                "q90": None, 
                "ok": False, 
                "reason": f"Need {required_length} rows, have {len(feature_df)}"
            }
        
        # Get most recent sequence
        recent_df = feature_df.tail(required_length).copy()
        recent_df = recent_df.reset_index(drop=True)
        
        # Debug: Check feature statistics
        logger.info(f"Feature columns ({len(recent_df.columns)}): {recent_df.columns.tolist()[:15]}...")  # First 15
        logger.info(f"Last 3 rows of 'close': {recent_df['close'].tail(3).tolist()}")
        logger.info(f"Last 3 rows of 'rsi14': {recent_df['rsi14'].tail(3).tolist() if 'rsi14' in recent_df.columns else 'N/A'}")
        logger.info(f"NaN count per column (top 5): {recent_df.isna().sum().nlargest(5).to_dict()}")
        logger.info(f"Feature value ranges - close: [{recent_df['close'].min():.2f}, {recent_df['close'].max():.2f}]")
        
        # Log all expected features
        time_varying_unknown = params.get('time_varying_unknown_reals', [])
        time_varying_known = params.get('time_varying_known_reals', [])
        logger.info(f"Model expects {len(time_varying_unknown)} time_varying_unknown_reals: {time_varying_unknown}")
        logger.info(f"Model expects {len(time_varying_known)} time_varying_known_reals: {time_varying_known}")
        
        # Check for missing features
        missing = [f for f in time_varying_unknown if f not in recent_df.columns]
        if missing:
            logger.warning(f"Missing expected features: {missing}")
        
        # Reset time_idx to be continuous (0, 1, 2, ...)
        time_idx_col = params.get("time_idx", "time_idx")
        recent_df[time_idx_col] = range(len(recent_df))
        
        # Create dataset with continuous time index
        dataset = pf.TimeSeriesDataSet(
            data=recent_df,
            time_idx=time_idx_col,
            target=params.get("target", "target"),
            group_ids=params.get("group_ids", ["series_id"]),
            max_encoder_length=max_encoder,
            max_prediction_length=max_pred,
            time_varying_known_reals=params.get("time_varying_known_reals", []),
            time_varying_known_categoricals=params.get("time_varying_known_categoricals", []),
            time_varying_unknown_reals=params.get("time_varying_unknown_reals", []),
            time_varying_unknown_categoricals=params.get("time_varying_unknown_categoricals", []),
            static_reals=params.get("static_reals", []),
            static_categoricals=params.get("static_categoricals", []),
            add_relative_time_idx=params.get("add_relative_time_idx", True),
            add_target_scales=params.get("add_target_scales", True),
            add_encoder_length=params.get("add_encoder_length", True),
            min_encoder_length=max_encoder,  # Ensure we use full encoder
            min_prediction_length=max_pred,
        )
        
        # Create dataloader
        dataloader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
        
        # Make predictions
        predictions = model.predict(
            dataloader,
            mode="quantiles",
            return_x=False,
        )
        
        # Debug: Log raw prediction tensor
        logger.info(f"Raw prediction tensor shape: {predictions.shape}")
        logger.info(f"Raw prediction tensor: {predictions}")
        logger.info(f"Prediction stats - min: {predictions.min():.6f}, max: {predictions.max():.6f}, mean: {predictions.mean():.6f}")
        
    except Exception as exc:  # pragma: no cover - defensive guard
        import traceback
        error_detail = traceback.format_exc()
        return {"q10": None, "q50": None, "q90": None, "ok": False, "reason": f"Prediction error: {str(exc)[:150]}"}

    # Convert to numpy
    predictions = predictions.detach().cpu().numpy()
    
    # Handle different output shapes
    # Expected shape: (batch_size, prediction_length, n_quantiles) or (batch_size, n_quantiles)
    if predictions.ndim == 3:
        # Take last prediction step
        predictions = predictions[:, -1, :]  # Shape: (batch_size, n_quantiles)
    elif predictions.ndim == 2:
        # Already in correct shape (batch_size, n_quantiles)
        pass
    else:
        return {"q10": None, "q50": None, "q90": None, "ok": False, "reason": f"Unexpected prediction shape: {predictions.shape}"}
    
    # Take first (and only) batch
    if len(predictions) > 0:
        pred_values = predictions[0]  # Shape: (n_quantiles,)
    else:
        return {"q10": None, "q50": None, "q90": None, "ok": False, "reason": "Empty predictions"}
    
    # Map quantiles to values
    # If we have exactly 3 values, assume they are [q10, q50, q90]
    if len(pred_values) == 3:
        quantile_map = {0.1: pred_values[0], 0.5: pred_values[1], 0.9: pred_values[2]}
    elif len(pred_values) == len(requested_quantiles):
        quantile_map = {q: pred_values[i] for i, q in enumerate(requested_quantiles)}
    else:
        # Try to map by requested quantiles length
        quantile_map = {q: pred_values[min(i, len(pred_values)-1)] for i, q in enumerate(requested_quantiles)}

    # Convert returns to actual prices
    # Model predicts returns (percentage changes), convert to prices
    # Formula: future_price = current_price * (1 + return)
    current_price = df["close"].iloc[-1]
    
    q10_return = quantile_map.get(0.1, 0.0)
    q50_return = quantile_map.get(0.5, 0.0)
    q90_return = quantile_map.get(0.9, 0.0)
    
    # Debug logging
    logger.info(f"Raw model outputs (returns): q10={q10_return:.6f}, q50={q50_return:.6f}, q90={q90_return:.6f}")
    logger.info(f"Current price: ${current_price:.2f}")
    
    # Convert to prices
    q10_price = current_price * (1 + q10_return)
    q50_price = current_price * (1 + q50_return)
    q90_price = current_price * (1 + q90_return)

    return {
        "q10": q10_price,
        "q50": q50_price,
        "q90": q90_price,
        "ok": True,
        "reason": None,
    }
