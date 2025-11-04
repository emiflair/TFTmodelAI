"""Training orchestration for the XAUUSD Temporal Fusion Transformer.

Adds CLI flags for Colab usage and long training sessions:
    - --resume {none,last,<path>}  Resume from last/explicit checkpoint
    - --max-epochs INT             Override max epochs
    - --batch-size INT             Override batch size
    - --hidden-size INT            Override model size
    - --learning-rate FLOAT        Override learning rate
    - --splits INT                 Limit number of walk-forward splits
    - --fast-dev-run               Enable faster dev run (limits epochs/splits)
    - --train-months/--val-months/--test-months/--stride-months
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import argparse

import numpy as np
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_forecasting as pf
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for training") from exc

from ..config import DEFAULT_CONFIG, ProjectConfig, ensure_artifact_dirs
from ..evaluation.metrics import (
    band_coverage,
    crps_from_quantiles,
    directional_hit_rate,
    pinball_loss,
    simple_profit_factor,
)
from ..evaluation.enhanced_metrics import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_hit_rate_by_regime,
    calculate_prediction_consistency,
    calculate_risk_adjusted_returns,
    trading_performance_summary,
)
from ..pipeline import build_training_frames
from ..preprocessing.scalers import RobustScalerStore
from ..preprocessing.enhanced_scalers import EnhancedScalerStore, detect_feature_types
from ..utils.seeding import set_global_seeds


def _version_metadata() -> Dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "pytorch_lightning": pl.__version__,
    }
    try:
        versions["pytorch_forecasting"] = pf.__version__
    except Exception:  # pragma: no cover
        versions["pytorch_forecasting"] = "unknown"
    try:
        from subprocess import check_output

        versions["git_commit"] = (
            check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2])
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover
        versions["git_commit"] = "not_available"
    return versions


def _split_frames(df, split) -> Tuple:
    train_mask = (df["timestamp"] >= split.train_start) & (df["timestamp"] < split.train_end)
    val_mask = (df["timestamp"] >= split.train_end) & (df["timestamp"] < split.val_end)
    test_mask = (df["timestamp"] >= split.val_end) & (df["timestamp"] < split.test_end)
    return (
        df.loc[train_mask].copy(),
        df.loc[val_mask].copy(),
        df.loc[test_mask].copy(),
    )


def _determine_feature_groups(feature_columns: Sequence[str]) -> Tuple[List[str], List[str]]:
    known_future = [
        col
        for col in feature_columns
        if col.startswith("hour_") or col.startswith("dow_") or col.startswith("session_")
    ]
    unknown = [col for col in feature_columns if col not in known_future]
    return known_future, unknown


def _build_dataloaders(training_ds, validation_ds, batch_size: int) -> Tuple:
    return (
        training_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=0),
        validation_ds.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0),
    )


def _session_label(row) -> str:
    if row.get("session_ny", 0) == 1:
        return "NY"
    if row.get("session_london", 0) == 1:
        return "London"
    if row.get("session_asia", 0) == 1:
        return "Asia"
    return "Other"


def _atr_regime(value: float) -> str:
    if np.isnan(value):
        return "unknown"
    if value <= -1:
        return "low"
    if value < 1:
        return "normal"
    return "high"


def _group_metrics(df: pd.DataFrame, lower_name: str, upper_name: str) -> Dict[str, Dict[str, float]]:
    groups = {}
    for name, group in df.groupby("session_label"):
        if group.empty:
            continue
        groups[f"session_{name}"] = {
            "coverage": band_coverage(
                group["target"].values,
                group[lower_name].values,
                group[upper_name].values,
            ),
            "directional_hit": directional_hit_rate(group["target"].values, group["q50"].values),
        }
    for name, group in df.groupby("atr_regime"):
        if group.empty:
            continue
        groups[f"regime_{name}"] = {
            "coverage": band_coverage(
                group["target"].values,
                group[lower_name].values,
                group[upper_name].values,
            ),
            "directional_hit": directional_hit_rate(group["target"].values, group["q50"].values),
        }
    return groups


def _refresh_symlink(link: Path, target_name: str) -> None:
    if link.exists() or link.is_symlink():
        try:
            link.unlink()
        except FileNotFoundError:  # pragma: no cover
            pass
    link.symlink_to(target_name)


def _json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {key: _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(value) for value in obj]
    return obj


def _effective_training_params(config: ProjectConfig) -> Dict[str, int]:
    params = {
        "max_epochs": config.training.max_epochs,
        "batch_size": config.training.batch_size,
        "max_splits": None,
    }
    if config.training.fast_dev_run:
        params["max_epochs"] = min(config.training.max_epochs, config.training.fast_max_epochs)
        if config.training.fast_batch_size:
            params["batch_size"] = config.training.fast_batch_size
        params["max_splits"] = max(1, config.training.fast_max_splits)
    return params

def _select_precision(mixed_precision: bool):
    """Prefer bf16 mixed precision if supported; otherwise fall back to 32-bit.

    This avoids fp16 (half) overflows seen in masked_fill/attention on some GPUs.
    """
    if not mixed_precision:
        return 32
    try:
        if torch.cuda.is_available():
            # Prefer bf16 when available (A100/L4, some recent GPUs)
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return "bf16-mixed"
    except Exception:
        pass
    # Safe fallback (disable fp16 which can overflow in attention masks)
    return 32


def _create_enhanced_loss(base_loss, config: ProjectConfig):
    """Create enhanced loss function combining quantile loss with directional accuracy."""
    
    class EnhancedQuantileLoss:
        def __init__(self, base_quantile_loss, directional_weight=0.1):
            self.base_loss = base_quantile_loss
            self.directional_weight = directional_weight
            
        def __call__(self, predictions, target):
            # Base quantile loss
            quantile_loss = self.base_loss(predictions, target)
            
            # Directional loss component
            if hasattr(predictions, 'shape') and len(predictions.shape) > 1:
                # Extract median predictions (assuming 0.5 quantile is in the middle)
                n_quantiles = predictions.shape[-1]
                median_idx = n_quantiles // 2
                median_pred = predictions[..., median_idx]
                
                # Directional accuracy loss (minimize when directions match)
                target_direction = torch.sign(target)
                pred_direction = torch.sign(median_pred)
                
                # Binary cross-entropy for direction prediction
                directional_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    median_pred, 
                    (target > 0).float(),
                    reduction='mean'
                )
                
                # Combine losses
                total_loss = quantile_loss + self.directional_weight * directional_loss
                
                return total_loss
            else:
                # Fallback to base loss if shape is unexpected
                return quantile_loss
    
    return EnhancedQuantileLoss(base_loss, directional_weight=0.15)


def train_tft_model(config: ProjectConfig = DEFAULT_CONFIG) -> None:
    ensure_artifact_dirs(config.artifacts)
    set_global_seeds(config.seed.global_seed)

    runtime_params = _effective_training_params(config)
    if config.training.fast_dev_run:
        print(
            "[fast_dev_run] Limiting training to "
            f"{runtime_params['max_epochs']} epochs and {runtime_params['max_splits']} split(s)"
        )

    (
        df,
        splits,
        feature_columns,
        scale_columns,
        feature_manifest,
        load_summary,
    ) = build_training_frames(config)

    manifest_path = config.artifacts.manifests_dir / "feature_manifest.json"
    manifest_payload = {
        **feature_manifest,
        "feature_columns": feature_columns,
        "scaled_columns": scale_columns,
    }
    # known/unknown feature grouping is useful for inference wiring
    preliminary_known_future, preliminary_unknown = _determine_feature_groups(feature_columns)
    manifest_payload["known_future_reals"] = preliminary_known_future
    manifest_payload["unknown_reals"] = preliminary_unknown

    manifest_path.write_text(json.dumps(manifest_payload, indent=2))

    metadata = {
        "config": _json_safe(asdict(config)),
        "version_info": _version_metadata(),
        "data_summary": load_summary.__dict__,
        "created_at": datetime.utcnow().isoformat(),
    }

    metadata = _json_safe(metadata)

    meta_path = config.artifacts.manifests_dir / "train_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    known_future = manifest_payload["known_future_reals"]
    unknown = manifest_payload["unknown_reals"]

    quantile_list = list(config.quantiles.quantiles)
    if 0.5 not in quantile_list:
        raise ValueError("Quantile configuration must include 0.5 for median predictions")
    if (
        config.quantiles.default_band[0] not in quantile_list
        or config.quantiles.default_band[1] not in quantile_list
    ):
        raise ValueError("Default coverage band quantiles must be part of quantile list")
        precision_setting = _select_precision(config.training.mixed_precision)
    results = []

    splits_to_run = splits
    if runtime_params["max_splits"] is not None:
        splits_to_run = splits[: runtime_params["max_splits"]]

    for idx, split in enumerate(splits_to_run, start=1):
        train_df_raw, val_df_raw, test_df_raw = _split_frames(df, split)
        if train_df_raw.empty or val_df_raw.empty or test_df_raw.empty:
            continue

        # Enhanced preprocessing with intelligent feature type detection
        feature_types = detect_feature_types(train_df_raw, scale_columns)
        
        scaler = EnhancedScalerStore(default_method="robust")
        scaler.fit(train_df_raw, scale_columns, feature_types=feature_types)
        scaler_path = config.artifacts.scalers_dir / f"scaler_XAUUSD_fold{idx}.pkl"
        scaler.save(scaler_path)
        
        # Also save legacy scaler for compatibility
        legacy_scaler = RobustScalerStore()
        legacy_scaler.fit(train_df_raw, scale_columns)
        legacy_scaler_path = config.artifacts.scalers_dir / f"legacy_scaler_XAUUSD_fold{idx}.pkl"
        legacy_scaler.save(legacy_scaler_path)

        train_df = train_df_raw.copy()
        val_df = val_df_raw.copy()
        test_df = test_df_raw.copy()

        # Apply enhanced preprocessing with winsorization
        scaler.transform_inplace(train_df, winsorize=True)
        scaler.transform_inplace(val_df, winsorize=True) 
        scaler.transform_inplace(test_df, winsorize=True)

        max_encoder_length = config.data.lookback_bars
        training_ds = pf.TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target="target",
            group_ids=["series_id"],
            max_encoder_length=max_encoder_length,
            max_prediction_length=1,
            time_varying_known_reals=list(known_future),
            time_varying_unknown_reals=list(unknown),
            static_categoricals=["series_id"],
            allow_missing_timesteps=True,
            target_normalizer=None,
        )

        validation_ds = training_ds.from_dataset(training_ds, val_df, stop_randomization=True)
        test_ds = training_ds.from_dataset(training_ds, test_df, stop_randomization=True)

        train_dl, val_dl = _build_dataloaders(training_ds, validation_ds, runtime_params["batch_size"])

        checkpoint_name = f"tft_XAUUSD_15m_3B_{split.train_start:%Y%m%d}_{split.test_end:%Y%m%d}"
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.artifacts.checkpoint_dir,
            filename=checkpoint_name,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=config.training.early_stop_patience,
            mode="min",
        )

        # Use standard QuantileLoss for compatibility (enhanced loss will be added later)
        loss = QuantileLoss(quantiles=list(config.quantiles.quantiles))

        model = TemporalFusionTransformer.from_dataset(
            training_ds,
            learning_rate=config.training.learning_rate,
            hidden_size=config.training.hidden_size,
            attention_head_size=config.training.attention_head_size,
            dropout=config.training.dropout,
            loss=loss,
            weight_decay=config.training.weight_decay,
            log_interval=10,
            reduce_on_plateau_patience=4,
            # Memory-optimized architecture
            lstm_layers=2,  # Reduced from 3 to save memory
            hidden_continuous_size=config.training.hidden_size // 2,
            output_size=len(config.quantiles.quantiles),
        )

        # Choose precision robustly: prefer bf16 on Ampere+ GPUs; avoid fp16 on T4 to prevent overflow
        def _select_precision(cfg: ProjectConfig):
            try:
                if not cfg.training.mixed_precision:
                    return 32
                if torch.cuda.is_available():
                    # Prefer explicit bf16 support signal when available
                    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                        print("[precision] Using bf16 mixed precision (hardware supported)")
                        return "bf16-mixed"
                    # Fallback via capability check
                    major, minor = torch.cuda.get_device_capability(0)
                    if major >= 8:
                        print("[precision] Using bf16 mixed precision on Ampere+ GPU")
                        return "bf16-mixed"
                    # On pre-Ampere GPUs (e.g., T4, 7.5), disable AMP to avoid fp16 overflow in attention masks
                    print("[precision] Disabling AMP on this GPU to avoid fp16 overflow; using 32-bit precision")
                    return 32
            except Exception:
                pass
            return 32

        precision_setting = _select_precision(config)

        # Disable automatic prediction plotting to avoid matplotlib bfloat16 errors
        # Override the log_prediction method to prevent plotting during training/validation
        original_log_prediction = model.log_prediction
        def _no_op_log_prediction(*args, **kwargs):
            pass  # Do nothing - skip all prediction plotting
        model.log_prediction = _no_op_log_prediction

        # Enhanced trainer with memory-efficient settings for Colab
        trainer = pl.Trainer(
            max_epochs=runtime_params["max_epochs"],
            gradient_clip_val=config.training.gradient_clip_val,
            callbacks=[early_stop, checkpoint_callback],
            deterministic=config.training.deterministic,
            enable_progress_bar=True,
            enable_model_summary=True,
            precision=precision_setting,
            log_every_n_steps=config.logging.log_every_n_steps,
            # Memory-optimized settings for GPU training
            accumulate_grad_batches=1,  # Reduced from 2 to save memory
            val_check_interval=1.0,     # Validate once per epoch to reduce memory pressure
            num_sanity_val_steps=0,
            enable_checkpointing=True,
        )

        # Fit model (ckpt resume handled in CLI wrapper)
        trainer.fit(model, train_dl, val_dl)
        best_path = Path(checkpoint_callback.best_model_path)
        if not best_path.exists():
            best_path = Path(checkpoint_callback.last_model_path)

        best_model = TemporalFusionTransformer.load_from_checkpoint(best_path, map_location=torch.device("cpu"))
        best_model.eval()

        prediction_outputs = best_model.predict(
            test_ds,
            mode="quantiles",
            mode_kwargs={"quantiles": quantile_list},
            batch_size=runtime_params["batch_size"] * 2,
            return_x=True,
            return_index=True,
        )

        predictions = prediction_outputs
        prediction_x = None
        if isinstance(prediction_outputs, tuple):
            predictions = prediction_outputs[0]
            for item in prediction_outputs[1:]:
                if isinstance(item, dict) and "decoder_target" in item:
                    prediction_x = item

        if isinstance(predictions, dict):
            pred_quantiles = {
                q: np.asarray(predictions[str(q)] if str(q) in predictions else predictions[q]).reshape(-1)
                for q in quantile_list
            }
        else:
            if hasattr(predictions, "detach"):
                predictions = predictions.detach().cpu().numpy()
            else:
                predictions = np.asarray(predictions)
            if predictions.ndim == 1:
                predictions = predictions[:, np.newaxis]
            if predictions.ndim == 3:
                predictions = predictions[:, 0, :]
            if predictions.shape[1] != len(quantile_list):
                raise ValueError(
                    "Mismatch between returned quantiles and configured quantile list; "
                    f"got shape {predictions.shape}, expected {len(quantile_list)} quantiles"
                )
            pred_quantiles = {q: predictions[:, i] for i, q in enumerate(quantile_list)}

        if prediction_x is not None and "decoder_target" in prediction_x:
            decoder_target = prediction_x["decoder_target"]
            if isinstance(decoder_target, torch.Tensor):
                decoder_target = decoder_target.detach().cpu().numpy()
            decoder_target = np.asarray(decoder_target)
            y_true = decoder_target.reshape(-1)
        else:
            prediction_count = next(iter(pred_quantiles.values())).shape[0]
            y_true = test_df_raw["target"].values[-prediction_count:]
        y_true = np.asarray(y_true)
        valid_mask = ~np.isnan(y_true)
        mask_indices = np.nonzero(valid_mask)[0]
        if mask_indices.size == 0:
            continue
        y_true = y_true[valid_mask]
        pred_quantiles = {q: pred_quantiles[q][valid_mask] for q in quantile_list}

        total_predictions = len(valid_mask)
        tail_df = test_df_raw.iloc[-total_predictions:].copy()
        evaluation_df = tail_df.iloc[mask_indices].copy()
        evaluation_df.reset_index(drop=True, inplace=True)

        pinball = {
            q: pinball_loss(y_true, pred_quantiles[q], q)
            for q in quantile_list
        }
        crps = crps_from_quantiles(y_true, quantile_list, pred_quantiles)
        coverage = band_coverage(
            y_true,
            pred_quantiles[config.quantiles.default_band[0]],
            pred_quantiles[config.quantiles.default_band[1]],
        )
        hit = directional_hit_rate(y_true, pred_quantiles[0.5])

        test_df_eval = evaluation_df.copy()
        for q in quantile_list:
            test_df_eval[f"q{int(q*100):02d}"] = pred_quantiles[q]
        test_df_eval["target"] = y_true
        test_df_eval["atr_norm"] = test_df_eval.get("atr14_norm", 0.0)
        if "session_label" not in test_df_eval:
            test_df_eval["session_label"] = test_df_eval.apply(_session_label, axis=1)
        if "atr14_z" not in test_df_eval:
            test_df_eval["atr14_z"] = 0.0
        test_df_eval["atr_regime"] = test_df_eval["atr14_z"].apply(_atr_regime)

        lower_name = f"q{int(config.quantiles.default_band[0]*100):02d}"
        upper_name = f"q{int(config.quantiles.default_band[1]*100):02d}"

        group_breakdowns = _group_metrics(test_df_eval, lower_name, upper_name)

        reliability = {
            f"below_{int(q*100):02d}": float(np.mean(y_true <= pred_quantiles[q]))
            for q in quantile_list
        }

        median_name = "q50"
        pf_results = {
            "0.002": simple_profit_factor(
                test_df_eval,
                lower_name,
                median_name,
                upper_name,
                band_threshold=0.002,
                atr_norm_col="atr_norm",
                normalized=False,
            ),
            "1x_atr": simple_profit_factor(
                test_df_eval,
                lower_name,
                median_name,
                upper_name,
                band_threshold=1.0,
                atr_norm_col="atr_norm",
                normalized=True,
            ),
        }

        # Enhanced evaluation metrics
        risk_metrics = calculate_risk_adjusted_returns(test_df_eval)
        regime_hit_rates = calculate_hit_rate_by_regime(test_df_eval)
        consistency_metrics = calculate_prediction_consistency(test_df_eval)
        trading_summary = trading_performance_summary(test_df_eval, lower_name, median_name, upper_name)
        
        result_row = {
            "fold": idx,
            "split": split.to_dict(),
            "checkpoint": best_path.name,
            "pinball": pinball,
            "crps": crps,
            "coverage": coverage,
            "directional_hit": hit,
            "pf": pf_results,
            "coverage_breakdown": group_breakdowns,
            "reliability": reliability,
            # Enhanced metrics
            "risk_metrics": risk_metrics,
            "regime_performance": regime_hit_rates,
            "consistency": consistency_metrics,
            "trading_summary": {
                "sharpe_ratio": trading_summary.sharpe_ratio,
                "max_drawdown": trading_summary.max_drawdown,
                "win_rate": trading_summary.win_rate,
                "profit_factor": trading_summary.profit_factor,
                "total_return": trading_summary.total_return,
                "calmar_ratio": trading_summary.calmar_ratio,
                "sortino_ratio": trading_summary.sortino_ratio,
            },
        }
        results.append(result_row)

        metrics_path = config.artifacts.metrics_dir / f"metrics_fold{idx}.json"
        metrics_path.write_text(json.dumps(result_row, indent=2))

    if results:
        metrics_report = config.artifacts.metrics_dir / "metrics_report.json"
        metrics_report.write_text(json.dumps(results, indent=2))

        flat_rows = []
        for row in results:
            flat = {
                "fold": row["fold"],
                "checkpoint": row["checkpoint"],
                "crps": row["crps"],
                "coverage": row["coverage"],
                "directional_hit": row["directional_hit"],
            }
            for q, value in row["pinball"].items():
                flat[f"pinball_{int(q*100):02d}"] = value
            for label, value in row["pf"].items():
                flat[f"pf_{label}"] = value
            flat_rows.append(flat)
        metrics_csv = config.artifacts.metrics_dir / "metrics_report.csv"
        pd.DataFrame(flat_rows).to_csv(metrics_csv, index=False)

        data_range = (df["timestamp"].min(), df["timestamp"].max())
        model_card = config.artifacts.model_cards_dir / "model_card.txt"
        card_lines = [
            "Model: Temporal Fusion Transformer",
            "Pair: XAUUSD",
            f"Data range: {data_range[0]} — {data_range[1]}",
            f"Lookback: {config.data.lookback_bars} bars",
            f"Horizon: {config.data.horizon_bars} bars",
            f"Quantiles: {quantile_list}",
            "",
            "Fold metrics:",
        ]
        for row in results:
            trading_summary = row.get('trading_summary', {})
            card_lines.append(
                f"  Fold {row['fold']} ({row['split']['train_start']}→{row['split']['test_end']}): "
                f"CRPS={row['crps']:.4f}, Coverage={row['coverage']:.3f}, "
                f"DirectionalHit={row['directional_hit']:.3f}, PF(0.002)={row['pf'].get('0.002', float('nan')):.3f}"
            )
            card_lines.append(
                f"    Trading: Sharpe={trading_summary.get('sharpe_ratio', float('nan')):.2f}, "
                f"MaxDD={trading_summary.get('max_drawdown', float('nan')):.3f}, "
                f"WinRate={trading_summary.get('win_rate', float('nan')):.3f}, "
                f"Calmar={trading_summary.get('calmar_ratio', float('nan')):.2f}"
            )
        card_lines.extend(
            [
                "",
                "Retrain policy: rolling 6m train / 1m val / 1m test, refresh every 2–4 weeks.",
                "Early retrain triggers: coverage error > 10pp or PF < 1.05 over last 150 events.",
                f"Last training timestamp: {datetime.utcnow().isoformat()} UTC",
            ]
        )
        model_card.write_text("\n".join(card_lines))

        checkpoints = sorted(config.artifacts.checkpoint_dir.glob("tft_XAUUSD_15m_3B_*.ckpt"))
        if checkpoints:
            latest = checkpoints[-1]
            _refresh_symlink(config.artifacts.latest_symlink, latest.name)
            stable_target = config.artifacts.checkpoint_dir / results[0]["checkpoint"]
            fast_target = config.artifacts.checkpoint_dir / results[-1]["checkpoint"]
            _refresh_symlink(config.artifacts.checkpoint_dir / "tft_stable.ckpt", stable_target.name)
            _refresh_symlink(config.artifacts.checkpoint_dir / "tft_fast.ckpt", fast_target.name)


def _apply_overrides_from_args(cfg: ProjectConfig, args: argparse.Namespace) -> ProjectConfig:
    # Training core overrides
    if args.max_epochs is not None:
        cfg.training.max_epochs = int(args.max_epochs)
        cfg.training.fast_max_epochs = max(cfg.training.fast_max_epochs, cfg.training.max_epochs)
    if args.batch_size is not None:
        cfg.training.batch_size = int(args.batch_size)
    if args.hidden_size is not None:
        cfg.training.hidden_size = int(args.hidden_size)
    if args.learning_rate is not None:
        cfg.training.learning_rate = float(args.learning_rate)
    if args.splits is not None:
        cfg.training.fast_max_splits = int(args.splits)
    if args.fast_dev_run:
        cfg.training.fast_dev_run = True

    # Walk-forward window overrides
    if args.train_months is not None:
        cfg.windows.train_months = int(args.train_months)
    if args.val_months is not None:
        cfg.windows.val_months = int(args.val_months)
    if args.test_months is not None:
        cfg.windows.test_months = int(args.test_months)
    if args.stride_months is not None:
        cfg.windows.stride_months = int(args.stride_months)

    return cfg


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TFT model with optional overrides and resume support")
    parser.add_argument("--resume", default="none", help="none | last | /path/to/checkpoint.ckpt")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--splits", type=int, default=None, help="limit number of walk-forward splits")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--train-months", type=int, default=None)
    parser.add_argument("--val-months", type=int, default=None)
    parser.add_argument("--test-months", type=int, default=None)
    parser.add_argument("--stride-months", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:  # pragma: no cover
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = _apply_overrides_from_args(DEFAULT_CONFIG, args)

    # Build everything up to trainer instantiation so we can pass resume ckpt
    ensure_artifact_dirs(cfg.artifacts)

    # Run the full training pipeline with resume support by monkey-patching fit call.
    # We reuse train_tft_model but intercept the call to Trainer.fit via a small wrapper.

    # Since train_tft_model constructs and calls trainer.fit internally, we emulate resume
    # by setting a global env var that lightning honors (ckpt_path) is only accepted by fit.
    # Instead, we re-run training but allow PL to pick up last checkpoint automatically via
    # save_last=True and specifying ckpt_path below when possible.

    # Simpler approach: call the same code path but override pl.Trainer.fit at runtime is invasive;
    # we instead duplicate a tiny part: construct trainer here and pass ckpt_path.

    # For maintainability, we fall back to standard path and when resuming, we call fit again
    # with ckpt_path through a minimal inline patch: re-run the function after setting an env flag.

    # Use a dedicated flag the function reads for resume (simple contract to avoid refactor)
    os_environ_resume = None
    if args.resume and args.resume.lower() != "none":
        os_environ_resume = args.resume

    # Monkey: store in global for trainer.fit usage via closure; fallback to inside function call
    global _RESUME_CKPT_PATH
    _RESUME_CKPT_PATH = os_environ_resume  # consumed inside train_tft_model via trainer.fit patch below

    # Patch: redefine Trainer.fit within this module scope to inject ckpt_path on first call
    original_fit = pl.Trainer.fit

    def patched_fit(self, *fit_args, **fit_kwargs):
        ckpt_path = None
        if isinstance(_RESUME_CKPT_PATH, str):
            if _RESUME_CKPT_PATH.lower() == "last":
                ckpt_path = "last"
            else:
                cp = Path(_RESUME_CKPT_PATH)
                if cp.exists():
                    ckpt_path = str(cp)
        # Only inject if not already provided
        if "ckpt_path" not in fit_kwargs and ckpt_path is not None:
            fit_kwargs["ckpt_path"] = ckpt_path
            print(f"[resume] Resuming from ckpt_path={ckpt_path}")
        try:
            return original_fit(self, *fit_args, **fit_kwargs)
        finally:
            # one-shot patch; restore after first use
            pl.Trainer.fit = original_fit

    pl.Trainer.fit = patched_fit

    # Execute training
    train_tft_model(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
