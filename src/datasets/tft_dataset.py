"""Dataset preparation helpers for PyTorch Forecasting Temporal Fusion Transformer."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

from pytorch_forecasting import TimeSeriesDataSet

from ..config import QuantileConfig, TrainingConfig


def build_tft_datasets(
    train_df,
    val_df,
    test_df,
    target: str,
    known_future_reals: Iterable[str],
    unknown_reals: Iterable[str],
    static_categoricals: Iterable[str],
    quantiles: QuantileConfig,
    training_cfg: TrainingConfig,
    add_categoricals: Iterable[str] = (),
) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
    parameters: Dict = {
        "time_idx": "time_idx",
        "target": target,
        "group_ids": ["series_id"],
        "max_encoder_length": training_cfg.batch_size // 2 if training_cfg.batch_size < 256 else 256,
        "max_prediction_length": 1,
        "static_categoricals": list(static_categoricals),
        "time_varying_known_reals": list(known_future_reals),
        "time_varying_unknown_reals": list(unknown_reals),
        "time_varying_known_categoricals": list(add_categoricals),
        "allow_missing_timesteps": False,
        "target_normalizer": None,
    }

    training = TimeSeriesDataSet(train_df, **parameters)
    validation = training.from_dataset(TimeSeriesDataSet(val_df, **parameters), val_df)
    testing = training.from_dataset(TimeSeriesDataSet(test_df, **parameters), test_df)
    return training, validation, testing
