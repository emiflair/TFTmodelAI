"""Walk-forward split utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from ..config import TrainingWindowConfig


@dataclass
class SplitWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self) -> dict:
        return {
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "val_end": self.val_end.isoformat(),
            "test_end": self.test_end.isoformat(),
        }


def generate_walk_forward_splits(df: pd.DataFrame, config: TrainingWindowConfig) -> List[SplitWindow]:
    timestamps = df["timestamp"].sort_values()
    start = timestamps.iloc[0].floor("h")
    end = timestamps.iloc[-1]
    windows: List[SplitWindow] = []

    current_start = start
    while True:
        train_end = current_start + pd.DateOffset(months=config.train_months)
        val_end = train_end + pd.DateOffset(months=config.val_months)
        test_end = val_end + pd.DateOffset(months=config.test_months)
        if test_end > end:
            break
        windows.append(
            SplitWindow(
                train_start=current_start,
                train_end=train_end,
                val_end=val_end,
                test_end=test_end,
            )
        )
        current_start += pd.DateOffset(months=config.stride_months)
    if not windows:
        raise ValueError("Insufficient history for requested walk-forward configuration")
    return windows
