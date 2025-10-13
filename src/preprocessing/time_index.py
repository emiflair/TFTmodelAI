"""Time index utilities."""
from __future__ import annotations

import pandas as pd


def add_time_index(df: pd.DataFrame, freq_minutes: int = 15) -> pd.DataFrame:
    augmented = df.copy()
    base = augmented["timestamp"].min()
    delta = (augmented["timestamp"] - base).dt.total_seconds() // (freq_minutes * 60)
    augmented["time_idx"] = delta.astype(int)
    augmented["series_id"] = "EURUSD"
    return augmented
