"""Target engineering utilities."""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def add_future_return(df: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, str]:
    enriched = df.copy()
    future_close = enriched["close"].shift(-horizon)
    target_name = f"target_r_{horizon}"
    enriched[target_name] = (future_close - enriched["close"]) / enriched["close"]
    if enriched[target_name].iloc[:-horizon].equals(
        (enriched["close"].shift(-horizon).iloc[:-horizon] - enriched["close"].iloc[:-horizon])
        / enriched["close"].iloc[:-horizon]
    ) is False:
        raise AssertionError("Target alignment mismatch")
    enriched.dropna(inplace=True)
    return enriched, target_name
