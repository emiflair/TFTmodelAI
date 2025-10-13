"""Evaluation metrics for TFT quantile forecasts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd


@dataclass
class EvaluationResult:
    pinball: Dict[float, float]
    crps: float
    coverage: float
    directional_hit: float
    profit_factor_band: Dict[str, float]


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.where(diff >= 0, q * diff, (q - 1) * diff)))


def crps_from_quantiles(y_true: np.ndarray, quantiles: Iterable[float], predictions: Dict[float, np.ndarray]) -> float:
    losses = [pinball_loss(y_true, predictions[q], q) for q in quantiles]
    return float(2 * np.sum(losses))


def band_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def directional_hit_rate(y_true: np.ndarray, median_pred: np.ndarray) -> float:
    eligible = (median_pred != 0) & (y_true != 0)
    if eligible.sum() == 0:
        return float("nan")
    hits = np.sign(median_pred[eligible]) == np.sign(y_true[eligible])
    return float(np.mean(hits))


def simple_profit_factor(
    df: pd.DataFrame,
    lower_q: str,
    median_q: str,
    upper_q: str,
    band_threshold: float,
    atr_norm_col: str,
    normalized: bool = False,
) -> float:
    trades = []
    for _, row in df.iterrows():
        spread = row[upper_q] - row[lower_q]
        if normalized:
            threshold = band_threshold * row[atr_norm_col]
        else:
            threshold = band_threshold
        if spread >= threshold:
            if row[lower_q] > 0 and row[median_q] > 0:
                trades.append(row["target"])
            elif row[upper_q] < 0 and row[median_q] < 0:
                trades.append(-row["target"])
    if not trades:
        return float("nan")
    gains = [t for t in trades if t > 0]
    losses = [-t for t in trades if t < 0]
    if not losses:
        return float("inf")
    return float(sum(gains) / sum(losses))
