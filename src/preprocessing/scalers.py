"""Robust scaling utilities with persistence."""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass
class FeatureScaler:
    median: float
    iqr: float

    def transform(self, values: pd.Series) -> pd.Series:
        denom = self.iqr if self.iqr != 0 else 1e-8
        return (values - self.median) / denom

    def inverse_transform(self, values: pd.Series) -> pd.Series:
        return values * (self.iqr if self.iqr != 0 else 1e-8) + self.median


class RobustScalerStore:
    def __init__(self) -> None:
        self.scalers: Dict[str, FeatureScaler] = {}

    def fit(self, df: pd.DataFrame, columns: Iterable[str]) -> None:
        for col in columns:
            series = df[col].dropna()
            median = float(series.median())
            q75 = float(series.quantile(0.75))
            q25 = float(series.quantile(0.25))
            self.scalers[col] = FeatureScaler(median=median, iqr=q75 - q25)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = df.copy()
        for col, scaler in self.scalers.items():
            transformed[col] = scaler.transform(transformed[col])
        return transformed

    def transform_inplace(self, df: pd.DataFrame) -> None:
        for col, scaler in self.scalers.items():
            df[col] = scaler.transform(df[col])

    def save(self, path: Path) -> None:
        serializable = {col: asdict(scaler) for col, scaler in self.scalers.items()}
        with path.open("wb") as fout:
            pickle.dump(serializable, fout)

    @classmethod
    def load(cls, path: Path) -> "RobustScalerStore":
        instance = cls()
        with path.open("rb") as fin:
            data = pickle.load(fin)
        
        # Handle both old format (flat dict) and new format (nested with 'scalers' key)
        if "scalers" in data:
            # New enhanced format
            scalers_data = data["scalers"]
        else:
            # Old flat format
            scalers_data = data
            
        for col, params in scalers_data.items():
            # Only use median and iqr for FeatureScaler
            instance.scalers[col] = FeatureScaler(
                median=params["median"],
                iqr=params["iqr"]
            )
        return instance

    def feature_list(self) -> List[str]:
        return list(self.scalers.keys())
