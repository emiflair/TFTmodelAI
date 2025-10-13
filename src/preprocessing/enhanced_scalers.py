"""Enhanced preprocessing utilities with robust scaling, outlier handling, and feature normalization."""
from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class EnhancedFeatureScaler:
    """Enhanced scaler with outlier handling and multiple scaling methods."""
    median: float
    iqr: float
    mean: float
    std: float
    q01: float  # 1st percentile
    q99: float  # 99th percentile
    method: str = "robust"  # "robust", "standard", or "minmax"

    def transform(self, values: pd.Series, winsorize: bool = True) -> pd.Series:
        """Transform values with optional winsorization."""
        transformed = values.copy()
        
        # Apply winsorization to handle outliers
        if winsorize:
            transformed = transformed.clip(lower=self.q01, upper=self.q99)
        
        # Apply scaling based on method
        if self.method == "robust":
            denom = self.iqr if self.iqr != 0 else 1e-8
            return (transformed - self.median) / denom
        elif self.method == "standard":
            denom = self.std if self.std != 0 else 1e-8
            return (transformed - self.mean) / denom
        elif self.method == "minmax":
            denom = (self.q99 - self.q01) if (self.q99 - self.q01) != 0 else 1e-8
            return (transformed - self.q01) / denom
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")

    def inverse_transform(self, values: pd.Series) -> pd.Series:
        """Inverse transform scaled values back to original scale."""
        if self.method == "robust":
            return values * (self.iqr if self.iqr != 0 else 1e-8) + self.median
        elif self.method == "standard":
            return values * (self.std if self.std != 0 else 1e-8) + self.mean
        elif self.method == "minmax":
            return values * (self.q99 - self.q01 if (self.q99 - self.q01) != 0 else 1e-8) + self.q01
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")


class EnhancedScalerStore:
    """Enhanced scaler store with multiple preprocessing techniques."""
    
    def __init__(self, default_method: str = "robust") -> None:
        self.scalers: Dict[str, EnhancedFeatureScaler] = {}
        self.default_method = default_method
        self.feature_types: Dict[str, str] = {}  # Track feature types for appropriate scaling
        
    def fit(self, df: pd.DataFrame, columns: Iterable[str], 
            feature_types: Optional[Dict[str, str]] = None) -> None:
        """Fit scalers on the data with enhanced statistics."""
        
        for col in columns:
            if col not in df.columns:
                continue
                
            series = df[col].dropna()
            if len(series) == 0:
                continue
            
            # Determine appropriate scaling method based on feature type
            feature_type = feature_types.get(col, "numeric") if feature_types else "numeric"
            self.feature_types[col] = feature_type
            
            # Choose scaling method based on feature characteristics
            scaling_method = self._determine_scaling_method(series, feature_type)
            
            # Compute comprehensive statistics
            stats = self._compute_feature_statistics(series)
            
            self.scalers[col] = EnhancedFeatureScaler(
                median=stats["median"],
                iqr=stats["iqr"],
                mean=stats["mean"],
                std=stats["std"],
                q01=stats["q01"],
                q99=stats["q99"],
                method=scaling_method
            )
    
    def _determine_scaling_method(self, series: pd.Series, feature_type: str) -> str:
        """Determine the best scaling method based on feature characteristics."""
        
        # Calculate distribution characteristics
        skewness = series.skew()
        kurtosis = series.kurtosis()
        
        # Count outliers (beyond 3 standard deviations)
        mean = series.mean()
        std = series.std()
        outlier_ratio = ((series - mean).abs() > 3 * std).mean()
        
        # Decision logic for scaling method
        if feature_type in ["price", "volume", "indicator"]:
            # For financial features that may have outliers
            if outlier_ratio > 0.05 or abs(skewness) > 2:
                return "robust"  # Robust to outliers
            else:
                return "standard"
        elif feature_type in ["return", "change", "ratio"]:
            # For return-like features
            if abs(skewness) > 1.5 or kurtosis > 3:
                return "robust"
            else:
                return "standard"
        elif feature_type in ["binary", "categorical"]:
            # For categorical features
            return "minmax"
        else:
            # Default to robust scaling
            return self.default_method
    
    def _compute_feature_statistics(self, series: pd.Series) -> Dict[str, float]:
        """Compute comprehensive statistics for a feature."""
        return {
            "median": float(series.median()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
            "q01": float(series.quantile(0.01)),
            "q99": float(series.quantile(0.99)),
        }
    
    def transform(self, df: pd.DataFrame, winsorize: bool = True) -> pd.DataFrame:
        """Transform dataframe with enhanced preprocessing."""
        transformed = df.copy()
        
        # Apply feature engineering preprocessing
        transformed = self._apply_feature_engineering(transformed)
        
        # Apply scaling
        for col, scaler in self.scalers.items():
            if col in transformed.columns:
                transformed[col] = scaler.transform(transformed[col], winsorize=winsorize)
        
        # Handle missing values after transformation
        transformed = self._handle_missing_values(transformed)
        
        return transformed

    def transform_inplace(self, df: pd.DataFrame, winsorize: bool = True) -> None:
        """Transform dataframe in-place with enhanced preprocessing."""
        
        # Apply feature engineering preprocessing
        self._apply_feature_engineering_inplace(df)
        
        # Apply scaling
        for col, scaler in self.scalers.items():
            if col in df.columns:
                df[col] = scaler.transform(df[col], winsorize=winsorize)
        
        # Handle missing values after transformation
        self._handle_missing_values_inplace(df)
    
    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply additional feature engineering transformations."""
        transformed = df.copy()
        
        # Log transformation for highly skewed features
        for col in df.columns:
            if col in self.scalers and self.feature_types.get(col) == "volume":
                # Apply log(1 + x) transformation for volume-like features
                if (df[col] > 0).all():
                    transformed[f"{col}_log"] = np.log1p(df[col])
        
        # Rolling z-scores for stationarity
        for col in df.columns:
            if col in self.scalers and "return" in col.lower():
                # Apply rolling z-score for return features
                rolling_mean = df[col].rolling(window=50, min_periods=10).mean()
                rolling_std = df[col].rolling(window=50, min_periods=10).std()
                transformed[f"{col}_zscore"] = (df[col] - rolling_mean) / rolling_std.replace(0, np.nan)
        
        return transformed
    
    def _apply_feature_engineering_inplace(self, df: pd.DataFrame) -> None:
        """Apply additional feature engineering transformations in-place."""
        
        # Log transformation for highly skewed features
        for col in list(df.columns):
            if col in self.scalers and self.feature_types.get(col) == "volume":
                if (df[col] > 0).all():
                    df[f"{col}_log"] = np.log1p(df[col])
        
        # Rolling z-scores for stationarity
        for col in list(df.columns):
            if col in self.scalers and "return" in col.lower():
                rolling_mean = df[col].rolling(window=50, min_periods=10).mean()
                rolling_std = df[col].rolling(window=50, min_periods=10).std()
                df[f"{col}_zscore"] = (df[col] - rolling_mean) / rolling_std.replace(0, np.nan)
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with intelligent imputation."""
        filled = df.copy()
        
        for col in df.columns:
            if df[col].isna().any():
                if col in self.feature_types:
                    feature_type = self.feature_types[col]
                    
                    if feature_type in ["price", "indicator"]:
                        # Forward fill for price-like features
                        filled[col] = filled[col].fillna(method='ffill')
                        # Backward fill for remaining NAs
                        filled[col] = filled[col].fillna(method='bfill')
                    elif feature_type in ["return", "change"]:
                        # Fill with 0 for return-like features
                        filled[col] = filled[col].fillna(0)
                    elif feature_type in ["volume"]:
                        # Fill with median for volume features
                        filled[col] = filled[col].fillna(filled[col].median())
                    else:
                        # Default to median imputation
                        filled[col] = filled[col].fillna(filled[col].median())
                else:
                    # Default to median imputation
                    filled[col] = filled[col].fillna(filled[col].median())
        
        return filled
    
    def _handle_missing_values_inplace(self, df: pd.DataFrame) -> None:
        """Handle missing values in-place with intelligent imputation."""
        
        for col in df.columns:
            if df[col].isna().any():
                if col in self.feature_types:
                    feature_type = self.feature_types[col]
                    
                    if feature_type in ["price", "indicator"]:
                        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                    elif feature_type in ["return", "change"]:
                        df[col] = df[col].fillna(0)
                    elif feature_type in ["volume"]:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].median())
    
    def get_feature_importance_weights(self) -> Dict[str, float]:
        """Calculate feature importance weights based on scaling statistics."""
        weights = {}
        
        for col, scaler in self.scalers.items():
            # Features with higher variability (IQR) get higher weights
            # Normalized by the median to account for scale differences
            if scaler.median != 0:
                variability_score = scaler.iqr / abs(scaler.median)
            else:
                variability_score = scaler.iqr
            
            # Cap the weight to prevent extreme values
            weights[col] = min(max(variability_score, 0.1), 10.0)
        
        return weights

    def save(self, path: Path) -> None:
        """Save enhanced scaler store to disk."""
        data = {
            "scalers": {col: asdict(scaler) for col, scaler in self.scalers.items()},
            "feature_types": self.feature_types,
            "default_method": self.default_method
        }
        
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "EnhancedScalerStore":
        """Load enhanced scaler store from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        store = cls(default_method=data.get("default_method", "robust"))
        store.feature_types = data.get("feature_types", {})
        
        # Reconstruct scalers
        for col, scaler_data in data["scalers"].items():
            store.scalers[col] = EnhancedFeatureScaler(**scaler_data)
        
        return store


def detect_feature_types(df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, str]:
    """Automatically detect feature types for appropriate scaling."""
    feature_types = {}
    
    for col in feature_columns:
        if col not in df.columns:
            continue
            
        col_lower = col.lower()
        
        # Price-related features
        if any(keyword in col_lower for keyword in ["price", "open", "high", "low", "close", "ema", "sma", "vwap"]):
            feature_types[col] = "price"
        # Return/change features
        elif any(keyword in col_lower for keyword in ["return", "roc", "change", "diff", "slope"]):
            feature_types[col] = "return"
        # Volume features
        elif any(keyword in col_lower for keyword in ["volume", "vol", "tick"]):
            feature_types[col] = "volume"
        # Technical indicators
        elif any(keyword in col_lower for keyword in ["rsi", "stochastic", "cci", "mfi", "atr", "bb"]):
            feature_types[col] = "indicator"
        # Binary/categorical features
        elif any(keyword in col_lower for keyword in ["session", "regime", "phase", "trend"]):
            feature_types[col] = "binary"
        # Ratio features
        elif any(keyword in col_lower for keyword in ["ratio", "dist", "norm", "efficiency"]):
            feature_types[col] = "ratio"
        else:
            # Default to numeric
            feature_types[col] = "numeric"
    
    return feature_types