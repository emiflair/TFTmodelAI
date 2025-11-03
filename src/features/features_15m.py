"""
MINIMAL WORKING ENHANCED FEATURES
=================================
This completely replaces the original features_15m.py with working enhanced features.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def add_15m_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Add working enhanced 15m features for TFT training"""
    print("🔧 Adding minimal working enhanced 15m features...")
    
    enriched, new_cols = _compute_working_features(df.copy())
    
    print(f"✅ Added {len(new_cols)} enhanced 15m features")
    print(f"📊 Total features: {len(enriched.columns)}")
    
    return enriched, new_cols

def _compute_working_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Compute working enhanced features"""
    
    new_columns = []
    
    print("   📈 Computing enhanced technical indicators...")
    
    # === CORE TECHNICAL INDICATORS ===
    
    # RSI variants (most important technical indicator)
    df["rsi14"] = _rsi(df["close"], 14)
    df["rsi7"] = _rsi(df["close"], 7) 
    df["rsi21"] = _rsi(df["close"], 21)
    new_columns.extend(["rsi14", "rsi7", "rsi21"])
    
    # Stochastic
    df["stoch_k"] = _stochastic(df, 14)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    new_columns.extend(["stoch_k", "stoch_d"])
    
    # Williams %R  
    df["williams_r"] = _williams_r(df, 14)
    new_columns.append("williams_r")
    
    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    new_columns.extend(["macd", "macd_signal", "macd_hist"])
    
    # === VOLATILITY & BANDS ===
    
    # ATR
    df["atr"] = _atr(df, 14)
    new_columns.append("atr")
    
    # Bollinger Bands
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_upper"] = sma20 + (2 * std20)
    df["bb_lower"] = sma20 - (2 * std20)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    new_columns.extend(["bb_upper", "bb_lower", "bb_width", "bb_position"])
    
    # === VOLUME ===
    
    df["volume_sma"] = df["tick_volume"].rolling(10).mean()
    df["volume_ratio"] = df["tick_volume"] / df["volume_sma"]
    new_columns.extend(["volume_sma", "volume_ratio"])
    
    # === SPREAD ===
    
    # Z-score of spread (if spread column exists)
    if "spread" in df.columns:
        spread_mean = df["spread"].rolling(20).mean()
        spread_std = df["spread"].rolling(20).std()
        df["spread_z"] = (df["spread"] - spread_mean) / (spread_std + 1e-8)
        new_columns.append("spread_z")
    else:
        # If no spread column, create a dummy zero column
        df["spread_z"] = 0.0
        new_columns.append("spread_z")
    
    # === MOMENTUM ===
    
    df["roc_5"] = df["close"].pct_change(5)
    df["roc_10"] = df["close"].pct_change(10)
    df["momentum"] = df["close"] / df["close"].shift(10) - 1
    new_columns.extend(["roc_5", "roc_10", "momentum"])
    
    # === PRICE ACTION ===
    
    df["body_size"] = abs(df["close"] - df["open"]) / df["open"]
    df["hl_ratio"] = df["high"] / df["low"]
    df["co_ratio"] = df["close"] / df["open"]
    new_columns.extend(["body_size", "hl_ratio", "co_ratio"])
    
    # === MOVING AVERAGES ===
    
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["price_to_ema"] = df["close"] / df["ema_50"]
    df["price_to_sma"] = df["close"] / df["sma_20"]
    new_columns.extend(["ema_50", "sma_20", "price_to_ema", "price_to_sma"])
    
    # === REGIME DETECTION ===
    
    # Volatility regime
    returns = df["close"].pct_change()
    vol_20 = returns.rolling(20).std()
    vol_100 = returns.rolling(100).std()
    df["vol_regime"] = (vol_20 > vol_100).astype(int)
    new_columns.append("vol_regime")
    
    # Trend strength
    df["trend_strength"] = abs(df["close"] - df["close"].shift(20)) / (df["atr"] + 1e-8)
    new_columns.append("trend_strength")
    
    print("   🔧 Filling NaN values...")
    # Fill NaN values
    for col in new_columns:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(0)
    
    print(f"✅ Generated {len(new_columns)} working enhanced features")
    
    return df, new_columns

# === HELPER FUNCTIONS ===

def _rsi(prices: pd.Series, period: int) -> pd.Series:
    """RSI calculation"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period).mean()
    avg_loss = loss.ewm(alpha=1/period).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))

def _stochastic(df: pd.DataFrame, period: int) -> pd.Series:
    """Stochastic %K"""
    lowest_low = df["low"].rolling(period).min()
    highest_high = df["high"].rolling(period).max()
    return 100 * (df["close"] - lowest_low) / (highest_high - lowest_low + 1e-8)

def _williams_r(df: pd.DataFrame, period: int) -> pd.Series:
    """Williams %R"""
    highest_high = df["high"].rolling(period).max()
    lowest_low = df["low"].rolling(period).min()
    return -100 * (highest_high - df["close"]) / (highest_high - lowest_low + 1e-8)

def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range"""
    hl = df["high"] - df["low"]
    hc = abs(df["high"] - df["close"].shift(1))
    lc = abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period).mean()

if __name__ == "__main__":
    print("📊 Minimal Working Enhanced Features - Ready!")
    print("✅ All problematic functions removed")
    print("🚀 Ready for TFT training!")