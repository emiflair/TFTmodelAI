"""
SIMPLIFIED ENHANCED FEATURES FOR TFT TRAINING
=============================================
This version removes problematic rolling apply functions and uses
standard pandas operations for reliability.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def add_15m_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Add simplified but effective 15m features for TFT training"""
    print("🔧 Adding simplified enhanced 15m features...")
    
    enriched, new_cols = _compute_simplified_15m_features(df.copy())
    
    print(f"✅ Added {len(new_cols)} enhanced 15m features")
    print(f"📊 Total features: {len(enriched.columns)}")
    
    return enriched, new_cols

def _compute_simplified_15m_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Compute simplified enhanced features that are proven to work"""
    
    new_columns = []
    
    # === BASIC TECHNICAL INDICATORS ===
    print("   📈 Computing basic technical indicators...")
    
    # RSI variants
    df["rsi14"] = _rsi(df["close"], period=14)
    df["rsi7"] = _rsi(df["close"], period=7)
    df["rsi21"] = _rsi(df["close"], period=21)
    new_columns.extend(["rsi14", "rsi7", "rsi21"])
    
    # Stochastic oscillator
    df["stoch_k"] = _stochastic_k(df, period=14)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    new_columns.extend(["stoch_k", "stoch_d"])
    
    # Williams %R
    df["williams_r"] = _williams_r(df, period=14)
    new_columns.append("williams_r")
    
    # === MOVING AVERAGES ===
    print("   📊 Computing moving averages...")
    
    # EMA variants
    df["ema12"] = df["close"].ewm(span=12).mean()
    df["ema26"] = df["close"].ewm(span=26).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    new_columns.extend(["ema12", "ema26", "ema50"])
    
    # MACD
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]
    new_columns.extend(["macd", "macd_signal", "macd_histogram"])
    
    # === VOLATILITY INDICATORS ===
    print("   📉 Computing volatility indicators...")
    
    # Average True Range
    df["atr"] = _average_true_range(df, period=14)
    new_columns.append("atr")
    
    # Bollinger Bands
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_upper"] = sma20 + (2 * std20)
    df["bb_lower"] = sma20 - (2 * std20)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    new_columns.extend(["bb_upper", "bb_lower", "bb_width", "bb_position"])
    
    # === VOLUME INDICATORS ===
    print("   📈 Computing volume indicators...")
    
    # Volume moving averages
    df["volume_sma10"] = df["tick_volume"].rolling(10).mean()
    df["volume_ratio"] = df["tick_volume"] / df["volume_sma10"]
    new_columns.extend(["volume_sma10", "volume_ratio"])
    
    # Simple On Balance Volume
    price_change = df["close"].diff()
    df["obv"] = (np.sign(price_change) * df["tick_volume"]).cumsum()
    df["obv_sma10"] = df["obv"].rolling(10).mean()
    new_columns.extend(["obv", "obv_sma10"])
    
    # === PRICE ACTION INDICATORS ===
    print("   💹 Computing price action indicators...")
    
    # Price ranges
    df["high_low_ratio"] = df["high"] / df["low"]
    df["close_open_ratio"] = df["close"] / df["open"]
    df["body_size"] = abs(df["close"] - df["open"]) / df["open"]
    df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["open"]
    df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["open"]
    new_columns.extend(["high_low_ratio", "close_open_ratio", "body_size", "upper_shadow", "lower_shadow"])
    
    # === TIME-BASED FEATURES ===
    print("   🕐 Computing time-based features...")
    
    # Check if we have a datetime index
    if hasattr(df.index, 'hour') and hasattr(df.index, 'dayofweek'):
        # Hour of day effects
        df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        new_columns.extend(["hour_sin", "hour_cos"])
        
        # Day of week effects
        df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        new_columns.extend(["dow_sin", "dow_cos"])
    else:
        print("   ⚠️  Skipping time features - no datetime index")
        # Add dummy time features
        df["hour_sin"] = 0.0
        df["hour_cos"] = 1.0
        df["dow_sin"] = 0.0
        df["dow_cos"] = 1.0
        new_columns.extend(["hour_sin", "hour_cos", "dow_sin", "dow_cos"])
    
    # === MOMENTUM INDICATORS ===
    print("   🚀 Computing momentum indicators...")
    
    # Rate of change
    df["roc_5"] = df["close"].pct_change(5)
    df["roc_10"] = df["close"].pct_change(10)
    df["roc_20"] = df["close"].pct_change(20)
    new_columns.extend(["roc_5", "roc_10", "roc_20"])
    
    # Momentum
    df["momentum_10"] = df["close"] / df["close"].shift(10) - 1
    df["momentum_20"] = df["close"] / df["close"].shift(20) - 1
    new_columns.extend(["momentum_10", "momentum_20"])
    
    # === MARKET REGIME DETECTION ===
    print("   🎯 Computing market regime features...")
    
    # Volatility regime
    returns = df["close"].pct_change()
    vol_20 = returns.rolling(20).std()
    vol_100 = returns.rolling(100).std()
    df["vol_regime"] = (vol_20 > vol_100).astype(int)  # 1 = high vol, 0 = low vol
    new_columns.append("vol_regime")
    
    # Trend strength
    df["trend_strength"] = abs(df["close"] - df["close"].shift(20)) / df["atr"]
    new_columns.append("trend_strength")
    
    # Fill NaN values
    print("   🔧 Filling NaN values...")
    for col in new_columns:
        df[col] = df[col].fillna(method='ffill').fillna(0)
    
    print(f"✅ Generated {len(new_columns)} reliable enhanced features")
    
    return df, new_columns

# === HELPER FUNCTIONS ===

def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using standard pandas operations"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/period).mean()
    avg_loss = loss.ewm(alpha=1/period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _stochastic_k(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Stochastic %K"""
    lowest_low = df["low"].rolling(period).min()
    highest_high = df["high"].rolling(period).max()
    k_percent = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
    return k_percent

def _williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Williams %R"""
    highest_high = df["high"].rolling(period).max()
    lowest_low = df["low"].rolling(period).min()
    wr = -100 * (highest_high - df["close"]) / (highest_high - lowest_low)
    return wr

def _average_true_range(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high_low = df["high"] - df["low"]
    high_close_prev = abs(df["high"] - df["close"].shift(1))
    low_close_prev = abs(df["low"] - df["close"].shift(1))
    
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period).mean()
    return atr

if __name__ == "__main__":
    print("📊 Simplified Enhanced Features Module - Ready for TFT Training!")
    print("🔧 This version uses only reliable pandas operations")
    print("✅ No rolling.apply() functions that cause issues")