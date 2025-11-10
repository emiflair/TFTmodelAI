"""
CLEAN TRADING BOT CONFIGURATION
================================

Simple, clear configuration for the trading bot after clean training.
"""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    # Use the clean trained model
    'checkpoint_path': 'artifacts/checkpoints/tft_XAUUSD_15m_128x3_final.ckpt',
    
    # Model expects these dimensions
    'lookback_bars': 128,        # Must match training
    'horizon_bars': 3,           # Must match training
    'fetch_bars': 200,           # Fetch extra for feature engineering (lose ~70 to NaN)
    'min_bars_required': 128,    # Minimum after feature engineering
}

# ============================================================================
# STRATEGY CONFIGURATION  
# ============================================================================

STRATEGY_CONFIG = {
    # Signal thresholds
    'min_confidence': 0.50,      # Minimum confidence to trade (0-1)
    'min_move_pct': 0.15,        # Minimum expected move % (0.15 = 0.15%)
    
    # Trade management
    'enable_trailing_stop': True,
    'trailing_distance_pips': 50,
    'partial_close_pct': 0.5,    # Close 50% at first target
    
    # Time filters
    'use_time_filter': False,    # Disabled for testing
    'allowed_hours': list(range(0, 24)),  # 24/7 trading
}

# ============================================================================
# RISK MANAGEMENT CONFIGURATION
# ============================================================================

RISK_CONFIG = {
    # Account settings
    'max_daily_loss_pct': 4.0,   # Stop trading if lose 4% in a day
    
    # Position sizing based on confidence
    'risk_low_confidence': 0.5,   # 0.5% risk for confidence 0.50-0.69
    'risk_medium_confidence': 1.0, # 1.0% risk for confidence 0.70-0.84
    'risk_high_confidence': 2.0,   # 2.0% risk for confidence 0.85-1.0
    
    # Risk limits
    'max_lot_size': 2.0,
    'max_open_positions': 3,
    'max_spread_pips': 25,
    
    # Risk:Reward ratios
    'rr_low_confidence': 0.5,     # 1:0.5 for low confidence
    'rr_medium_confidence': 1.0,  # 1:1 for medium confidence
    'rr_high_confidence': 2.0,    # 1:2 for high confidence
    
    # Consecutive loss protection
    'max_consecutive_losses': 2,
    'consecutive_loss_reduction': 0.5,  # Cut risk by 50% after 2 losses
}

# ============================================================================
# MT5 CONNECTION
# ============================================================================

MT5_CONFIG = {
    'login': 52587216,
    'password': 'Y7p$bW1iJKmtSq',
    'server': 'ICMarketsSC-Demo',
}

# ============================================================================
# BOT SETTINGS
# ============================================================================

BOT_CONFIG = {
    'symbol': 'XAUUSD',
    'timeframe': '15m',
    'update_interval': 30,  # Update every 30 seconds
    'test_mode': False,     # Set True to inject fake signals for testing
}
