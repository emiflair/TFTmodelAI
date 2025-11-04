"""
Trading Bot Configuration
Central configuration for the trading bot
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
SCALERS_DIR = ARTIFACTS_DIR / "scalers"
MANIFESTS_DIR = ARTIFACTS_DIR / "manifests"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    # Symbol and timeframe
    'symbol': 'XAUUSD',
    'timeframe': '15m',
    
    # Model paths - Use latest checkpoint (Sep 2024 - Mar 2025)
    # This checkpoint was trained yesterday (Nov 3, 2025)
    'checkpoint_path': CHECKPOINTS_DIR / "tft_XAUUSD_15m_3B_20240902_20250302.ckpt",
    'scaler_path': None,      # Let it auto-detect the matching scaler
    'manifest_path': MANIFESTS_DIR / "feature_manifest.json",
    
    # Prediction parameters
    'lookback_bars': 128,     # Number of historical bars required
    'prediction_horizon': 3,  # Bars ahead to predict
}

# ============================================================================
# STRATEGY CONFIGURATION
# ============================================================================

STRATEGY_CONFIG = {
    # Signal generation
    'min_confidence': 0.30,         # Minimum prediction confidence (0-1) - LOWERED for testing
    'min_move_pct': 0.10,           # Minimum predicted move percentage - LOWERED for testing
    'min_reward_risk': 0.5,         # Minimum reward:risk ratio - LOWERED for testing
    
    # Position management
    'enable_trailing_stop': True,   # Enable trailing stops
    'trailing_distance_pips': 50,   # Trailing stop distance in pips
    'partial_close_pct': 0.5,       # % of position to close at TP1 (50%)
    
    # Time filters
    'use_time_filter': False,       # DISABLED for testing - allow trading anytime
    'allowed_hours': list(range(0, 24)),  # Trading hours in UTC (24/7 for testing)
    
    # Signal interval
    'min_signal_interval_minutes': 15,  # Minimum time between signals
}

# ============================================================================
# RISK MANAGEMENT CONFIGURATION
# ============================================================================

RISK_CONFIG = {
    # Account settings
    'account_balance': 10000.0,     # Starting account balance in USD
    'max_daily_loss': 4.0,          # Maximum daily loss (% of balance) = $400
    'max_daily_profit': 5.0,        # Maximum daily profit target (% of balance) = $500
    
    # Dynamic position sizing based on AI confidence
    'risk_low_confidence': 0.5,     # Low confidence (0.3-0.49) → 0.5% risk
    'risk_medium_confidence': 1.0,  # Medium confidence (0.5-0.69) → 1% risk
    'risk_high_confidence': 2.0,    # High confidence (0.85-1.0) → 2% risk
    
    # Position limits
    'max_lot_size': 2.0,            # Maximum lot size cap
    'max_open_positions': 2,        # Maximum concurrent positions (same direction only)
    'max_leverage': 10.0,           # Maximum leverage ratio
    'same_direction_only': True,    # Only allow trades in same direction
    
    # Dynamic R:R based on AI confidence
    'rr_defensive': 0.5,            # Confidence 0.3-0.49 → 1:0.5
    'rr_normal': 1.0,               # Confidence 0.5-0.69 → 1:1
    'rr_strong': 2.0,               # Confidence 0.7-0.84 → 1:2
    'rr_high': 3.0,                 # Confidence 0.85-1.0 → 1:3
    
    # Cost adjustments for XAUUSD
    'pip_value': 1.0,               # $1 per pip per 0.01 lot (micro lot)
    'spread_cost_per_microlot': 0.5,  # Spread cost in $ per 0.01 lot
    'commission_per_lot': 6.0,      # Commission per 1.0 lot round-turn
    'max_spread_pips': 25,          # Skip trade if spread > 25 pips
    
    # Slippage and safety
    'slippage_tolerance_pips': 10,  # ±10 pips slippage tolerance
    'consecutive_loss_reduction': 0.5,  # Cut risk by 50% after 2 losses
    'max_consecutive_losses': 2,    # Trigger risk reduction threshold
    
    # Legacy compatibility
    'max_correlation': 0.7,         # Maximum correlation between positions
}

# ============================================================================
# MT5 CONFIGURATION
# ============================================================================

MT5_CONFIG = {
    # Connection settings (leave None to use default/last used)
    'login': 52587216,          # Your MT5 account number (int)
    'password': 'Y7p$bW1iJKmtSq',  # Your MT5 password (str)
    'server': 'ICMarketsSC-Demo',  # Your broker's server name (str)
    
    # Trading settings
    'magic_number': 123456,     # Magic number for bot trades
    'deviation': 20,            # Slippage in points
    'filling_type': 'FOK',      # Order filling type: 'FOK' or 'IOC'
    
    # Symbol settings
    'symbol': 'XAUUSD',         # Trading symbol
    'timeframe': '15m',         # Timeframe
}

# ============================================================================
# BOT CONFIGURATION
# ============================================================================

BOT_CONFIG = {
    # Update frequency
    'update_interval': 60,      # Seconds between bot updates
    
    # Logging
    'log_file': 'trading_bot/bot.log',
    'log_level': 'INFO',        # DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    # Trade logging (CSV)
    'trade_log_file': 'trading_bot/trade_log.csv',
    'enable_trade_logging': True,
    
    # Safety
    'enable_live_trading': False,  # Set to True to enable live trading
    'paper_trading': True,         # Run in paper trading mode (simulation)
}

# ============================================================================
# COMPLETE CONFIGURATION
# ============================================================================

CONFIG = {
    'symbol': MODEL_CONFIG['symbol'],
    'timeframe': MODEL_CONFIG['timeframe'],
    'update_interval': BOT_CONFIG['update_interval'],
    
    'model_checkpoint': MODEL_CONFIG['checkpoint_path'],
    'scaler_path': MODEL_CONFIG['scaler_path'],
    'manifest_path': MODEL_CONFIG['manifest_path'],
    
    'strategy_config': STRATEGY_CONFIG,
    'risk_config': RISK_CONFIG,
    'mt5_config': MT5_CONFIG,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_latest_checkpoint():
    """Get path to latest model checkpoint"""
    if not CHECKPOINTS_DIR.exists():
        return None
    
    checkpoints = list(CHECKPOINTS_DIR.glob("*.ckpt"))
    if not checkpoints:
        return None
    
    # Get most recent by modification time
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)


def get_latest_scaler():
    """Get path to latest scaler file"""
    if not SCALERS_DIR.exists():
        return None
    
    scalers = list(SCALERS_DIR.glob("*.pkl"))
    if not scalers:
        return None
    
    # Get most recent by modification time
    latest = max(scalers, key=lambda p: p.stat().st_mtime)
    return str(latest)


def get_manifest():
    """Get path to feature manifest"""
    manifest_path = MANIFESTS_DIR / "feature_manifest.json"
    if manifest_path.exists():
        return str(manifest_path)
    return None


def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check if artifacts directory exists
    if not ARTIFACTS_DIR.exists():
        errors.append(f"Artifacts directory not found: {ARTIFACTS_DIR}")
    
    # Check model files
    if MODEL_CONFIG['checkpoint_path'] is None:
        checkpoint = get_latest_checkpoint()
        if checkpoint is None:
            errors.append("No model checkpoint found")
    
    if MODEL_CONFIG['scaler_path'] is None:
        scaler = get_latest_scaler()
        if scaler is None:
            errors.append("No scaler file found")
    
    if MODEL_CONFIG['manifest_path'] is None:
        manifest = get_manifest()
        if manifest is None:
            errors.append("No feature manifest found")
    
    # Check risk settings
    if RISK_CONFIG['max_risk_per_trade'] > RISK_CONFIG['max_daily_risk']:
        errors.append("max_risk_per_trade cannot exceed max_daily_risk")
    
    if RISK_CONFIG['min_reward_risk'] < 1.0:
        errors.append("min_reward_risk should be >= 1.0")
    
    # Check strategy settings
    if STRATEGY_CONFIG['min_confidence'] < 0 or STRATEGY_CONFIG['min_confidence'] > 1:
        errors.append("min_confidence must be between 0 and 1")
    
    if STRATEGY_CONFIG['min_move_pct'] < 0:
        errors.append("min_move_pct must be positive")
    
    # Return validation result
    if errors:
        return {
            'valid': False,
            'errors': errors
        }
    
    return {
        'valid': True,
        'errors': []
    }


def print_config():
    """Print current configuration"""
    print("\n" + "=" * 80)
    print("TRADING BOT CONFIGURATION")
    print("=" * 80)
    
    print("\n[MODEL]")
    print(f"  Symbol: {MODEL_CONFIG['symbol']}")
    print(f"  Timeframe: {MODEL_CONFIG['timeframe']}")
    print(f"  Checkpoint: {MODEL_CONFIG['checkpoint_path'] or 'Auto-detect'}")
    print(f"  Scaler: {MODEL_CONFIG['scaler_path'] or 'Auto-detect'}")
    print(f"  Manifest: {MODEL_CONFIG['manifest_path'] or 'Auto-detect'}")
    
    print("\n[STRATEGY]")
    print(f"  Min Confidence: {STRATEGY_CONFIG['min_confidence']}")
    print(f"  Min Move %: {STRATEGY_CONFIG['min_move_pct']}%")
    print(f"  Min R:R: {STRATEGY_CONFIG['min_reward_risk']}")
    print(f"  Trailing Stop: {STRATEGY_CONFIG['enable_trailing_stop']} ({STRATEGY_CONFIG['trailing_distance_pips']} pips)")
    print(f"  Partial Close: {STRATEGY_CONFIG['partial_close_pct'] * 100}%")
    print(f"  Time Filter: {STRATEGY_CONFIG['use_time_filter']} ({STRATEGY_CONFIG['allowed_hours']})")
    
    print("\n[RISK MANAGEMENT]")
    print(f"  Risk per Trade: {RISK_CONFIG['max_risk_per_trade']}%")
    print(f"  Daily Risk Limit: {RISK_CONFIG['max_daily_risk']}%")
    print(f"  Max Positions: {RISK_CONFIG['max_open_positions']}")
    print(f"  Max Leverage: {RISK_CONFIG['max_leverage']}x")
    print(f"  Min R:R: {RISK_CONFIG['min_reward_risk']}")
    
    print("\n[MT5]")
    print(f"  Symbol: {MT5_CONFIG['symbol']}")
    print(f"  Timeframe: {MT5_CONFIG['timeframe']}")
    print(f"  Magic Number: {MT5_CONFIG['magic_number']}")
    print(f"  Login: {'***' if MT5_CONFIG['login'] else 'Default'}")
    
    print("\n[BOT]")
    print(f"  Update Interval: {BOT_CONFIG['update_interval']}s")
    print(f"  Log Level: {BOT_CONFIG['log_level']}")
    print(f"  Live Trading: {BOT_CONFIG['enable_live_trading']}")
    print(f"  Paper Trading: {BOT_CONFIG['paper_trading']}")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    # Print configuration
    print_config()
    
    # Validate configuration
    validation = validate_config()
    
    if validation['valid']:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration errors:")
        for error in validation['errors']:
            print(f"   - {error}")
