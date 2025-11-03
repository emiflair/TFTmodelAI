"""
Quick Start Configuration - AI Trading Bot with Dynamic Risk Management
Copy this configuration to get started quickly
"""

# ============================================================================
# QUICK START: $10,000 ACCOUNT - BALANCED PROFILE
# ============================================================================

QUICK_START_CONFIG = {
    'symbol': 'XAUUSD',
    'timeframe': '15m',
    'update_interval': 60,  # Check every 60 seconds
    
    # Strategy: AI-driven with low thresholds (dynamic risk will filter)
    'strategy_config': {
        'min_confidence': 0.30,          # Accept trades from 30% confidence
        'min_move_pct': 0.10,            # Min 0.10% predicted move
        'enable_trailing_stop': True,
        'trailing_distance_pips': 50,
        'partial_close_pct': 0.5,        # Close 50% at first target
        'use_time_filter': True,
        'allowed_hours': list(range(7, 20)),  # 7 AM - 8 PM UTC
    },
    
    # Risk: AI-driven dynamic system
    'risk_config': {
        # Account
        'account_balance': 10000.0,
        'max_daily_loss': 4.0,           # $400 max daily loss
        
        # Dynamic risk tiers
        'risk_low_confidence': 0.5,      # 0.3-0.49 conf → $50 risk
        'risk_medium_confidence': 1.0,   # 0.5-0.69 conf → $100 risk
        'risk_high_confidence': 2.0,     # 0.85-1.0 conf → $200 risk
        
        # Dynamic R:R ratios
        'rr_defensive': 0.5,             # Low conf → 1:0.5
        'rr_normal': 1.0,                # Med conf → 1:1
        'rr_strong': 2.0,                # Strong conf → 1:2
        'rr_high': 3.0,                  # High conf → 1:3
        
        # XAUUSD costs
        'pip_value': 1.0,
        'spread_cost_per_microlot': 0.5,
        'commission_per_lot': 6.0,
        
        # Safety
        'max_lot_size': 2.0,
        'max_open_positions': 3,
        'max_leverage': 10.0,
        'max_spread_pips': 25,
        'slippage_tolerance_pips': 10,
        'consecutive_loss_reduction': 0.5,
        'max_consecutive_losses': 2,
    },
    
    # MT5: Leave None to use default connection
    'mt5_config': {
        'login': None,
        'password': None,
        'server': None,
        'magic_number': 123456,
    }
}


# ============================================================================
# CONSERVATIVE PROFILE: Lower risk, more selective
# ============================================================================

CONSERVATIVE_CONFIG = {
    'symbol': 'XAUUSD',
    'timeframe': '15m',
    'update_interval': 60,
    
    'strategy_config': {
        'min_confidence': 0.50,          # Higher threshold
        'min_move_pct': 0.15,
        'enable_trailing_stop': True,
        'trailing_distance_pips': 40,    # Tighter trailing
        'partial_close_pct': 0.5,
        'use_time_filter': True,
        'allowed_hours': list(range(8, 18)),  # Only London/NY overlap
    },
    
    'risk_config': {
        'account_balance': 10000.0,
        'max_daily_loss': 2.0,           # $200 max (2%)
        'risk_low_confidence': 0.25,     # $25 risk
        'risk_medium_confidence': 0.5,   # $50 risk
        'risk_high_confidence': 1.0,     # $100 risk
        'rr_defensive': 0.5,
        'rr_normal': 1.0,
        'rr_strong': 1.5,                # Less aggressive
        'rr_high': 2.0,
        'pip_value': 1.0,
        'spread_cost_per_microlot': 0.5,
        'commission_per_lot': 6.0,
        'max_lot_size': 1.0,             # Cap at 1.0 lot
        'max_open_positions': 2,         # Max 2 positions
        'max_leverage': 5.0,
        'max_spread_pips': 20,           # Stricter spread filter
        'slippage_tolerance_pips': 10,
        'consecutive_loss_reduction': 0.5,
        'max_consecutive_losses': 2,
    },
    
    'mt5_config': {
        'login': None,
        'password': None,
        'server': None,
        'magic_number': 123456,
    }
}


# ============================================================================
# AGGRESSIVE PROFILE: Higher risk, more trades
# ============================================================================

AGGRESSIVE_CONFIG = {
    'symbol': 'XAUUSD',
    'timeframe': '15m',
    'update_interval': 60,
    
    'strategy_config': {
        'min_confidence': 0.25,          # Very low threshold
        'min_move_pct': 0.08,
        'enable_trailing_stop': True,
        'trailing_distance_pips': 60,    # Wider stops
        'partial_close_pct': 0.5,
        'use_time_filter': False,        # Trade all hours
        'allowed_hours': list(range(0, 24)),
    },
    
    'risk_config': {
        'account_balance': 10000.0,
        'max_daily_loss': 6.0,           # $600 max (6%)
        'risk_low_confidence': 1.0,      # $100 risk
        'risk_medium_confidence': 2.0,   # $200 risk
        'risk_high_confidence': 3.0,     # $300 risk
        'rr_defensive': 0.5,
        'rr_normal': 1.0,
        'rr_strong': 2.5,                # More aggressive
        'rr_high': 4.0,
        'pip_value': 1.0,
        'spread_cost_per_microlot': 0.5,
        'commission_per_lot': 6.0,
        'max_lot_size': 5.0,             # Up to 5.0 lots
        'max_open_positions': 5,         # Max 5 positions
        'max_leverage': 20.0,
        'max_spread_pips': 30,
        'slippage_tolerance_pips': 15,
        'consecutive_loss_reduction': 0.5,
        'max_consecutive_losses': 3,     # Tolerate more losses
    },
    
    'mt5_config': {
        'login': None,
        'password': None,
        'server': None,
        'magic_number': 123456,
    }
}


# ============================================================================
# USAGE
# ============================================================================

"""
To use a profile:

1. Import in bot.py:
   from quick_start_configs import QUICK_START_CONFIG, CONSERVATIVE_CONFIG, AGGRESSIVE_CONFIG

2. Replace config = {...} with:
   config = QUICK_START_CONFIG  # or CONSERVATIVE_CONFIG or AGGRESSIVE_CONFIG

3. Run:
   python bot.py

Or create your own custom config by mixing parameters from the profiles above.
"""


# ============================================================================
# EXPECTED RESULTS BY PROFILE
# ============================================================================

PROFILE_EXPECTATIONS = {
    'CONSERVATIVE': {
        'trades_per_day': '2-4',
        'avg_risk_per_trade': '$25-100',
        'max_daily_loss': '$200',
        'win_rate_target': '60-65%',
        'drawdown': 'Low',
        'suitable_for': 'Risk-averse traders, smaller accounts, beginners',
    },
    
    'BALANCED (Quick Start)': {
        'trades_per_day': '4-8',
        'avg_risk_per_trade': '$50-200',
        'max_daily_loss': '$400',
        'win_rate_target': '55-60%',
        'drawdown': 'Moderate',
        'suitable_for': 'Most traders, $10k+ accounts, balanced approach',
    },
    
    'AGGRESSIVE': {
        'trades_per_day': '8-15',
        'avg_risk_per_trade': '$100-300',
        'max_daily_loss': '$600',
        'win_rate_target': '50-55%',
        'drawdown': 'Higher',
        'suitable_for': 'Experienced traders, larger accounts, high risk tolerance',
    }
}


# ============================================================================
# CUSTOMIZATION GUIDE
# ============================================================================

CUSTOMIZATION_TIPS = """
Key Parameters to Adjust:

1. ACCOUNT SIZE
   - Scale risk_low/medium/high_confidence proportionally
   - Example for $5k account: Divide all risk% by 2

2. RISK TOLERANCE
   - Conservative: Lower max_daily_loss, lower risk percentages
   - Aggressive: Higher max_daily_loss, higher risk percentages

3. TRADE FREQUENCY
   - More trades: Lower min_confidence, lower min_move_pct
   - Fewer trades: Higher min_confidence, higher min_move_pct

4. PROFIT TARGETS
   - Conservative: Lower rr_strong and rr_high
   - Aggressive: Higher rr_strong and rr_high

5. POSITION MANAGEMENT
   - Tight stops: Lower trailing_distance_pips
   - Wide stops: Higher trailing_distance_pips

6. MARKET CONDITIONS
   - High volatility: Increase max_spread_pips
   - Low volatility: Decrease max_spread_pips

TESTING RECOMMENDATIONS:

1. Start with CONSERVATIVE profile on demo account
2. Run for 1-2 weeks and analyze results
3. Adjust parameters gradually
4. Move to BALANCED profile when comfortable
5. Only use AGGRESSIVE with proven results and proper capital
"""

if __name__ == "__main__":
    print("=" * 80)
    print("AI TRADING BOT - CONFIGURATION PROFILES")
    print("=" * 80)
    
    print("\n📋 Available Profiles:\n")
    
    for profile_name, expectations in PROFILE_EXPECTATIONS.items():
        print(f"\n{profile_name}:")
        print("-" * 60)
        for key, value in expectations.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print("\n" + "=" * 80)
    print("💡 CUSTOMIZATION TIPS")
    print("=" * 80)
    print(CUSTOMIZATION_TIPS)
