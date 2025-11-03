"""
Risk Management Module
Position sizing, risk control, and portfolio management
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """Manages trading risk and position sizing with dynamic AI-driven adjustments"""
    
    def __init__(
        self,
        account_balance: float = 10000.0,
        max_daily_loss: float = 4.0,          # % of balance per day = $400
        max_daily_profit: float = 5.0,        # % of balance per day = $500
        risk_low_confidence: float = 0.5,     # Low confidence risk %
        risk_medium_confidence: float = 1.0,  # Medium confidence risk %
        risk_high_confidence: float = 2.0,    # High confidence risk %
        max_lot_size: float = 2.0,            # Max lot size cap
        max_open_positions: int = 2,          # Max concurrent positions
        max_leverage: float = 10.0,
        same_direction_only: bool = True,     # Only allow same direction trades
        rr_defensive: float = 0.5,            # R:R for low confidence (0.3-0.49)
        rr_normal: float = 1.0,               # R:R for medium confidence (0.5-0.69)
        rr_strong: float = 2.0,               # R:R for strong confidence (0.7-0.84)
        rr_high: float = 3.0,                 # R:R for high confidence (0.85-1.0)
        pip_value: float = 1.0,               # $1 per pip per 0.01 lot
        spread_cost_per_microlot: float = 0.5,  # Spread cost per 0.01 lot
        commission_per_lot: float = 6.0,      # Commission per 1.0 lot
        max_spread_pips: float = 25,          # Max acceptable spread
        slippage_tolerance_pips: float = 10,  # Slippage tolerance
        consecutive_loss_reduction: float = 0.5,  # Risk reduction after losses
        max_consecutive_losses: int = 2,      # Trigger for risk reduction
        max_correlation: float = 0.7,
    ):
        """
        Initialize risk manager with dynamic AI-driven parameters
        
        Args:
            account_balance: Starting account balance in USD
            max_daily_loss: Maximum daily loss as % of balance
            max_daily_profit: Maximum daily profit target as % of balance
            risk_low_confidence: Risk % for confidence 0.3-0.49
            risk_medium_confidence: Risk % for confidence 0.5-0.69
            risk_high_confidence: Risk % for confidence 0.85-1.0
            max_lot_size: Maximum lot size cap
            max_open_positions: Maximum concurrent positions
            max_leverage: Maximum leverage allowed
            same_direction_only: Only allow trades in same direction
            rr_defensive: Reward:risk for confidence 0.3-0.49
            rr_normal: Reward:risk for confidence 0.5-0.69
            rr_strong: Reward:risk for confidence 0.7-0.84
            rr_high: Reward:risk for confidence 0.85-1.0
            pip_value: Dollar value per pip per 0.01 lot
            spread_cost_per_microlot: Spread cost per 0.01 lot
            commission_per_lot: Commission per 1.0 lot round-turn
            max_spread_pips: Maximum acceptable spread in pips
            slippage_tolerance_pips: Acceptable slippage in pips
            consecutive_loss_reduction: Risk reduction factor after losses
            max_consecutive_losses: Number of losses to trigger reduction
            max_correlation: Maximum allowed correlation between positions
        """
        self.account_balance = account_balance
        self.max_daily_loss = max_daily_loss
        self.max_daily_profit = max_daily_profit
        self.risk_low_confidence = risk_low_confidence
        self.risk_medium_confidence = risk_medium_confidence
        self.risk_high_confidence = risk_high_confidence
        self.max_lot_size = max_lot_size
        self.max_open_positions = max_open_positions
        self.max_leverage = max_leverage
        self.same_direction_only = same_direction_only
        
        # Dynamic R:R ratios
        self.rr_defensive = rr_defensive
        self.rr_normal = rr_normal
        self.rr_strong = rr_strong
        self.rr_high = rr_high
        
        # XAUUSD specific costs
        self.pip_value = pip_value
        self.spread_cost_per_microlot = spread_cost_per_microlot
        self.commission_per_lot = commission_per_lot
        self.max_spread_pips = max_spread_pips
        self.slippage_tolerance_pips = slippage_tolerance_pips
        
        # Loss protection
        self.consecutive_loss_reduction = consecutive_loss_reduction
        self.max_consecutive_losses = max_consecutive_losses
        self.max_correlation = max_correlation
        
        # Track daily performance
        self.daily_pnl = 0.0
        self.daily_start_balance = None
        self.last_reset_date = None
        self.trades_today = 0
        self.consecutive_losses = 0
        self.risk_reduction_active = False
        
        logger.info(
            f"Risk manager initialized: Account=${account_balance:.2f}, "
            f"Max daily loss={max_daily_loss}% (${account_balance * max_daily_loss / 100:.2f}), "
            f"Max daily profit={max_daily_profit}% (${account_balance * max_daily_profit / 100:.2f})"
        )
        logger.info(
            f"Dynamic risk: Low={risk_low_confidence}%, Med={risk_medium_confidence}%, High={risk_high_confidence}%"
        )
        logger.info(
            f"Dynamic R:R: Defensive={rr_defensive}, Normal={rr_normal}, Strong={rr_strong}, High={rr_high}"
        )
        logger.info(
            f"Position limits: Max {max_open_positions} positions, Same direction only: {same_direction_only}"
        )
    
    def is_trading_hours_valid(self) -> Dict:
        """
        Check if current time is within valid trading hours
        Skips first 2 hours after market open and last 1 hour before close
        
        Gold/Forex Market Hours (GMT):
        - Opens: Sunday 23:00 GMT
        - Closes: Friday 22:00 GMT
        - Avoid: Sunday 23:00 - Monday 01:00 (first 2 hours)
        - Avoid: Friday 21:00 - 22:00 (last 1 hour)
        
        Returns:
            Dict with 'ok': bool, 'reason': str
        """
        now = datetime.now()
        current_hour = now.hour
        current_minute = now.minute
        current_day = now.weekday()  # 0=Monday, 4=Friday, 5=Saturday, 6=Sunday
        
        # Weekend - Market is closed
        if current_day == 5:  # Saturday - Full day closed
            return {
                'ok': False,
                'reason': "MARKET CLOSED - WAIT FOR MARKET TO OPEN (Saturday)"
            }
        
        if current_day == 6:  # Sunday - Opens at 23:00
            if current_hour < 23:
                return {
                    'ok': False,
                    'reason': "MARKET CLOSED - WAIT FOR MARKET TO OPEN (Sunday before 23:00)"
                }
            # Sunday 23:00+ is market open, but we skip first 2 hours
            # So we need to wait until Monday 01:00
            # This means NO trading on Sunday at all
            return {
                'ok': False,
                'reason': "MARKET CLOSED - WAIT FOR MARKET TO OPEN (2 hours after Sunday open)"
            }
        
        # Monday - Skip first 2 hours after Sunday open (avoid 00:00-00:59)
        if current_day == 0:  # Monday
            if current_hour == 0:  # Midnight hour (00:00-00:59)
                return {
                    'ok': False,
                    'reason': "MARKET CLOSED - WAIT FOR MARKET TO OPEN (2 hours after market open)"
                }
        
        # Friday - Stop 1 hour before close (avoid 21:00+)
        if current_day == 4:  # Friday
            if current_hour >= 21:  # 21:00 onwards
                return {
                    'ok': False,
                    'reason': "MARKET CLOSED - WAIT FOR MARKET TO OPEN (1 hour before Friday close)"
                }
        
        # All other times are valid
        return {'ok': True, 'reason': 'Trading hours valid'}
    
    def get_dynamic_risk_percent(self, confidence: float) -> float:
        """
        Get risk percentage based on AI confidence score
        
        Args:
            confidence: AI confidence score (0-1)
        
        Returns:
            Risk percentage to use for this trade
        """
        # Apply consecutive loss reduction if active
        reduction_factor = 1.0
        if self.risk_reduction_active:
            reduction_factor = self.consecutive_loss_reduction
            logger.info(f"Risk reduction active: {reduction_factor}x multiplier")
        
        # Map confidence to risk percentage
        if confidence < 0.5:
            # Low confidence: 0.3-0.49 → 0.5% risk
            risk_pct = self.risk_low_confidence
        elif confidence < 0.7:
            # Medium confidence: 0.5-0.69 → 1% risk
            risk_pct = self.risk_medium_confidence
        elif confidence < 0.85:
            # Strong confidence: 0.7-0.84 → interpolate between 1% and 2%
            # Linear interpolation from 1% to 2%
            t = (confidence - 0.7) / (0.85 - 0.7)
            risk_pct = self.risk_medium_confidence + t * (self.risk_high_confidence - self.risk_medium_confidence)
        else:
            # High confidence: 0.85-1.0 → 2% risk
            risk_pct = self.risk_high_confidence
        
        # Apply reduction factor
        risk_pct *= reduction_factor
        
        logger.info(f"Confidence {confidence:.2f} → Risk {risk_pct:.2f}%")
        return risk_pct
    
    def get_dynamic_rr_ratio(self, confidence: float) -> float:
        """
        Get reward:risk ratio based on AI confidence score
        
        Args:
            confidence: AI confidence score (0-1)
        
        Returns:
            Reward:risk ratio to use
        """
        if confidence < 0.5:
            # Defensive mode: 0.3-0.49 → 1:0.5
            rr = self.rr_defensive
        elif confidence < 0.7:
            # Normal: 0.5-0.69 → 1:1
            rr = self.rr_normal
        elif confidence < 0.85:
            # Strong confidence: 0.7-0.84 → 1:2
            rr = self.rr_strong
        else:
            # High conviction: 0.85-1.0 → 1:3
            rr = self.rr_high
        
        logger.info(f"Confidence {confidence:.2f} → R:R 1:{rr}")
        return rr
    
    def reset_daily_stats(self, current_balance: float):
        """Reset daily statistics"""
        today = datetime.now().date()
        if self.last_reset_date != today:
            self.daily_pnl = 0.0
            self.daily_start_balance = current_balance
            self.last_reset_date = today
            self.trades_today = 0
            logger.info(f"Daily stats reset - Starting balance: ${current_balance:.2f}")

    
    def check_daily_risk_limit(self, current_balance: float) -> Dict:
        """
        Check if daily risk limit has been exceeded
        
        Returns:
            Dict with:
            {
                'ok': bool,
                'reason': str,
                'daily_loss': float,
                'daily_loss_pct': float
            }
        """
        if self.daily_start_balance is None:
            self.daily_start_balance = current_balance
            return {
                'ok': True,
                'reason': "First check of the day",
                'daily_loss': 0.0,
                'daily_loss_pct': 0.0
            }
        
        # Calculate daily loss (including floating + realized)
        daily_loss = self.daily_start_balance - current_balance
        daily_loss_pct = (daily_loss / self.daily_start_balance) * 100
        
        # Calculate daily profit
        daily_profit = current_balance - self.daily_start_balance  
        daily_profit_pct = (daily_profit / self.daily_start_balance) * 100
        
        # Check if exceeded loss limit
        max_loss_amount = self.daily_start_balance * (self.max_daily_loss / 100)
        
        if daily_loss >= max_loss_amount:
            logger.error(
                f"🚨 DAILY LOSS LIMIT HIT: ${daily_loss:.2f} ({daily_loss_pct:.2f}%) "
                f">= ${max_loss_amount:.2f} ({self.max_daily_loss}%)"
            )
            return {
                'ok': False,
                'reason': f"Daily loss limit exceeded: ${daily_loss:.2f} >= ${max_loss_amount:.2f}",
                'daily_loss': daily_loss,
                'daily_loss_pct': daily_loss_pct,
                'daily_profit': daily_profit,
                'daily_profit_pct': daily_profit_pct
            }
        
        # Check if hit profit target
        max_profit_amount = self.daily_start_balance * (self.max_daily_profit / 100)
        
        if daily_profit >= max_profit_amount:
            logger.info(
                f"🎯 DAILY PROFIT TARGET HIT: ${daily_profit:.2f} ({daily_profit_pct:.2f}%) "
                f">= ${max_profit_amount:.2f} ({self.max_daily_profit}%)"
            )
            return {
                'ok': False,
                'reason': f"Daily profit target reached: ${daily_profit:.2f} >= ${max_profit_amount:.2f}",
                'daily_loss': daily_loss,
                'daily_loss_pct': daily_loss_pct,
                'daily_profit': daily_profit,
                'daily_profit_pct': daily_profit_pct,
                'profit_target_hit': True
            }
        
        return {
            'ok': True,
            'reason': f"Within daily limits: Loss ${daily_loss:.2f}, Profit ${daily_profit:.2f}",
            'daily_loss': daily_loss,
            'daily_loss_pct': daily_loss_pct,
            'daily_profit': daily_profit,
            'daily_profit_pct': daily_profit_pct
        }
    
    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        confidence: float,
        symbol_info: Dict
    ) -> Dict:
        """
        Calculate optimal position size based on dynamic risk and AI confidence
        
        Formula: Lot Size = Risk Amount / (StopLoss in pips * PipValue)
        
        Args:
            account_balance: Current account balance
            entry_price: Intended entry price
            stop_loss: Stop loss price
            confidence: AI confidence score (0-1)
            symbol_info: Symbol information from MT5
        
        Returns:
            Dict with position sizing details:
            {
                'volume': float,               # Position size in lots
                'risk_amount': float,          # Gross risk in account currency
                'total_cost': float,           # Including spread + commission
                'risk_percent': float,         # Risk as % of balance
                'stop_loss_pips': float,       # SL distance in pips
                'confidence': float,           # AI confidence used
                'ok': bool,
                'reason': str
            }
        """
        try:
            # Get dynamic risk percentage based on confidence
            risk_pct = self.get_dynamic_risk_percent(confidence)
            
            # Calculate risk amount
            risk_amount = account_balance * (risk_pct / 100)
            
            # Calculate stop loss distance
            sl_distance = abs(entry_price - stop_loss)
            
            if sl_distance == 0:
                return {
                    'ok': False,
                    'reason': "Stop loss cannot equal entry price"
                }
            
            # Get symbol parameters
            point = symbol_info['point']
            contract_size = symbol_info['contract_size']
            volume_min = symbol_info['volume_min']
            volume_max = symbol_info['volume_max']
            volume_step = symbol_info['volume_step']
            
            # Calculate stop loss in pips (1 pip = 10 points for XAUUSD)
            sl_pips = sl_distance / (point * 10)
            
            # Calculate position size using formula:
            # Lot Size = Risk Amount / (StopLoss in pips * PipValue)
            # For XAUUSD: pip_value = $1 per 0.01 lot
            volume = risk_amount / (sl_pips * self.pip_value)
            
            # Round to volume step
            volume = round(volume / volume_step) * volume_step
            
            # Enforce max lot size cap
            if volume > self.max_lot_size:
                logger.warning(f"Volume {volume:.2f} exceeds max {self.max_lot_size}, capping")
                volume = self.max_lot_size
            
            # Enforce broker limits
            volume = max(volume_min, min(volume_max, volume))
            
            # Calculate costs
            spread_cost = volume * 100 * self.spread_cost_per_microlot  # Convert lots to micro lots
            commission_cost = volume * self.commission_per_lot
            total_cost = spread_cost + commission_cost
            
            # Calculate actual risk with rounded volume
            actual_risk = (volume * sl_pips * self.pip_value) + total_cost
            actual_risk_pct = (actual_risk / account_balance) * 100
            
            result = {
                'volume': volume,
                'risk_amount': volume * sl_pips * self.pip_value,  # Gross risk
                'total_cost': total_cost,
                'risk_percent': actual_risk_pct,
                'stop_loss_pips': sl_pips,
                'confidence': confidence,
                'spread_cost': spread_cost,
                'commission_cost': commission_cost,
                'ok': True,
                'reason': None
            }
            
            logger.info(
                f"Position size: {volume:.2f} lots | "
                f"Risk: ${actual_risk:.2f} ({actual_risk_pct:.2f}%) | "
                f"SL: {sl_pips:.1f} pips | "
                f"Costs: ${total_cost:.2f} (spread ${spread_cost:.2f} + comm ${commission_cost:.2f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {
                'ok': False,
                'reason': str(e)
            }
    
    def validate_spread(self, current_spread_pips: float) -> Dict:
        """
        Validate if spread is acceptable for trading
        
        Args:
            current_spread_pips: Current spread in pips
        
        Returns:
            Dict with validation result
        """
        if current_spread_pips > self.max_spread_pips:
            logger.warning(
                f"Spread too wide: {current_spread_pips:.1f} pips > {self.max_spread_pips} pips"
            )
            return {
                'ok': False,
                'reason': f"Spread {current_spread_pips:.1f} pips exceeds max {self.max_spread_pips} pips"
            }
        
        return {
            'ok': True,
            'reason': f"Spread acceptable: {current_spread_pips:.1f} pips"
        }
    
    def validate_trade(
        self,
        signal: Dict,
        account_info: Dict,
        open_positions: list,
        current_spread_pips: float = 0
    ) -> Dict:
        """
        Validate if trade should be taken based on risk rules
        
        Args:
            signal: Trading signal from strategy
            account_info: Account information
            open_positions: List of currently open positions
            current_spread_pips: Current market spread in pips
        
        Returns:
            Dict with validation result:
            {
                'approved': bool,
                'reason': str,
                'position_size': float,    # If approved
                'risk_amount': float       # If approved
            }
        """
        # Reset daily stats if new day
        self.reset_daily_stats(account_info['balance'])
        
        # Check if we should trade at all
        if signal['signal'] == 'HOLD':
            return {
                'approved': False,
                'reason': "No trading signal"
            }
        
        # Check market hours (skip first 2 hours and last 1 hour)
        hours_check = self.is_trading_hours_valid()
        if not hours_check['ok']:
            return {
                'approved': False,
                'reason': hours_check['reason']
            }
        
        # Check daily risk limit (with equity protection)
        daily_check = self.check_daily_risk_limit(account_info['equity'])
        if not daily_check['ok']:
            return {
                'approved': False,
                'reason': daily_check['reason']
            }
        
        # Check spread filter
        if current_spread_pips > 0:
            spread_check = self.validate_spread(current_spread_pips)
            if not spread_check['ok']:
                return {
                    'approved': False,
                    'reason': spread_check['reason']
                }
        
        # Check maximum open positions
        if len(open_positions) >= self.max_open_positions:
            return {
                'approved': False,
                'reason': f"Max positions reached: {len(open_positions)}/{self.max_open_positions}"
            }
        
        # Check same direction requirement
        if self.same_direction_only and len(open_positions) > 0:
            # Get direction of existing positions
            existing_direction = open_positions[0]['type']  # 'BUY' or 'SELL'
            new_direction = signal['signal']  # 'BUY' or 'SELL'
            
            if existing_direction != new_direction:
                return {
                    'approved': False,
                    'reason': f"Direction mismatch: existing={existing_direction}, new={new_direction}. Close existing trades first."
                }
        
        # Get dynamic R:R based on confidence
        confidence = signal.get('confidence', 0)
        min_rr = self.get_dynamic_rr_ratio(confidence)
        
        # Check reward:risk ratio against dynamic threshold
        if signal.get('reward_risk_ratio', 0) < min_rr:
            return {
                'approved': False,
                'reason': f"R:R {signal.get('reward_risk_ratio', 0):.2f} < {min_rr} (conf={confidence:.2f})"
            }
        
        # Check confidence threshold
        if confidence < 0.30:  # Minimum absolute confidence
            return {
                'approved': False,
                'reason': f"Confidence too low: {confidence:.2f} < 0.30"
            }
        
        # Check leverage
        current_leverage = account_info['equity'] / account_info['balance'] if account_info['balance'] > 0 else 1.0
        if current_leverage >= self.max_leverage:
            return {
                'approved': False,
                'reason': f"Max leverage reached: {current_leverage:.1f}x"
            }
        
        # All checks passed
        return {
            'approved': True,
            'reason': "All risk checks passed"
        }
    
    def should_close_opposite_positions(
        self,
        open_positions: list,
        new_signal_direction: str
    ) -> Dict:
        """
        Check if opposite direction positions should be closed before opening new trade
        
        Args:
            open_positions: List of currently open positions
            new_signal_direction: Direction of new signal ('BUY' or 'SELL')
        
        Returns:
            Dict with:
            {
                'should_close': bool,
                'positions_to_close': list,  # List of ticket numbers
                'reason': str
            }
        """
        if not self.same_direction_only or len(open_positions) == 0:
            return {
                'should_close': False,
                'positions_to_close': [],
                'reason': "No opposite positions"
            }
        
        # Check if any position is opposite direction
        opposite_positions = []
        for pos in open_positions:
            if pos['type'] != new_signal_direction:
                opposite_positions.append(pos['ticket'])
        
        if opposite_positions:
            return {
                'should_close': True,
                'positions_to_close': opposite_positions,
                'reason': f"Close {len(opposite_positions)} opposite {open_positions[0]['type']} position(s) before opening {new_signal_direction}"
            }
        
        return {
            'should_close': False,
            'positions_to_close': [],
            'reason': "All positions same direction"
        }
    
    def should_close_position(
        self,
        position: Dict,
        current_price: float,
        prediction: Dict
    ) -> Dict:
        """
        Determine if an open position should be closed
        
        Args:
            position: Open position details
            current_price: Current market price
            prediction: Latest model prediction
        
        Returns:
            Dict with close decision:
            {
                'should_close': bool,
                'reason': str
            }
        """
        pos_type = position['type']
        entry_price = position['price_open']
        sl = position['sl']
        tp = position['tp']
        profit = position['profit']
        
        # Check if prediction changed direction - SMART EXIT LOGIC
        pred_direction = prediction.get('direction', 'NEUTRAL')
        pred_confidence = prediction.get('confidence', 0)
        
        # Only close on reversal if trade is losing or has small profit
        if pos_type == 'BUY' and pred_direction == 'DOWN':
            # Cut losses immediately
            if profit < 0:
                return {
                    'should_close': True,
                    'reason': f"Prediction reversed to DOWN (cutting loss: ${profit:.2f})"
                }
            # Close small winners with low confidence
            elif profit < 100 and pred_confidence >= 0.65:
                return {
                    'should_close': True,
                    'reason': f"Prediction reversed to DOWN (small profit: ${profit:.2f}, conf: {pred_confidence:.2f})"
                }
            # Keep large winners - let TP/SL work
            else:
                return {
                    'should_close': False,
                    'reason': f"Keeping winner despite reversal (profit: ${profit:.2f})"
                }
        
        if pos_type == 'SELL' and pred_direction == 'UP':
            # Cut losses immediately
            if profit < 0:
                return {
                    'should_close': True,
                    'reason': f"Prediction reversed to UP (cutting loss: ${profit:.2f})"
                }
            # Close small winners with low confidence
            elif profit < 100 and pred_confidence >= 0.65:
                return {
                    'should_close': True,
                    'reason': f"Prediction reversed to UP (small profit: ${profit:.2f}, conf: {pred_confidence:.2f})"
                }
            # Keep large winners - let TP/SL work
            else:
                return {
                    'should_close': False,
                    'reason': f"Keeping winner despite reversal (profit: ${profit:.2f})"
                }
        
        # Check if profit target partially hit (can trail stop)
        if tp and pos_type == 'BUY':
            target_move = tp - entry_price
            current_move = current_price - entry_price
            if current_move > target_move * 0.75:  # 75% to target
                return {
                    'should_close': False,  # Don't close, but could trail SL
                    'reason': "Near profit target (trail stop)"
                }
        
        if tp and pos_type == 'SELL':
            target_move = entry_price - tp
            current_move = entry_price - current_price
            if current_move > target_move * 0.75:
                return {
                    'should_close': False,
                    'reason': "Near profit target (trail stop)"
                }
        
        # Check prediction confidence loss
        if prediction.get('confidence', 1.0) < 0.70:
            return {
                'should_close': True,
                'reason': "Prediction confidence dropped below 70%"
            }
        
        # Hold position
        return {
            'should_close': False,
            'reason': "Position still valid"
        }
    
    def update_daily_pnl(self, trade_pnl: float, was_winner: bool):
        """
        Update daily P&L tracking and consecutive loss counter
        
        Args:
            trade_pnl: Profit/loss from closed trade
            was_winner: True if trade was profitable
        """
        self.daily_pnl += trade_pnl
        self.trades_today += 1
        
        # Track consecutive losses
        if was_winner:
            # Reset consecutive losses on win
            if self.consecutive_losses > 0:
                logger.info(f"✅ Win breaks losing streak of {self.consecutive_losses}")
            self.consecutive_losses = 0
            self.risk_reduction_active = False
        else:
            # Increment consecutive losses
            self.consecutive_losses += 1
            logger.warning(f"❌ Loss #{self.consecutive_losses} in a row")
            
            # Activate risk reduction if threshold hit
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.risk_reduction_active = True
                logger.warning(
                    f"🚨 RISK REDUCTION ACTIVE: {self.consecutive_losses} consecutive losses. "
                    f"Reducing risk by {(1 - self.consecutive_loss_reduction) * 100:.0f}%"
                )
        
        logger.info(f"Daily P&L: ${self.daily_pnl:.2f} | Trades today: {self.trades_today}")
    
    def close_all_positions_for_protection(self, mt5_connector) -> int:
        """
        Close all open positions when daily loss limit is hit
        
        Args:
            mt5_connector: MT5 connection instance
        
        Returns:
            Number of positions closed
        """
        logger.error("🚨 CLOSING ALL POSITIONS - DAILY LOSS LIMIT HIT")
        
        positions = mt5_connector.get_open_positions()
        closed_count = 0
        
        for pos in positions:
            ticket = pos['ticket']
            logger.info(f"Closing position #{ticket}...")
            
            result = mt5_connector.close_position(ticket)
            if result:
                closed_count += 1
                logger.info(f"✓ Position #{ticket} closed")
            else:
                logger.error(f"✗ Failed to close position #{ticket}")
        
        logger.info(f"Closed {closed_count}/{len(positions)} positions")
        return closed_count
    
    def get_risk_report(self, account_info: Dict, open_positions: list) -> Dict:
        """Generate risk status report"""
        balance = account_info['balance']
        equity = account_info['equity']
        
        # Calculate total exposure
        total_exposure = sum(
            pos['volume'] * pos['price_current'] 
            for pos in open_positions
        )
        
        # Calculate current leverage
        current_leverage = total_exposure / equity if equity > 0 else 0
        
        # Calculate daily risk usage
        if self.daily_start_balance:
            daily_loss = self.daily_start_balance - balance
            daily_risk_used = (daily_loss / self.daily_start_balance) * 100
        else:
            daily_risk_used = 0
        
        return {
            'open_positions': len(open_positions),
            'max_positions': self.max_open_positions,
            'current_leverage': current_leverage,
            'max_leverage': self.max_leverage,
            'daily_risk_used': daily_risk_used,
            'daily_risk_limit': self.max_daily_risk,
            'daily_pnl': self.daily_pnl,
            'trades_today': self.trades_today,
            'can_trade': (
                len(open_positions) < self.max_open_positions and
                daily_risk_used < self.max_daily_risk and
                current_leverage < self.max_leverage
            )
        }
