"""
Trading Strategy Module
Combines model predictions with trading rules and position management
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TradingStrategy:
    """Main trading strategy using TFT model predictions"""
    
    def __init__(
        self,
        min_confidence: float = 0.30,         # Minimum prediction confidence (lowered for dynamic risk)
        min_move_pct: float = 0.10,           # Minimum predicted move %
        enable_trailing_stop: bool = True,    # Enable trailing stops
        trailing_distance_pips: float = 50,   # Trailing stop distance
        partial_close_pct: float = 0.5,       # Partial close at TP1 (50%)
        use_time_filter: bool = True,         # Enable time-of-day filter
        allowed_hours: List[int] = None,      # Trading hours (UTC)
        risk_manager = None,                  # RiskManager instance for dynamic R:R
    ):
        """
        Initialize trading strategy
        
        Args:
            min_confidence: Minimum model confidence to take trade (dynamic risk adjusts)
            min_move_pct: Minimum predicted price move percentage
            enable_trailing_stop: Whether to use trailing stops
            trailing_distance_pips: Distance in pips for trailing stop
            partial_close_pct: Percentage of position to close at first target
            use_time_filter: Filter trades by time of day
            allowed_hours: List of allowed trading hours in UTC (e.g., [7, 8, 9, ..., 16])
            risk_manager: RiskManager instance for dynamic R:R calculation
        """
        self.min_confidence = min_confidence
        self.min_move_pct = min_move_pct
        self.enable_trailing_stop = enable_trailing_stop
        self.trailing_distance_pips = trailing_distance_pips
        self.partial_close_pct = partial_close_pct
        self.use_time_filter = use_time_filter
        self.risk_manager = risk_manager
        
        # Default trading hours (London + NY session overlap)
        if allowed_hours is None:
            allowed_hours = list(range(7, 20))  # 7 AM - 8 PM UTC
        self.allowed_hours = allowed_hours
        
        # Track recent signals to avoid over-trading
        self.last_signal_time = None
        self.min_signal_interval = timedelta(minutes=15)  # Min time between signals
        
        logger.info(
            f"Strategy initialized: conf>={min_confidence}, move>={min_move_pct}%, "
            f"trailing={enable_trailing_stop}, dynamic_rr={risk_manager is not None}"
        )
    
    def generate_signal(self, prediction: Dict) -> Dict:
        """
        Generate trading signal from model prediction
        
        Args:
            prediction: Model prediction dict with:
                - direction: 'UP', 'DOWN', or 'NEUTRAL'
                - confidence: float 0-1
                - current_price: float
                - q10, q50, q90: quantile predictions
                - move_pct: predicted move percentage
        
        Returns:
            Trading signal dict:
            {
                'signal': 'BUY', 'SELL', or 'HOLD',
                'entry_price': float,
                'stop_loss': float,
                'take_profit': float,
                'take_profit_2': float,        # Second target
                'confidence': float,
                'reward_risk_ratio': float,
                'reason': str
            }
        """
        # Check time filter
        if self.use_time_filter and not self._is_trading_hours():
            return {
                'signal': 'HOLD',
                'reason': "Outside trading hours"
            }
        
        # Check signal interval
        if self.last_signal_time:
            time_since_last = datetime.now() - self.last_signal_time
            if time_since_last < self.min_signal_interval:
                return {
                    'signal': 'HOLD',
                    'reason': f"Too soon after last signal ({time_since_last.seconds}s)"
                }
        
        # Check prediction confidence
        confidence = prediction.get('confidence', 0)
        if confidence < self.min_confidence:
            return {
                'signal': 'HOLD',
                'reason': f"Low confidence: {confidence:.2f} < {self.min_confidence}"
            }
        
        # Check predicted move size
        move_pct = abs(prediction.get('move_pct', 0))
        if move_pct < self.min_move_pct:
            return {
                'signal': 'HOLD',
                'reason': f"Small predicted move: {move_pct:.2f}% < {self.min_move_pct}%"
            }
        
        # Get direction
        direction = prediction.get('direction', 'NEUTRAL')
        if direction == 'NEUTRAL':
            return {
                'signal': 'HOLD',
                'reason': "Neutral prediction"
            }
        
        # Get quantile predictions
        q10 = prediction['q10']
        q50 = prediction['q50']
        q90 = prediction['q90']
        current_price = prediction['current_price']
        
        # Get dynamic R:R ratio if risk manager is available
        if self.risk_manager:
            rr_ratio = self.risk_manager.get_dynamic_rr_ratio(confidence)
        else:
            # Fallback to basic logic
            if confidence >= 0.85:
                rr_ratio = 3.0
            elif confidence >= 0.7:
                rr_ratio = 2.0
            elif confidence >= 0.5:
                rr_ratio = 1.0
            else:
                rr_ratio = 0.5
        
        # Generate BUY signal
        if direction == 'UP':
            entry = current_price
            stop_loss = q10  # Conservative stop at 10th percentile
            
            # Calculate take profit based on dynamic R:R
            risk = entry - stop_loss
            if risk <= 0:
                return {
                    'signal': 'HOLD',
                    'reason': "Invalid stop loss (not below entry)"
                }
            
            # Dynamic TP: TP = Entry + (Risk × R:R)
            take_profit = entry + (risk * rr_ratio)
            take_profit_2 = take_profit + risk  # Extended target
            
            # Actual R:R achieved
            reward = take_profit - entry
            actual_rr = reward / risk
            
            # Valid BUY signal
            self.last_signal_time = datetime.now()
            
            return {
                'signal': 'BUY',
                'entry_price': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'take_profit_2': take_profit_2,
                'confidence': confidence,
                'reward_risk_ratio': actual_rr,
                'reason': f"UP prediction: {move_pct:.2f}% move, {confidence:.2f} confidence, 1:{rr_ratio} R:R"
            }
        
        # Generate SELL signal
        elif direction == 'DOWN':
            entry = current_price
            stop_loss = q90  # Conservative stop at 90th percentile
            
            # Calculate take profit based on dynamic R:R
            risk = stop_loss - entry
            if risk <= 0:
                return {
                    'signal': 'HOLD',
                    'reason': "Invalid stop loss (not above entry)"
                }
            
            # Dynamic TP: TP = Entry - (Risk × R:R)
            take_profit = entry - (risk * rr_ratio)
            take_profit_2 = take_profit - risk  # Extended target
            
            # Actual R:R achieved
            reward = entry - take_profit
            actual_rr = reward / risk
            
            # Valid SELL signal
            self.last_signal_time = datetime.now()
            
            return {
                'signal': 'SELL',
                'entry_price': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'take_profit_2': take_profit_2,
                'confidence': confidence,
                'reward_risk_ratio': actual_rr,
                'reason': f"DOWN prediction: {move_pct:.2f}% move, {confidence:.2f} confidence, 1:{rr_ratio} R:R"
            }
        
        return {
            'signal': 'HOLD',
            'reason': "Unknown direction"
        }
    
    def manage_position(
        self,
        position: Dict,
        current_price: float,
        prediction: Dict
    ) -> Dict:
        """
        Manage existing position (trailing stop, partial close, etc.)
        
        Args:
            position: Open position details
            current_price: Current market price
            prediction: Latest model prediction
        
        Returns:
            Management action dict:
            {
                'action': 'HOLD', 'CLOSE_FULL', 'CLOSE_PARTIAL', 'TRAIL_STOP',
                'new_stop_loss': float,        # If action is TRAIL_STOP
                'close_volume': float,         # If action is CLOSE_PARTIAL
                'reason': str
            }
        """
        pos_type = position['type']
        entry_price = position['price_open']
        current_sl = position['sl']
        current_tp = position['tp']
        volume = position['volume']
        profit = position['profit']
        
        # Check if prediction reversed - SMART EXIT LOGIC
        pred_direction = prediction.get('direction', 'NEUTRAL')
        pred_confidence = prediction.get('confidence', 0)
        
        if pos_type == 'BUY' and pred_direction == 'DOWN':
            # Cut losses immediately
            if profit < 0:
                return {
                    'action': 'CLOSE_FULL',
                    'reason': f"Prediction reversed to DOWN (cutting loss: ${profit:.2f})"
                }
            # Close small winners with high confidence reversal
            elif profit < 100 and pred_confidence >= 0.65:
                return {
                    'action': 'CLOSE_FULL',
                    'reason': f"Prediction reversed to DOWN (small profit: ${profit:.2f})"
                }
            # Keep large winners
            else:
                return {
                    'action': 'HOLD',
                    'reason': f"Keeping winner despite reversal (${profit:.2f})"
                }
        
        if pos_type == 'SELL' and pred_direction == 'UP':
            # Cut losses immediately
            if profit < 0:
                return {
                    'action': 'CLOSE_FULL',
                    'reason': f"Prediction reversed to UP (cutting loss: ${profit:.2f})"
                }
            # Close small winners with high confidence reversal
            elif profit < 100 and pred_confidence >= 0.65:
                return {
                    'action': 'CLOSE_FULL',
                    'reason': f"Prediction reversed to UP (small profit: ${profit:.2f})"
                }
            # Keep large winners
            else:
                return {
                    'action': 'HOLD',
                    'reason': f"Keeping winner despite reversal (${profit:.2f})"
                }
        
        # Check if first target reached (partial close)
        if pos_type == 'BUY':
            if current_price >= current_tp and not position.get('tp1_closed', False):
                return {
                    'action': 'CLOSE_PARTIAL',
                    'close_volume': volume * self.partial_close_pct,
                    'reason': f"First target reached: {current_price:.2f} >= {current_tp:.2f}"
                }
            
            # Trailing stop logic
            if self.enable_trailing_stop and profit > 0:
                # Calculate new trailing stop
                trailing_sl = current_price - (self.trailing_distance_pips * 0.0001)  # Assuming 4 decimal places
                
                # Only trail if new SL is higher than current
                if trailing_sl > current_sl:
                    return {
                        'action': 'TRAIL_STOP',
                        'new_stop_loss': trailing_sl,
                        'reason': f"Trailing stop: {current_sl:.5f} -> {trailing_sl:.5f}"
                    }
        
        elif pos_type == 'SELL':
            if current_price <= current_tp and not position.get('tp1_closed', False):
                return {
                    'action': 'CLOSE_PARTIAL',
                    'close_volume': volume * self.partial_close_pct,
                    'reason': f"First target reached: {current_price:.2f} <= {current_tp:.2f}"
                }
            
            # Trailing stop logic
            if self.enable_trailing_stop and profit > 0:
                # Calculate new trailing stop
                trailing_sl = current_price + (self.trailing_distance_pips * 0.0001)
                
                # Only trail if new SL is lower than current
                if trailing_sl < current_sl:
                    return {
                        'action': 'TRAIL_STOP',
                        'new_stop_loss': trailing_sl,
                        'reason': f"Trailing stop: {current_sl:.5f} -> {trailing_sl:.5f}"
                    }
        
        # Hold position
        return {
            'action': 'HOLD',
            'reason': "Position management: no action needed"
        }
    
    def _is_trading_hours(self) -> bool:
        """Check if current time is within allowed trading hours"""
        current_hour = datetime.utcnow().hour
        return current_hour in self.allowed_hours
    
    def get_market_regime(self, market_data: pd.DataFrame) -> str:
        """
        Determine current market regime (trending/ranging/volatile)
        
        Args:
            market_data: DataFrame with OHLCV data
        
        Returns:
            Market regime: 'trending_up', 'trending_down', 'ranging', 'volatile'
        """
        if len(market_data) < 50:
            return 'unknown'
        
        # Calculate indicators for regime detection
        close = market_data['close'].values
        
        # Trend: 20-period SMA slope
        sma_20 = pd.Series(close).rolling(20).mean()
        sma_slope = (sma_20.iloc[-1] - sma_20.iloc[-10]) / sma_20.iloc[-10]
        
        # Volatility: 20-period ATR
        high = market_data['high'].values
        low = market_data['low'].values
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
        atr_20 = pd.Series(tr).rolling(20).mean().iloc[-1]
        avg_atr = pd.Series(tr).rolling(50).mean().iloc[-1]
        
        # Range: Price deviation from mean
        price_range = (close.max() - close.min()) / close.mean()
        
        # Determine regime
        if sma_slope > 0.02:
            return 'trending_up'
        elif sma_slope < -0.02:
            return 'trending_down'
        elif atr_20 > avg_atr * 1.5:
            return 'volatile'
        else:
            return 'ranging'
    
    def adjust_parameters_for_regime(self, regime: str):
        """
        Adjust strategy parameters based on market regime
        
        Args:
            regime: Market regime from get_market_regime()
        """
        if regime == 'trending_up' or regime == 'trending_down':
            # In trending markets, be more aggressive
            self.min_confidence = 0.85
            self.min_move_pct = 0.12
            logger.info(f"Adjusted for {regime}: confidence={self.min_confidence}, move={self.min_move_pct}%")
        
        elif regime == 'ranging':
            # In ranging markets, be more selective
            self.min_confidence = 0.92
            self.min_move_pct = 0.18
            logger.info(f"Adjusted for {regime}: confidence={self.min_confidence}, move={self.min_move_pct}%")
        
        elif regime == 'volatile':
            # In volatile markets, widen stops and be cautious
            self.min_confidence = 0.93
            self.min_move_pct = 0.20
            self.trailing_distance_pips = 75  # Wider trailing stop
            logger.info(f"Adjusted for {regime}: confidence={self.min_confidence}, move={self.min_move_pct}%")
