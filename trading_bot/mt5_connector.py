"""
MetaTrader 5 Integration Module
Handles connection, data fetching, and order execution with MT5
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MT5Connection:
    """Manages MetaTrader 5 connection and operations"""
    
    def __init__(self, login: int = None, password: str = None, server: str = None):
        """
        Initialize MT5 connection
        
        Args:
            login: MT5 account login (optional if already logged in)
            password: MT5 account password
            server: MT5 server name
        """
        self.login = login
        self.password = password
        self.server = server
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to MT5 terminal"""
        try:
            # Initialize MT5
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Login if credentials provided
            if self.login and self.password and self.server:
                if not mt5.login(self.login, self.password, self.server):
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
                    mt5.shutdown()
                    return False
            
            # Check connection
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                mt5.shutdown()
                return False
            
            self.connected = True
            logger.info(f"Connected to MT5 - Account: {account_info.login}, Balance: {account_info.balance}")
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")
    
    def initialize_symbol(self, symbol: str) -> bool:
        """
        Initialize and verify symbol is ready for use
        
        Args:
            symbol: Trading symbol (e.g., "XAUUSD")
            
        Returns:
            True if symbol is ready, False otherwise
        """
        if not self.connected:
            logger.error("Not connected to MT5")
            return False
        
        try:
            # Select symbol in Market Watch
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select {symbol}: {mt5.last_error()}")
                return False
            
            # Get symbol info
            info = mt5.symbol_info(symbol)
            if info is None:
                logger.error(f"Symbol {symbol} not found: {mt5.last_error()}")
                return False
            
            # Check if visible
            if not info.visible:
                logger.warning(f"{symbol} not visible, attempting to show...")
                if not mt5.symbol_select(symbol, True):
                    logger.error(f"Failed to make {symbol} visible")
                    return False
                # Recheck
                info = mt5.symbol_info(symbol)
                if info is None or not info.visible:
                    logger.error(f"{symbol} still not visible")
                    return False
            
            # Verify we can get quotes
            if info.bid == 0 or info.ask == 0:
                logger.warning(f"{symbol} has no quotes yet (bid={info.bid}, ask={info.ask})")
                # Wait a moment for quotes
                import time
                time.sleep(1)
                info = mt5.symbol_info(symbol)
                if info is None or info.bid == 0:
                    logger.error(f"{symbol} still has no quotes")
                    return False
            
            logger.info(f"Symbol {symbol} initialized: Bid={info.bid:.2f}, Ask={info.ask:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing symbol {symbol}: {e}")
            return False
    
    def get_account_info(self) -> Optional[Dict]:
        """Get account information"""
        if not self.connected:
            logger.error("Not connected to MT5")
            return None
        
        info = mt5.account_info()
        if info is None:
            return None
        
        return {
            'login': info.login,
            'balance': info.balance,
            'equity': info.equity,
            'margin': info.margin,
            'free_margin': info.margin_free,
            'margin_level': info.margin_level,
            'profit': info.profit,
            'leverage': info.leverage,
            'currency': info.currency
        }
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        if not self.connected:
            logger.error("Not connected to MT5")
            return None
        
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Symbol {symbol} not found")
            return None
        
        return {
            'symbol': info.name,
            'bid': info.bid,
            'ask': info.ask,
            'spread': info.spread,
            'point': info.point,
            'digits': info.digits,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step,
            'contract_size': info.trade_contract_size
        }
    
    def get_historical_data(
        self, 
        symbol: str, 
        timeframe: int, 
        bars: int = 500
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data from MT5
        
        Args:
            symbol: Trading symbol (e.g., "XAUUSD")
            timeframe: MT5 timeframe constant (e.g., mt5.TIMEFRAME_M15)
            bars: Number of bars to fetch
            
        Returns:
            DataFrame with columns: time, open, high, low, close, tick_volume
        """
        if not self.connected:
            logger.error("Not connected to MT5")
            return None
        
        try:
            # Initialize symbol first (ensures it's ready)
            if not self.initialize_symbol(symbol):
                logger.error(f"Failed to initialize symbol {symbol}")
                return None
            
            # Fetch data
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            
            if rates is None:
                logger.error(f"copy_rates_from_pos returned None: {mt5.last_error()}")
                return None
                
            if len(rates) == 0:
                logger.error(f"No data returned for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            
            # Convert time to datetime with UTC timezone
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            
            # Rename 'time' to 'timestamp' (API expects 'timestamp')
            # Keep 'tick_volume' as-is (API expects 'tick_volume')
            df = df.rename(columns={
                'time': 'timestamp'
            })
            
            # Select required columns for model inference
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'tick_volume']]
            
            # Calculate spread (approximate)
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info:
                df['spread'] = symbol_info['spread'] * symbol_info['point']
            else:
                df['spread'] = 0
            
            logger.info(f"Fetched {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None
    
    def get_latest_bars(
        self, 
        symbol: str, 
        timeframe: int, 
        count: int = 128
    ) -> Optional[pd.DataFrame]:
        """
        Get latest bars for model prediction
        
        Args:
            symbol: Trading symbol
            timeframe: MT5 timeframe
            count: Number of bars (must match model lookback)
            
        Returns:
            DataFrame ready for model inference
        """
        return self.get_historical_data(symbol, timeframe, count)
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current bid/ask prices"""
        if not self.connected:
            return None
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        
        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'spread': tick.ask - tick.bid,
            'time': datetime.fromtimestamp(tick.time)
        }
    
    def calculate_position_size(
        self,
        symbol: str,
        risk_percent: float,
        stop_loss_pips: float,
        account_balance: float = None
    ) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            symbol: Trading symbol
            risk_percent: Risk per trade as percentage (e.g., 1.0 for 1%)
            stop_loss_pips: Stop loss distance in pips
            account_balance: Account balance (if None, fetches current)
            
        Returns:
            Position size in lots
        """
        if not self.connected:
            return 0.0
        
        # Get account balance
        if account_balance is None:
            account_info = self.get_account_info()
            if not account_info:
                return 0.0
            account_balance = account_info['balance']
        
        # Get symbol info
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            return 0.0
        
        # Calculate risk amount in account currency
        risk_amount = account_balance * (risk_percent / 100)
        
        # Convert pips to price
        pip_value = symbol_info['point'] * 10  # Standard pip
        stop_loss_price = stop_loss_pips * pip_value
        
        # Calculate position size
        contract_size = symbol_info['contract_size']
        position_size = risk_amount / (stop_loss_pips * pip_value * contract_size)
        
        # Round to allowed step
        volume_step = symbol_info['volume_step']
        position_size = round(position_size / volume_step) * volume_step
        
        # Enforce min/max limits
        position_size = max(symbol_info['volume_min'], position_size)
        position_size = min(symbol_info['volume_max'], position_size)
        
        return position_size
    
    def send_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: float = None,
        sl: float = None,
        tp: float = None,
        comment: str = "TFT_Bot",
        magic: int = 234000
    ) -> Optional[Dict]:
        """
        Send trading order to MT5
        
        Args:
            symbol: Trading symbol
            order_type: 'BUY' or 'SELL'
            volume: Position size in lots
            price: Entry price (None for market order)
            sl: Stop loss price
            tp: Take profit price
            comment: Order comment
            magic: Magic number for bot identification
            
        Returns:
            Order result dict or None
        """
        if not self.connected:
            logger.error("Not connected to MT5")
            return None
        
        try:
            # Determine order type
            if order_type.upper() == 'BUY':
                trade_type = mt5.ORDER_TYPE_BUY
                if price is None:
                    price = mt5.symbol_info_tick(symbol).ask
            elif order_type.upper() == 'SELL':
                trade_type = mt5.ORDER_TYPE_SELL
                if price is None:
                    price = mt5.symbol_info_tick(symbol).bid
            else:
                logger.error(f"Invalid order type: {order_type}")
                return None
            
            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": trade_type,
                "price": price,
                "deviation": 20,
                "magic": magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add SL/TP if provided
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                logger.error("Order send failed: No result")
                return None
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.retcode} - {result.comment}")
                return None
            
            logger.info(f"Order executed: {order_type} {volume} {symbol} @ {result.price}")
            
            return {
                'retcode': result.retcode,  # Add retcode to return dict
                'order': result.order,
                'volume': result.volume,
                'price': result.price,
                'bid': result.bid,
                'ask': result.ask,
                'comment': result.comment
            }
            
        except Exception as e:
            logger.error(f"Error sending order: {e}")
            return None
    
    def get_open_positions(self, symbol: str = None) -> List[Dict]:
        """Get all open positions"""
        if not self.connected:
            return []
        
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if positions is None:
            return []
        
        result = []
        for pos in positions:
            result.append({
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'price_current': pos.price_current,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit,
                'comment': pos.comment,
                'magic': pos.magic
            })
        
        return result
    
    def close_position(self, ticket: int) -> Dict:
        """
        Close an open position
        
        Returns:
            Dict with close details: success, exit_price, profit, etc.
        """
        if not self.connected:
            return {'success': False, 'reason': 'Not connected to MT5'}
        
        try:
            # Get position info before closing
            position = mt5.positions_get(ticket=ticket)
            if not position or len(position) == 0:
                logger.error(f"Position {ticket} not found")
                return {'success': False, 'reason': 'Position not found'}
            
            pos = position[0]
            
            # Store pre-close details
            entry_price = pos.price_open
            current_profit = pos.profit
            volume = pos.volume
            symbol = pos.symbol
            open_time = datetime.fromtimestamp(pos.time)
            
            # Determine close type (opposite of open)
            close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            # Get current price
            tick = mt5.symbol_info_tick(pos.symbol)
            exit_price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": ticket,
                "price": exit_price,
                "deviation": 20,
                "magic": pos.magic,
                "comment": "Close by TFT_Bot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send close order
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Calculate duration
                close_time = datetime.now()
                duration_minutes = (close_time - open_time).total_seconds() / 60
                
                # Calculate pips (for gold: 1 pip = 0.01 usually)
                symbol_info_obj = mt5.symbol_info(symbol)
                if symbol_info_obj:
                    point = symbol_info_obj.point
                else:
                    point = 0.01  # Default for gold
                
                if pos.type == mt5.POSITION_TYPE_BUY:
                    pips = (exit_price - entry_price) / point / 10  # Divide by 10 for gold
                else:
                    pips = (entry_price - exit_price) / point / 10
                
                logger.info(f"Position {ticket} closed successfully")
                
                return {
                    'success': True,
                    'ticket': ticket,
                    'exit_price': exit_price,
                    'profit': current_profit,
                    'pips': pips,
                    'duration_minutes': duration_minutes,
                    'volume': volume,
                    'retcode': result.retcode,
                    'comment': result.comment
                }
            else:
                logger.error(f"Failed to close position {ticket}: {result.comment}")
                return {
                    'success': False,
                    'reason': result.comment,
                    'retcode': result.retcode
                }
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {'success': False, 'reason': str(e)}
            return False


# Timeframe mapping
TIMEFRAME_MAP = {
    '1M': mt5.TIMEFRAME_M1,
    '5M': mt5.TIMEFRAME_M5,
    '15M': mt5.TIMEFRAME_M15,
    '30M': mt5.TIMEFRAME_M30,
    '1H': mt5.TIMEFRAME_H1,
    '4H': mt5.TIMEFRAME_H4,
    '1D': mt5.TIMEFRAME_D1
}
