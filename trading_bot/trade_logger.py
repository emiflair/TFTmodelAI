"""
Trade Logger - CSV logging for all trades
Tracks every trade detail for model improvement and analysis
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TradeLogger:
    """Logs all trades to CSV for analysis and model improvement"""
    
    def __init__(self, log_file: str = "trade_log.csv"):
        """
        Initialize trade logger
        
        Args:
            log_file: Path to CSV log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # CSV columns
        self.columns = [
            # Trade identification
            'timestamp',
            'trade_id',
            'ticket',
            
            # Trade details
            'symbol',
            'action',  # BUY or SELL
            'volume',
            'entry_price',
            'exit_price',
            'stop_loss',
            'take_profit',
            
            # AI prediction details
            'ai_direction',
            'ai_confidence',
            'predicted_move_pct',
            'predicted_q10',
            'predicted_q50',
            'predicted_q90',
            
            # Risk management
            'risk_amount',
            'risk_pct',
            'rr_ratio',
            
            # Trade outcome
            'status',  # OPEN, CLOSED_WIN, CLOSED_LOSS, CLOSED_BREAKEVEN
            'pips',
            'profit_loss',
            'profit_loss_pct',
            'duration_minutes',
            
            # Market conditions
            'spread',
            'slippage',
            
            # Account state
            'balance_before',
            'balance_after',
            'equity',
            
            # Additional metadata
            'close_reason',  # TP, SL, TRAILING_STOP, MANUAL, PROFIT_TARGET, LOSS_LIMIT
            'notes'
        ]
        
        # Initialize CSV if it doesn't exist
        if not self.log_file.exists():
            self._initialize_csv()
            logger.info(f"Created new trade log: {self.log_file}")
        else:
            logger.info(f"Using existing trade log: {self.log_file}")
    
    def _initialize_csv(self):
        """Create new CSV with headers"""
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
    
    def log_trade_open(
        self,
        trade_id: str,
        ticket: int,
        symbol: str,
        action: str,
        volume: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        ai_prediction: Dict,
        risk_details: Dict,
        balance: float,
        equity: float,
        spread: float,
        notes: str = ""
    ):
        """
        Log a newly opened trade
        
        Args:
            trade_id: Unique trade identifier
            ticket: MT5 ticket number
            symbol: Trading symbol
            action: BUY or SELL
            volume: Lot size
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            ai_prediction: Dict with AI prediction details
            risk_details: Dict with risk management details
            balance: Account balance
            equity: Account equity
            spread: Spread in pips
            notes: Additional notes
        """
        row = {
            # Trade identification
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'trade_id': trade_id,
            'ticket': ticket,
            
            # Trade details
            'symbol': symbol,
            'action': action,
            'volume': volume,
            'entry_price': entry_price,
            'exit_price': '',
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            
            # AI prediction details
            'ai_direction': ai_prediction.get('direction', ''),
            'ai_confidence': ai_prediction.get('confidence', 0),
            'predicted_move_pct': ai_prediction.get('move_pct', 0),
            'predicted_q10': ai_prediction.get('q10', 0),
            'predicted_q50': ai_prediction.get('q50', 0),
            'predicted_q90': ai_prediction.get('q90', 0),
            
            # Risk management
            'risk_amount': risk_details.get('risk_amount', 0),
            'risk_pct': risk_details.get('risk_percent', 0),
            'rr_ratio': risk_details.get('rr_ratio', 0),
            
            # Trade outcome (empty for open trades)
            'status': 'OPEN',
            'pips': '',
            'profit_loss': '',
            'profit_loss_pct': '',
            'duration_minutes': '',
            
            # Market conditions
            'spread': spread,
            'slippage': '',
            
            # Account state
            'balance_before': balance,
            'balance_after': '',
            'equity': equity,
            
            # Additional metadata
            'close_reason': '',
            'notes': notes
        }
        
        self._write_row(row)
        logger.info(f"Logged trade open: {trade_id} | {action} {volume} {symbol} @ {entry_price}")
    
    def log_trade_close(
        self,
        trade_id: str,
        ticket: int,
        exit_price: float,
        pips: float,
        profit_loss: float,
        balance_after: float,
        close_reason: str,
        duration_minutes: Optional[float] = None,
        slippage: Optional[float] = None,
        notes: str = ""
    ):
        """
        Log a closed trade (update the existing row or create new entry)
        
        Args:
            trade_id: Trade identifier
            ticket: MT5 ticket number
            exit_price: Exit price
            pips: Pips gained/lost
            profit_loss: Profit/loss in dollars
            balance_after: Account balance after close
            close_reason: Reason for close (TP, SL, TRAILING_STOP, etc.)
            duration_minutes: Trade duration in minutes
            slippage: Slippage in pips
            notes: Additional notes
        """
        # For simplicity, we'll append a close record
        # In production, you might want to update the existing row
        
        # Determine status
        if profit_loss > 1:
            status = 'CLOSED_WIN'
        elif profit_loss < -1:
            status = 'CLOSED_LOSS'
        else:
            status = 'CLOSED_BREAKEVEN'
        
        # Read existing file to find the balance_before
        balance_before = None
        try:
            with open(self.log_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['trade_id'] == trade_id and row['status'] == 'OPEN':
                        balance_before = float(row['balance_before'])
                        break
        except Exception as e:
            logger.error(f"Error reading trade log: {e}")
        
        profit_loss_pct = (profit_loss / balance_before * 100) if balance_before else 0
        
        row = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'trade_id': trade_id,
            'ticket': ticket,
            'symbol': '',
            'action': '',
            'volume': '',
            'entry_price': '',
            'exit_price': exit_price,
            'stop_loss': '',
            'take_profit': '',
            'ai_direction': '',
            'ai_confidence': '',
            'predicted_move_pct': '',
            'predicted_q10': '',
            'predicted_q50': '',
            'predicted_q90': '',
            'risk_amount': '',
            'risk_pct': '',
            'rr_ratio': '',
            'status': status,
            'pips': pips,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'duration_minutes': duration_minutes or '',
            'spread': '',
            'slippage': slippage or '',
            'balance_before': balance_before or '',
            'balance_after': balance_after,
            'equity': '',
            'close_reason': close_reason,
            'notes': notes
        }
        
        self._write_row(row)
        logger.info(f"Logged trade close: {trade_id} | {status} | P/L: ${profit_loss:.2f}")
    
    def _write_row(self, row: Dict):
        """Write a single row to CSV"""
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                writer.writerow(row)
        except Exception as e:
            logger.error(f"Error writing to trade log: {e}")
    
    def get_statistics(self) -> Dict:
        """
        Get trading statistics from the log
        
        Returns:
            Dict with win rate, profit factor, avg win/loss, etc.
        """
        try:
            with open(self.log_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                wins = []
                losses = []
                
                for row in reader:
                    if row['status'] == 'CLOSED_WIN':
                        wins.append(float(row['profit_loss']))
                    elif row['status'] == 'CLOSED_LOSS':
                        losses.append(float(row['profit_loss']))
                
                total_trades = len(wins) + len(losses)
                if total_trades == 0:
                    return {
                        'total_trades': 0,
                        'wins': 0,
                        'losses': 0,
                        'win_rate': 0,
                        'profit_factor': 0,
                        'avg_win': 0,
                        'avg_loss': 0,
                        'total_profit': 0
                    }
                
                win_rate = len(wins) / total_trades * 100
                total_wins = sum(wins)
                total_losses = abs(sum(losses))
                profit_factor = total_wins / total_losses if total_losses > 0 else 0
                
                return {
                    'total_trades': total_trades,
                    'wins': len(wins),
                    'losses': len(losses),
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'avg_win': sum(wins) / len(wins) if wins else 0,
                    'avg_loss': sum(losses) / len(losses) if losses else 0,
                    'total_profit': sum(wins) + sum(losses)
                }
        
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {}
