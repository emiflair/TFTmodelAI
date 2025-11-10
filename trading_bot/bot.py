"""
Main Trading Bot Orchestrator
Coordinates all components and executes trading loop
"""

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging
import traceback
import json
import MetaTrader5 as mt5
import os
import sys

# Rich terminal UI
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.text import Text

from mt5_connector import MT5Connection
from model_predictor import ModelPredictor
from strategy import TradingStrategy
from risk_manager import RiskManager
from trade_logger import TradeLogger

# Initialize Rich console with Windows-safe encoding
console = Console(force_terminal=True, legacy_windows=False)

# Configure logging (file only, console handled by Rich)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# TEST MODE CONFIGURATION
# ============================================================================
# Set to True to inject fake signals for testing (alternates BUY/SELL)
# Set to False to use real model predictions
TEST_MODE = False

# Timeframe mapping
TIMEFRAME_MAP = {
    '1m': mt5.TIMEFRAME_M1,
    '5m': mt5.TIMEFRAME_M5,
    '15m': mt5.TIMEFRAME_M15,
    '30m': mt5.TIMEFRAME_M30,
    '1h': mt5.TIMEFRAME_H1,
    '4h': mt5.TIMEFRAME_H4,
    '1d': mt5.TIMEFRAME_D1,
}


class TradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(
        self,
        symbol: str = "XAUUSD",
        timeframe: str = "15m",
        model_checkpoint: str = None,
        scaler_path: str = None,
        manifest_path: str = None,
        strategy_config: Dict = None,
        risk_config: Dict = None,
        mt5_config: Dict = None,
        update_interval: int = 60,  # seconds
    ):
        """
        Initialize trading bot
        
        Args:
            symbol: Trading symbol (e.g., "XAUUSD")
            timeframe: Timeframe (e.g., "15m")
            model_checkpoint: Path to model checkpoint (None = auto-find latest)
            scaler_path: Path to scaler file (None = auto-find latest)
            manifest_path: Path to manifest file (None = auto-find latest)
            strategy_config: Strategy configuration dict
            risk_config: Risk management configuration dict
            mt5_config: MT5 connection configuration dict
            update_interval: Seconds between bot updates
        """
        self.symbol = symbol
        self.timeframe_str = timeframe
        # Convert string timeframe to MT5 constant
        self.timeframe = TIMEFRAME_MAP.get(timeframe, mt5.TIMEFRAME_M15)
        self.update_interval = update_interval
        self.running = False
        
        # Stats tracking
        self.iteration = 0
        self.last_prediction = None
        self.last_signal = None
        self.stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_profit': 0.0
        }
        
        # Track trade IDs for proper logging (ticket -> trade_id mapping)
        self.active_trades = {}  # {ticket: {'trade_id': str, 'entry_time': datetime}}
        
        # Clear screen and show startup banner
        os.system('cls' if os.name == 'nt' else 'clear')
        self._show_startup_banner()
        
        # Initialize MT5 connection with progress
        with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"), console=console) as progress:
            task = progress.add_task("[cyan]Connecting to MetaTrader 5...", total=None)
            self.mt5 = MT5Connection(**(mt5_config or {}))
            if not self.mt5.connect():
                console.print("[red]✗ Failed to connect to MT5[/red]")
                raise ConnectionError("Failed to connect to MT5")
            progress.update(task, completed=True)
            console.print("[green]✓ Connected to MT5[/green]")
            
            # Initialize symbol
            task = progress.add_task(f"[cyan]Initializing {symbol}...", total=None)
            if not self.mt5.initialize_symbol(symbol):
                console.print(f"[red]✗ Failed to initialize {symbol}[/red]")
                raise ConnectionError(f"Failed to initialize symbol {symbol}")
            progress.update(task, completed=True)
            console.print(f"[green]✓ Symbol {symbol} ready[/green]")
            
            # Load model
            task = progress.add_task("[cyan]Loading TFT model...", total=None)
            self.predictor = ModelPredictor(
                checkpoint_path=model_checkpoint,
                scaler_path=scaler_path,
                manifest_path=manifest_path
            )
            progress.update(task, completed=True)
            console.print(f"[green]+ Model loaded[/green]")
            
            # Initialize risk manager
            task = progress.add_task("[cyan]Initializing risk manager...", total=None)
            self.risk_manager = RiskManager(**(risk_config or {}))
            progress.update(task, completed=True)
            console.print(f"[green]+ Risk manager ready[/green]")
            
            # Initialize strategy
            task = progress.add_task("[cyan]Initializing strategy...", total=None)
            if strategy_config:
                strategy_config['risk_manager'] = self.risk_manager
            else:
                strategy_config = {'risk_manager': self.risk_manager}
            self.strategy = TradingStrategy(**strategy_config)
            progress.update(task, completed=True)
            console.print(f"[green]+ Strategy initialized[/green]")
            
            # Initialize trade logger
            task = progress.add_task("[cyan]Initializing trade logger...", total=None)
            trade_log_config = (risk_config or {}).get('trade_log_file', 'trading_bot/trade_log.csv')
            enable_logging = (risk_config or {}).get('enable_trade_logging', True)
            self.trade_logger = TradeLogger(trade_log_config) if enable_logging else None
            progress.update(task, completed=True)
            if enable_logging:
                console.print(f"[green]+ Trade logger active: {trade_log_config}[/green]")
            else:
                console.print(f"[yellow]- Trade logging disabled[/yellow]")
        
        console.print("\n[bold green]" + "="*60 + "[/bold green]")
        console.print("[bold green]           TRADING BOT READY[/bold green]")
        console.print("[bold green]" + "="*60 + "[/bold green]\n")
    
    def _show_startup_banner(self):
        """Display startup banner with Windows-safe characters"""
        console.print("\n[bold cyan]" + "="*60 + "[/bold cyan]")
        console.print("[bold cyan]        AI-POWERED TRADING BOT[/bold cyan]")
        console.print("[bold cyan]" + "="*60 + "[/bold cyan]")
        console.print(f"[white]Symbol:[/white] [cyan]{self.symbol}[/cyan]")
        console.print(f"[white]Timeframe:[/white] [cyan]{self.timeframe_str}[/cyan]")
        console.print(f"[white]Update Cycle:[/white] [cyan]{self.update_interval}s[/cyan]")
        console.print(f"[white]Model:[/white] [green]Temporal Fusion Transformer[/green]")
        console.print("[bold cyan]" + "="*60 + "[/bold cyan]")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")
    
    def run(self):
        """Main bot loop"""
        """Main bot loop with beautiful UI"""
        self.running = True
        
        try:
            while self.running:
                self.iteration += 1
                loop_start = time.time()
                
                # Show iteration header
                self._show_iteration_header()
                
                try:
                    # Execute one bot cycle with progress bar
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[bold blue]{task.description}"),
                        BarColumn(bar_width=40),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        console=console
                    ) as progress:
                        
                        # Create tasks
                        main_task = progress.add_task("[cyan]Processing cycle...", total=100)
                        
                        # Execute cycle
                        self._execute_cycle(progress, main_task)
                        
                        progress.update(main_task, completed=100)
                    
                except Exception as e:
                    console.print(f"[red]✗ Error: {str(e)}[/red]")
                    logger.error(f"Error in bot cycle: {e}")
                    logger.error(traceback.format_exc())
                
                # Calculate sleep time and show countdown
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.update_interval - elapsed)
                
                console.print(f"\n[dim]Cycle completed in {elapsed:.2f}s[/dim]")
                
                # Countdown with progress bar
                if sleep_time > 0:
                    with Progress(
                        TextColumn("[bold yellow]Next update in:"),
                        BarColumn(bar_width=40, complete_style="yellow", finished_style="green"),
                        TextColumn("[bold]{task.fields[time_left]:.1f}s"),
                        console=console,
                        transient=True
                    ) as progress:
                        task = progress.add_task("waiting", total=sleep_time, time_left=sleep_time)
                        
                        for i in range(int(sleep_time * 10)):  # Update 10x per second
                            time.sleep(0.1)
                            remaining = sleep_time - (i * 0.1)
                            progress.update(task, completed=i * 0.1, time_left=remaining)
                        
                        progress.update(task, completed=sleep_time, time_left=0)
                
                console.print("\n")
        
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️  Keyboard interrupt received[/yellow]")
        
        finally:
            self.stop()
    
    def _show_iteration_header(self):
        """Display iteration header"""
        console.print("\n[bold cyan]" + "═" * 70 + "[/bold cyan]")
        console.print(f"[bold white]  Iteration #{self.iteration}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/bold white]")
        console.print("[bold cyan]" + "═" * 70 + "[/bold cyan]\n")
    
    def _execute_cycle(self, progress=None, main_task=None):
        """Execute one bot cycle"""
        
        # Helper to update progress
        def update_progress(value, description=None):
            if progress and main_task is not None:
                progress.update(main_task, completed=value, description=description or "Processing...")
        
        # 1. Get account info (10%)
        update_progress(10, "[cyan]Fetching account info...")
        account_info = self.mt5.get_account_info()
        if not account_info:
            console.print("[red]✗ Failed to get account info[/red]")
            return
        
        # 2. Check daily loss limit (20%)
        update_progress(20, "[cyan]Checking risk limits...")
        daily_check = self.risk_manager.check_daily_risk_limit(account_info['equity'])
        if not daily_check['ok']:
            console.print(f"[red]🚨 {daily_check['reason']}[/red]")
            console.print("[red]Closing all positions...[/red]")
            self.risk_manager.close_all_positions_for_protection(self.mt5)
            return
        
        # 3. Get open positions (30%)
        update_progress(30, "[cyan]Fetching open positions...")
        open_positions = self.mt5.get_open_positions(self.symbol)
        
        # 4. Get symbol info (40%)
        update_progress(40, "[cyan]Getting market info...")
        symbol_info = self.mt5.get_symbol_info(self.symbol)
        if symbol_info:
            spread_pips = symbol_info.get('spread', 0) / 10
        else:
            spread_pips = 0
        
        # 5. Get market data (50%) - Fetch extra bars to account for NaN dropping in feature engineering
        # Need 128 final rows after features (model trained with lookback_bars=128), but lose ~50 rows to NaN dropping, so fetch 200
        update_progress(50, "[cyan]Fetching market data (200 bars for features)...")
        market_data = self.mt5.get_latest_bars(
            symbol=self.symbol,
            timeframe=self.timeframe,
            count=200
        )
        
        if market_data is None or len(market_data) < 128:
            console.print(f"[red]✗ Insufficient data: {len(market_data) if market_data is not None else 0} bars (need 128+)[/red]")
            return
        
        current_price = market_data['close'].iloc[-1]
        
        # 6. Get model prediction (70%)
        update_progress(70, "[cyan]Running AI prediction...")
        try:
            prediction = self.predictor.predict(market_data)
            self.last_prediction = prediction
            
            # **TEST MODE**: Override prediction with fake signal for testing
            self._test_mode_active = False  # Reset flag
            if TEST_MODE and self.iteration % 3 == 1:  # Every 3rd iteration
                # Alternate between BUY and SELL signals for testing both directions
                signal_count = (self.iteration // 3)
                is_buy_signal = (signal_count % 2 == 0)  # Even = BUY, Odd = SELL
                
                if is_buy_signal:
                    logger.info("TEST MODE: Injecting fake BUY signal")
                    prediction['move_pct'] = 0.25  # 0.25% move (above 0.10% threshold)
                    prediction['direction'] = 'UP'
                    prediction['confidence'] = 0.75  # High confidence
                    prediction['q50'] = current_price * 1.0025  # Target 0.25% higher
                    prediction['q10'] = current_price * 0.999   # Downside risk
                    prediction['q90'] = current_price * 1.005   # Upside potential
                    console.print("[yellow]⚠️  TEST MODE: Fake BUY signal injected[/yellow]")
                else:
                    logger.info("TEST MODE: Injecting fake SELL signal")
                    prediction['move_pct'] = 0.25  # 0.25% move (above 0.10% threshold)
                    prediction['direction'] = 'DOWN'
                    prediction['confidence'] = 0.75  # High confidence
                    prediction['q50'] = current_price * 0.9975  # Target 0.25% lower
                    prediction['q10'] = current_price * 0.995   # Downside potential
                    prediction['q90'] = current_price * 1.001   # Upside risk
                    console.print("[yellow]⚠️  TEST MODE: Fake SELL signal injected[/yellow]")
                
                self._test_mode_active = True  # Mark this prediction as test mode
            
        except Exception as e:
            console.print(f"[red]✗ Prediction error: {str(e)}[/red]")
            logger.error(f"Prediction error: {e}")
            return
        
        # 7. Generate signal (85%)
        update_progress(85, "[cyan]Generating trading signal...")
        signal = self.strategy.generate_signal(prediction)
        self.last_signal = signal
        
        # 8. Display beautiful dashboard (95%)
        update_progress(95, "[cyan]Updating dashboard...")
        self._display_dashboard(account_info, open_positions, current_price, spread_pips, prediction, signal, daily_check)
        
        # 9. Check for new trade
        if len(open_positions) < self.risk_manager.max_open_positions:
            if signal['signal'] != 'HOLD':
                self._check_new_trade(prediction, account_info, open_positions, spread_pips)
        
        # 10. Manage existing positions
        if open_positions:
            self._manage_open_positions(open_positions, current_price, prediction)
    
    def _display_dashboard(self, account_info, positions, price, spread, prediction, signal, daily_check):
        """Display beautiful trading dashboard"""
        
        # Create main table
        table = Table(show_header=False, border_style="cyan", padding=(0, 1))
        table.add_column("Item", style="bold white", width=20)
        table.add_column("Value", style="cyan")
        
        # Account section
        equity_color = "green" if account_info['equity'] >= account_info['balance'] else "red"
        table.add_row("[bold yellow]ACCOUNT[/bold yellow]", "")
        table.add_row("  Balance", f"[white]${account_info['balance']:,.2f}[/white]")
        table.add_row("  Equity", f"[{equity_color}]${account_info['equity']:,.2f}[/{equity_color}]")
        table.add_row("  P&L", f"[{equity_color}]{account_info['profit']:+,.2f}[/{equity_color}]")
        table.add_row("  Free Margin", f"[white]${account_info['free_margin']:,.2f}[/white]")
        
        # Daily risk section
        daily_loss_pct = daily_check.get('daily_loss_pct', 0)
        daily_color = "red" if daily_loss_pct > 2 else "yellow" if daily_loss_pct > 1 else "green"
        table.add_row("", "")
        table.add_row("[bold yellow]DAILY RISK[/bold yellow]", "")
        table.add_row("  Daily Loss", f"[{daily_color}]${daily_check.get('daily_loss', 0):.2f} ({daily_loss_pct:.2f}%)[/{daily_color}]")
        table.add_row("  Limit", f"[red]${self.risk_manager.max_daily_loss * account_info['balance'] / 100:.2f} (4.0%)[/red]")
        
        # Market section
        table.add_row("", "")
        table.add_row(f"[bold yellow]MARKET ({self.symbol})[/bold yellow]", "")
        table.add_row("  Current Price", f"[white]${price:,.2f}[/white]")
        table.add_row("  Spread", f"[white]{spread:.1f} pips[/white]")
        table.add_row("  Open Positions", f"[cyan]{len(positions)}[/cyan]")
        
        # AI Prediction section
        if prediction and prediction.get('ok', False):
            direction_color = "green" if prediction['direction'] == 'UP' else "red" if prediction['direction'] == 'DOWN' else "yellow"
            conf_color = "green" if prediction['confidence'] >= 0.7 else "yellow" if prediction['confidence'] >= 0.5 else "red"
            
            table.add_row("", "")
            table.add_row("[bold yellow]AI PREDICTION[/bold yellow]", "")
            table.add_row("  Direction", f"[{direction_color}]{prediction['direction']}[/{direction_color}]")
            table.add_row("  Confidence", f"[{conf_color}]{prediction['confidence']:.2%}[/{conf_color}]")
            table.add_row("  Expected Move", f"[white]{prediction['move_pct']:+.2f}%[/white]")
            table.add_row("  Target (q50)", f"[cyan]${prediction['q50']:,.2f}[/cyan]")
            table.add_row("  Range", f"[dim]${prediction['q10']:,.2f} - ${prediction['q90']:,.2f}[/dim]")
        else:
            table.add_row("", "")
            table.add_row("[bold yellow]AI PREDICTION[/bold yellow]", "")
            table.add_row("  Status", f"[red]X Error: {prediction.get('reason', 'Unknown error')}[/red]")
        
        # Trading Signal section
        if signal and signal.get('signal'):
            signal_color = "green" if signal['signal'] == 'BUY' else "red" if signal['signal'] == 'SELL' else "yellow"
            table.add_row("", "")
            table.add_row("[bold yellow]SIGNAL[/bold yellow]", "")
            table.add_row("  Action", f"[{signal_color}]{signal['signal']}[/{signal_color}]")
            table.add_row("  Reason", f"[dim]{signal.get('reason', 'N/A')[:40]}[/dim]")
        
        # Display with simple border
        console.print("\n[bold cyan]" + "="*70 + "[/bold cyan]")
        console.print(f"[bold white]Trading Dashboard - Iteration #{self.iteration}[/bold white]")
        console.print("[bold cyan]" + "="*70 + "[/bold cyan]")
        console.print(table)
    
    def _manage_open_positions(self, positions: list, current_price: float, prediction: Dict):
        """Manage existing open positions"""
        
        for pos in positions:
            ticket = pos['ticket']
            pos_type = pos['type']
            profit = pos['profit']
            
            logger.info(f"Position #{ticket} ({pos_type}): ${profit:.2f} profit")
            
            # Check if should close
            close_decision = self.risk_manager.should_close_position(pos, current_price, prediction)
            
            if close_decision['should_close']:
                logger.info(f"Closing position: {close_decision['reason']}")
                result = self.mt5.close_position(ticket)
                
                if result and result.get('success'):
                    logger.info(f"✓ Position closed successfully")
                    was_winner = profit > 0
                    self.risk_manager.update_daily_pnl(profit, was_winner)
                    
                    # Log trade close to CSV using the original trade_id
                    if self.trade_logger:
                        # Get account info for balance_after
                        account_info = self.mt5.get_account_info()
                        
                        # Lookup original trade_id or create new one if not found
                        if ticket in self.active_trades:
                            trade_id = self.active_trades[ticket]['trade_id']
                            entry_time = self.active_trades[ticket]['entry_time']
                            duration_minutes = (datetime.now() - entry_time).total_seconds() / 60
                            del self.active_trades[ticket]  # Remove from tracking
                        else:
                            # Trade not tracked (maybe opened before bot started)
                            trade_id = f"{self.symbol}_{ticket}_UNKNOWN"
                            duration_minutes = None
                        
                        self.trade_logger.log_trade_close(
                            trade_id=trade_id,
                            ticket=ticket,
                            exit_price=result.get('exit_price', 0),
                            pips=result.get('pips', 0),
                            profit_loss=result.get('profit', 0),
                            balance_after=account_info.get('balance', 0),
                            close_reason=close_decision.get('reason', 'UNKNOWN'),
                            duration_minutes=duration_minutes,
                            slippage=None,
                            notes=""
                        )
                else:
                    logger.error(f"✗ Failed to close position")
                
                continue
            
            # Check for position management actions
            mgmt_action = self.strategy.manage_position(pos, current_price, prediction)
            
            if mgmt_action['action'] == 'CLOSE_FULL':
                logger.info(f"Strategy says close: {mgmt_action['reason']}")
                result = self.mt5.close_position(ticket)
                
                if result and result.get('success'):
                    logger.info(f"✓ Position closed")
                    was_winner = profit > 0
                    self.risk_manager.update_daily_pnl(profit, was_winner)
                    
                    # Log trade close to CSV using the original trade_id
                    if self.trade_logger:
                        account_info = self.mt5.get_account_info()
                        
                        # Lookup original trade_id
                        if ticket in self.active_trades:
                            trade_id = self.active_trades[ticket]['trade_id']
                            entry_time = self.active_trades[ticket]['entry_time']
                            duration_minutes = (datetime.now() - entry_time).total_seconds() / 60
                            del self.active_trades[ticket]
                        else:
                            trade_id = f"{self.symbol}_{ticket}_UNKNOWN"
                            duration_minutes = None
                        
                        self.trade_logger.log_trade_close(
                            trade_id=trade_id,
                            ticket=ticket,
                            exit_price=result.get('exit_price', 0),
                            pips=result.get('pips', 0),
                            profit_loss=result.get('profit', 0),
                            balance_after=account_info.get('balance', 0),
                            close_reason=mgmt_action.get('reason', 'STRATEGY'),
                            duration_minutes=duration_minutes,
                            slippage=None,
                            notes=""
                        )
            
            elif mgmt_action['action'] == 'CLOSE_PARTIAL':
                logger.info(f"Partial close: {mgmt_action['reason']}")
                # TODO: Implement partial close in MT5 connector
                # For now, just log
                logger.info(f"Partial close not yet implemented")
            
            elif mgmt_action['action'] == 'TRAIL_STOP':
                logger.info(f"Trailing stop: {mgmt_action['reason']}")
                # TODO: Implement modify position in MT5 connector
                # For now, just log
                logger.info(f"Modify SL to {mgmt_action['new_stop_loss']:.5f}")
            
            else:
                logger.info(f"Hold position: {mgmt_action['reason']}")
    
    def _check_new_trade(self, prediction: Dict, account_info: Dict, open_positions: list, spread_pips: float = 0):
        """Check if should open new trade"""
        
        # Signal already generated in _execute_cycle
        signal = self.last_signal
        
        if signal['signal'] == 'HOLD':
            return
        
        # Check if we need to close opposite direction positions first
        close_check = self.risk_manager.should_close_opposite_positions(open_positions, signal['signal'])
        if close_check['should_close']:
            console.print(f"[yellow]⚠  {close_check['reason']}[/yellow]")
            console.print(f"[yellow]   Closing {len(close_check['positions_to_close'])} opposite position(s)...[/yellow]")
            
            # Close all opposite positions
            for ticket in close_check['positions_to_close']:
                logger.info(f"Closing opposite position {ticket} before opening {signal['signal']} trade")
                result = self.mt5.close_position(ticket)
                
                if result and result.get('success'):
                    console.print(f"[green]✓ Closed position {ticket}[/green]")
                    
                    # Update daily P&L
                    profit = result.get('profit', 0)
                    was_winner = profit > 0
                    self.risk_manager.update_daily_pnl(profit, was_winner)
                    
                    # Log trade close using original trade_id
                    if self.trade_logger:
                        account_info_updated = self.mt5.get_account_info()
                        
                        # Lookup original trade_id
                        if ticket in self.active_trades:
                            trade_id = self.active_trades[ticket]['trade_id']
                            entry_time = self.active_trades[ticket]['entry_time']
                            duration_minutes = (datetime.now() - entry_time).total_seconds() / 60
                            del self.active_trades[ticket]
                        else:
                            trade_id = f"{self.symbol}_{ticket}_UNKNOWN"
                            duration_minutes = None
                        
                        self.trade_logger.log_trade_close(
                            trade_id=trade_id,
                            ticket=ticket,
                            exit_price=result.get('exit_price', 0),
                            pips=result.get('pips', 0),
                            profit_loss=result.get('profit', 0),
                            balance_after=account_info_updated.get('balance', 0),
                            close_reason="DIRECTION_CHANGE",
                            duration_minutes=duration_minutes,
                            slippage=None,
                            notes="Closed for opposite direction trade"
                        )
                else:
                    console.print(f"[red]✗ Failed to close position {ticket}[/red]")
                    logger.error(f"Failed to close opposite position {ticket}")
                    return  # Don't open new trade if we can't close opposite
            
            # Refresh account info and positions after closing
            account_info = self.mt5.get_account_info()
            open_positions = self.mt5.get_open_positions(self.symbol)
        
        # Validate with risk manager
        validation = self.risk_manager.validate_trade(signal, account_info, open_positions, spread_pips)
        
        if not validation['approved']:
            console.print(f"[yellow]⚠  Trade rejected: {validation['reason']}[/yellow]")
            return
        
        # Get symbol info
        symbol_info = self.mt5.get_symbol_info(self.symbol)
        if not symbol_info:
            console.print("[red]✗ Failed to get symbol info[/red]")
            return
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            account_balance=account_info['balance'],
            entry_price=signal['entry_price'],
            stop_loss=signal['stop_loss'],
            confidence=signal['confidence'],
            symbol_info=symbol_info
        )
        
        if not position_size['ok']:
            console.print(f"[red]✗ Position sizing failed: {position_size.get('reason', 'Unknown')}[/red]")
            return
        
        volume = position_size['volume']
        
        # Show trade plan
        console.print("\n[bold green]" + "="*60 + "[/bold green]")
        console.print("[bold white]EXECUTING TRADE[/bold white]")
        console.print("[bold green]" + "="*60 + "[/bold green]")
        
        trade_table = Table(show_header=False, border_style="green")
        trade_table.add_column("Item", style="bold white")
        trade_table.add_column("Value", style="cyan")
        
        signal_color = "green" if signal['signal'] == 'BUY' else "red"
        trade_table.add_row("Action", f"[{signal_color}]{signal['signal']}[/{signal_color}]")
        trade_table.add_row("Entry Price", f"${signal['entry_price']:,.2f}")
        trade_table.add_row("Stop Loss", f"${signal['stop_loss']:,.2f}")
        trade_table.add_row("Take Profit", f"${signal['take_profit']:,.2f}")
        trade_table.add_row("Volume", f"{volume:.2f} lots")
        trade_table.add_row("Risk Amount", f"${position_size['risk_amount']:.2f}")
        trade_table.add_row("Risk %", f"{position_size['risk_percent']:.2f}%")
        trade_table.add_row("R:R Ratio", f"1:{signal['reward_risk_ratio']:.2f}")
        trade_table.add_row("Confidence", f"{signal['confidence']:.2%}")
        
        console.print(trade_table)
        
        # Execute trade
        order_result = self.mt5.send_order(
            symbol=self.symbol,
            order_type=signal['signal'],
            volume=volume,
            price=signal['entry_price'],
            sl=signal['stop_loss'],
            tp=signal['take_profit'],
            comment=f"TFT_{signal['confidence']:.2f}"
        )
        
        if order_result is not None and order_result.get('retcode', 0) == 10009:
            ticket = order_result.get('order', 0)
            console.print(f"[bold green]✅ Trade executed successfully! Ticket: {ticket}[/bold green]\n")
            self.stats['total_trades'] += 1
            
            # Log trade to CSV and track the trade_id
            if self.trade_logger:
                trade_id = f"{self.symbol}_{ticket}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Store trade_id mapping for later lookup when closing
                self.active_trades[ticket] = {
                    'trade_id': trade_id,
                    'entry_time': datetime.now()
                }
                
                self.trade_logger.log_trade_open(
                    trade_id=trade_id,
                    ticket=ticket,
                    symbol=self.symbol,
                    action=signal['signal'],
                    volume=volume,
                    entry_price=signal['entry_price'],
                    stop_loss=signal['stop_loss'],
                    take_profit=signal['take_profit'],
                    ai_prediction={
                        'direction': prediction.get('direction', ''),
                        'confidence': signal['confidence'],
                        'move_pct': prediction.get('move_pct', 0),
                        'q10': prediction.get('q10', 0),
                        'q50': prediction.get('q50', 0),
                        'q90': prediction.get('q90', 0)
                    },
                    risk_details={
                        'risk_amount': position_size['risk_amount'],
                        'risk_percent': position_size['risk_percent'],
                        'rr_ratio': signal['reward_risk_ratio']
                    },
                    balance=account_info['balance'],
                    equity=account_info['equity'],
                    spread=spread_pips,
                    notes="TEST_MODE" if hasattr(self, '_test_mode_active') and self._test_mode_active else ""
                )
        else:
            error_msg = order_result.get('comment', 'Unknown error') if order_result else 'No response from MT5'
            console.print(f"[bold red]❌ Trade failed: {error_msg}[/bold red]\n")
            return
        
        volume = position_size['volume']
        logger.info(
            f"Position size: {volume:.2f} lots | "
            f"Risk: ${position_size['risk_amount']:.2f} ({position_size['risk_percent']:.2f}%) | "
            f"Total cost: ${position_size['total_cost']:.2f} | "
            f"Confidence: {position_size['confidence']:.2f}"
        )
        
        # Execute trade
        logger.info(f"\n🎯 Executing {signal['signal']} trade:")
        logger.info(f"   Entry: {signal['entry_price']:.2f}")
        logger.info(f"   SL: {signal['stop_loss']:.2f}")
        logger.info(f"   TP: {signal['take_profit']:.2f}")
        logger.info(f"   Volume: {volume} lots")
        logger.info(f"   R:R: {signal['reward_risk_ratio']:.2f}")
        
        order_result = self.mt5.send_order(
            symbol=self.symbol,
            order_type=signal['signal'],
            volume=volume,
            price=signal['entry_price'],
            sl=signal['stop_loss'],
            tp=signal['take_profit'],
            comment=f"TFT_{signal['confidence']:.2f}"
        )
        
        if order_result and order_result['retcode'] == 10009:  # TRADE_RETCODE_DONE
            logger.info(f"✅ Trade executed successfully! Ticket: {order_result.get('order', 'N/A')}")
        else:
            logger.error(f"❌ Trade failed: {order_result.get('comment', 'Unknown error')}")
    
    def stop(self):
        """Stop the trading bot"""
        console.print("\n")
        console.print("[bold yellow]" + "═" * 70 + "[/bold yellow]")
        console.print("[bold yellow]           Stopping Trading Bot[/bold yellow]")
        console.print("[bold yellow]" + "=" * 70 + "[/bold yellow]")
        
        self.running = False
        
        # Show summary
        console.print("\n[bold yellow]SESSION SUMMARY[/bold yellow]")
        summary_table = Table(show_header=False, border_style="yellow")
        summary_table.add_column("Metric", style="bold white")
        summary_table.add_column("Value", style="cyan")
        
        summary_table.add_row("Total Iterations", str(self.iteration))
        summary_table.add_row("Total Trades", str(self.stats['total_trades']))
        summary_table.add_row("Wins", f"[green]{self.stats['wins']}[/green]")
        summary_table.add_row("Losses", f"[red]{self.stats['losses']}[/red]")
        
        win_rate = (self.stats['wins'] / self.stats['total_trades'] * 100) if self.stats['total_trades'] > 0 else 0
        summary_table.add_row("Win Rate", f"{win_rate:.1f}%")
        
        console.print(summary_table)
        
        # Show trade log statistics if available
        if self.trade_logger:
            console.print("\n[bold cyan]TRADE LOG STATISTICS[/bold cyan]")
            log_stats = self.trade_logger.get_statistics()
            
            if log_stats.get('total_trades', 0) > 0:
                log_table = Table(show_header=False, border_style="cyan")
                log_table.add_column("Metric", style="bold white")
                log_table.add_column("Value", style="cyan")
                
                log_table.add_row("Total Logged", str(log_stats['total_trades']))
                log_table.add_row("Wins", f"[green]{log_stats['wins']}[/green]")
                log_table.add_row("Losses", f"[red]{log_stats['losses']}[/red]")
                log_table.add_row("Win Rate", f"{log_stats['win_rate']:.1f}%")
                log_table.add_row("Profit Factor", f"{log_stats['profit_factor']:.2f}")
                log_table.add_row("Avg Win", f"[green]${log_stats['avg_win']:.2f}[/green]")
                log_table.add_row("Avg Loss", f"[red]${log_stats['avg_loss']:.2f}[/red]")
                log_table.add_row("Total Profit", f"${log_stats['total_profit']:.2f}")
                
                console.print(log_table)
            else:
                console.print("[yellow]No completed trades in log yet[/yellow]")
        
        # Disconnect from MT5
        self.mt5.disconnect()
        console.print("\n[green]+ Disconnected from MT5[/green]")
        console.print("[bold green]Trading bot stopped successfully[/bold green]\n")
    
    def get_status(self) -> Dict:
        """Get bot status summary"""
        account_info = self.mt5.get_account_info()
        open_positions = self.mt5.get_open_positions(self.symbol)
        risk_report = self.risk_manager.get_risk_report(account_info, open_positions)
        
        return {
            'running': self.running,
            'iteration': self.iteration,
            'last_update': self.last_update_time,
            'account': account_info,
            'open_positions': len(open_positions),
            'last_prediction': self.last_prediction,
            'last_signal': self.last_signal,
            'risk_report': risk_report
        }


def main():
    """Main entry point"""
    
    # Configuration
    config = {
        'symbol': 'XAUUSD',
        'timeframe': '15m',
        'update_interval': 30,  # ⏱️ Update every 30 seconds
        
        # Model paths (Colab-trained checkpoint)
        'model_checkpoint': 'artifacts/checkpoints/tft_XAUUSD_15m_3B_latest.ckpt',
        'scaler_path': None,    # Auto-detect
        'manifest_path': None,  # Auto-detect
        
        # Strategy configuration
        'strategy_config': {
            'min_confidence': 0.30,        # Lowered - dynamic risk adjusts based on confidence
            'min_move_pct': 0.10,          # Lowered - more opportunities
            'enable_trailing_stop': True,
            'trailing_distance_pips': 50,
            'partial_close_pct': 0.5,
            'use_time_filter': False,      # DISABLED for testing - allow 24/7
            'allowed_hours': list(range(0, 24)),  # 24/7 for testing
        },
        
        # Risk management configuration (NEW DYNAMIC SYSTEM)
        'risk_config': {
            'account_balance': 10000.0,
            'max_daily_loss': 4.0,         # 4% = $400 max daily loss
            'risk_low_confidence': 0.5,    # 0.5% risk for confidence 0.3-0.49
            'risk_medium_confidence': 1.0, # 1% risk for confidence 0.5-0.69
            'risk_high_confidence': 2.0,   # 2% risk for confidence 0.85-1.0
            'max_lot_size': 2.0,           # Cap at 2.0 lots
            'max_open_positions': 3,
            'max_leverage': 10.0,
            'rr_defensive': 0.5,           # R:R 1:0.5 for low confidence
            'rr_normal': 1.0,              # R:R 1:1 for medium confidence
            'rr_strong': 2.0,              # R:R 1:2 for strong confidence
            'rr_high': 3.0,                # R:R 1:3 for high confidence
            'pip_value': 1.0,              # $1 per pip per 0.01 lot
            'spread_cost_per_microlot': 0.5,
            'commission_per_lot': 6.0,
            'max_spread_pips': 25,         # Skip trade if spread > 25 pips
            'slippage_tolerance_pips': 10,
            'consecutive_loss_reduction': 0.5,  # Cut risk by 50% after 2 losses
            'max_consecutive_losses': 2,
        },
        
        # MT5 configuration (optional - will use defaults)
        'mt5_config': {
            # Add your MT5 credentials here if needed
            'login': 52587216,
            'password': 'Y7p$bW1iJKmtSq',
            'server': 'ICMarketsSC-Demo',
        }
    }
    
    # Create and run bot
    try:
        bot = TradingBot(**config)
        bot.run()
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
