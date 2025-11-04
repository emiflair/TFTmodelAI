"""
Import All MT5 Trades to Log
Imports all closed trades from MT5 history into the trade log
"""

import MetaTrader5 as mt5
import csv
from datetime import datetime, timedelta
from pathlib import Path

def import_all_trades(days_back=7):
    """Import all trades from MT5 history"""
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        return
    
    log_file = Path("trading_bot/trade_log.csv")
    
    # Get all tickets already in log
    logged_tickets = set()
    if log_file.exists():
        with open(log_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticket = row.get('ticket', '')
                if ticket and ticket != '':
                    logged_tickets.add(str(ticket))
    
    print(f"📊 Currently logged tickets: {len(logged_tickets)}")
    
    # Get MT5 trade history
    history = mt5.history_deals_get(
        datetime.now() - timedelta(days=days_back),
        datetime.now()
    )
    
    if not history:
        print("⚠️  No trade history found in MT5")
        mt5.shutdown()
        return
    
    print(f"📊 Found {len(history)} deals in MT5 (last {days_back} days)")
    
    # Group deals by position_id (this links entry and exit together)
    positions = {}
    for deal in history:
        pos_id = deal.position_id
        if pos_id not in positions:
            positions[pos_id] = []
        positions[pos_id].append(deal)
    
    print(f"📊 Grouped into {len(positions)} unique positions")
    
    # Get account balance
    account_info = mt5.account_info()
    balance = account_info.balance if account_info else 0
    
    # Process each closed position (must have both entry and exit)
    new_trades = []
    
    for pos_id, deals in positions.items():
        pos_id_str = str(pos_id)
        
        # Skip if already logged (check by position ID)
        if pos_id_str in logged_tickets:
            continue
        
        # Must have at least 2 deals (entry + exit)
        if len(deals) < 2:
            continue
        
        # Sort by time
        deals = sorted(deals, key=lambda d: d.time)
        
        # First deal should be entry (Entry=IN), last should be exit (Entry=OUT)
        entry_deals = [d for d in deals if d.entry == 0]  # 0 = IN
        exit_deals = [d for d in deals if d.entry == 1]   # 1 = OUT
        
        if not entry_deals or not exit_deals:
            continue
        
        entry = entry_deals[0]  # First entry
        exit_deal = exit_deals[-1]  # Last exit
        
        # Calculate totals
        total_profit = sum([d.profit for d in deals])
        total_commission = sum([d.commission for d in deals])
        
        duration = (exit_deal.time - entry.time) / 60  # timestamps in seconds, convert to minutes
        
        # Calculate pips
        if entry.type == 0:  # BUY
            pips = (exit_deal.price - entry.price) / 0.01
        else:  # SELL
            pips = (entry.price - exit_deal.price) / 0.01
        
        # Determine status
        if total_profit > 1:
            status = 'CLOSED_WIN'
        elif total_profit < -1:
            status = 'CLOSED_LOSS'
        else:
            status = 'CLOSED_BREAKEVEN'
        
        action = "BUY" if entry.type == 0 else "SELL"
        
        new_trades.append({
            'timestamp_open': datetime.fromtimestamp(entry.time),
            'timestamp_close': datetime.fromtimestamp(exit_deal.time),
            'position_id': pos_id,
            'symbol': entry.symbol,
            'action': action,
            'volume': entry.volume,
            'entry_price': entry.price,
            'exit_price': exit_deal.price,
            'pips': pips,
            'profit_loss': total_profit,
            'duration_minutes': duration,
            'status': status,
        })
    
    print(f"\n🔄 Found {len(new_trades)} new trades to import")
    
    if new_trades:
        # Append to CSV
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            fieldnames = [
                'timestamp', 'trade_id', 'ticket', 'symbol', 'action', 'volume',
                'entry_price', 'exit_price', 'stop_loss', 'take_profit',
                'ai_direction', 'ai_confidence', 'predicted_move_pct',
                'predicted_q10', 'predicted_q50', 'predicted_q90',
                'risk_amount', 'risk_pct', 'rr_ratio', 'status',
                'pips', 'profit_loss', 'profit_loss_pct', 'duration_minutes',
                'spread', 'slippage', 'balance_before', 'balance_after',
                'equity', 'close_reason', 'notes'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Write each trade as both OPEN and CLOSED entries
            for trade in new_trades:
                # OPEN entry
                row_open = {col: '' for col in fieldnames}
                row_open['timestamp'] = trade['timestamp_open'].strftime('%Y-%m-%d %H:%M:%S')
                row_open['trade_id'] = f"{trade['symbol']}_{trade['position_id']}_IMPORTED"
                row_open['ticket'] = trade['position_id']
                row_open['symbol'] = trade['symbol']
                row_open['action'] = trade['action']
                row_open['volume'] = trade['volume']
                row_open['entry_price'] = trade['entry_price']
                row_open['status'] = 'OPEN'
                row_open['notes'] = 'Imported from MT5 history'
                writer.writerow(row_open)
                
                # CLOSED entry
                row_close = {col: '' for col in fieldnames}
                row_close['timestamp'] = trade['timestamp_close'].strftime('%Y-%m-%d %H:%M:%S')
                row_close['trade_id'] = f"{trade['symbol']}_{trade['position_id']}_IMPORTED"
                row_close['ticket'] = trade['position_id']
                row_close['exit_price'] = trade['exit_price']
                row_close['status'] = trade['status']
                row_close['pips'] = round(trade['pips'], 1)
                row_close['profit_loss'] = round(trade['profit_loss'], 2)
                row_close['duration_minutes'] = round(trade['duration_minutes'], 1)
                row_close['balance_after'] = balance
                row_close['close_reason'] = 'IMPORTED'
                row_close['notes'] = 'Imported from MT5 history'
                writer.writerow(row_close)
                
                status_color = "WIN" if trade['status'] == 'CLOSED_WIN' else "LOSS" if trade['status'] == 'CLOSED_LOSS' else "BE"
                print(f"  ✅ {trade['symbol']} {trade['action']} | Position:{trade['position_id']} | "
                      f"{status_color} ${trade['profit_loss']:+.2f} | {trade['pips']:+.1f} pips | "
                      f"{trade['duration_minutes']:.0f}min")
        
        print(f"\n✅ Successfully imported {len(new_trades)} trades!")
        print(f"   Total profit from imports: ${sum([t['profit_loss'] for t in new_trades]):+.2f}")
    else:
        print("\n✅ No new trades to import!")
    
    mt5.shutdown()

if __name__ == '__main__':
    print("=" * 70)
    print("IMPORT MT5 TRADES TO LOG")
    print("=" * 70)
    print()
    
    import_all_trades(days_back=1)  # Last 24 hours
    
    print("\n" + "=" * 70)
    print("Import complete!")
    print("=" * 70)
