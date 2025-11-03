"""
Trade Log Viewer
Simple script to view and analyze trade log CSV
"""

import pandas as pd
from pathlib import Path
import sys

def view_trade_log(log_file='trading_bot/trade_log.csv'):
    """Display trade log statistics and recent trades"""
    
    log_path = Path(log_file)
    
    if not log_path.exists():
        print(f"❌ Trade log not found: {log_file}")
        return
    
    # Read CSV
    df = pd.read_csv(log_file)
    
    print("\n" + "="*80)
    print("                          TRADE LOG ANALYSIS")
    print("="*80)
    
    # Overall statistics
    open_trades = df[df['status'] == 'OPEN']
    closed_trades = df[df['status'].str.contains('CLOSED', na=False)]
    wins = df[df['status'] == 'CLOSED_WIN']
    losses = df[df['status'] == 'CLOSED_LOSS']
    
    print(f"\n📊 OVERALL STATISTICS")
    print(f"{'─'*80}")
    print(f"  Total Open Trades:     {len(open_trades)}")
    print(f"  Total Closed Trades:   {len(closed_trades)}")
    print(f"  Wins:                  {len(wins)} ({len(wins)/len(closed_trades)*100 if len(closed_trades) > 0 else 0:.1f}%)")
    print(f"  Losses:                {len(losses)} ({len(losses)/len(closed_trades)*100 if len(closed_trades) > 0 else 0:.1f}%)")
    
    if len(closed_trades) > 0:
        total_profit = closed_trades['profit_loss'].sum()
        avg_win = wins['profit_loss'].mean() if len(wins) > 0 else 0
        avg_loss = losses['profit_loss'].mean() if len(losses) > 0 else 0
        profit_factor = abs(wins['profit_loss'].sum() / losses['profit_loss'].sum()) if len(losses) > 0 and losses['profit_loss'].sum() != 0 else 0
        
        print(f"\n💰 PROFIT ANALYSIS")
        print(f"{'─'*80}")
        print(f"  Total Profit/Loss:     ${total_profit:,.2f}")
        print(f"  Average Win:           ${avg_win:,.2f}")
        print(f"  Average Loss:          ${avg_loss:,.2f}")
        print(f"  Profit Factor:         {profit_factor:.2f}")
        print(f"  Largest Win:           ${wins['profit_loss'].max() if len(wins) > 0 else 0:,.2f}")
        print(f"  Largest Loss:          ${losses['profit_loss'].min() if len(losses) > 0 else 0:,.2f}")
    
    # AI Prediction Analysis
    if len(open_trades) > 0:
        print(f"\n🤖 AI PREDICTION STATS (Open Trades)")
        print(f"{'─'*80}")
        print(f"  Average Confidence:    {open_trades['ai_confidence'].mean():.2%}")
        print(f"  Avg Predicted Move:    {open_trades['predicted_move_pct'].mean():.3f}%")
    
    if len(closed_trades) > 0:
        print(f"\n🤖 AI PREDICTION STATS (Closed Trades)")
        print(f"{'─'*80}")
        print(f"  Average Confidence:    {closed_trades['ai_confidence'].mean():.2%}")
        
        # Win rate by confidence levels
        high_conf = closed_trades[closed_trades['ai_confidence'] >= 0.7]
        med_conf = closed_trades[(closed_trades['ai_confidence'] >= 0.5) & (closed_trades['ai_confidence'] < 0.7)]
        low_conf = closed_trades[closed_trades['ai_confidence'] < 0.5]
        
        print(f"\n  Win Rate by Confidence:")
        if len(high_conf) > 0:
            high_wr = len(high_conf[high_conf['status'] == 'CLOSED_WIN']) / len(high_conf) * 100
            print(f"    High (≥70%):         {high_wr:.1f}% ({len(high_conf)} trades)")
        if len(med_conf) > 0:
            med_wr = len(med_conf[med_conf['status'] == 'CLOSED_WIN']) / len(med_conf) * 100
            print(f"    Medium (50-70%):     {med_wr:.1f}% ({len(med_conf)} trades)")
        if len(low_conf) > 0:
            low_wr = len(low_conf[low_conf['status'] == 'CLOSED_WIN']) / len(low_conf) * 100
            print(f"    Low (<50%):          {low_wr:.1f}% ({len(low_conf)} trades)")
    
    # Recent trades
    print(f"\n📜 RECENT TRADES (Last 10)")
    print(f"{'─'*80}")
    
    recent = df.tail(10)[['timestamp', 'ticket', 'action', 'status', 'ai_confidence', 'profit_loss', 'notes']]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    print(recent.to_string(index=False))
    
    # Export summary
    print(f"\n{'─'*80}")
    print(f"📁 Trade log location: {log_path.absolute()}")
    print(f"📊 Total entries:      {len(df)}")
    print("="*80 + "\n")

if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'trading_bot/trade_log.csv'
    view_trade_log(log_file)
