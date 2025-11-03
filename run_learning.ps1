# ====================================================================
# Model Learning from Trade Log (PowerShell)
# Trains improved confidence model using real trading results
# ====================================================================

Write-Host ""
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "MODEL LEARNING FROM TRADE LOG" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if trade log exists
$tradeLog = "trading_bot\trading_bot\trade_log.csv"

if (-not (Test-Path $tradeLog)) {
    Write-Host "ERROR: Trade log not found!" -ForegroundColor Red
    Write-Host "Expected location: $tradeLog" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "The trade log is created automatically when the bot trades." -ForegroundColor Yellow
    Write-Host "Please run the trading bot first to accumulate trade data." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Count trades in log
$lineCount = (Get-Content $tradeLog | Measure-Object -Line).Lines
$tradeCount = $lineCount - 1  # Subtract header row

Write-Host "Found trade log: $tradeLog" -ForegroundColor Green
Write-Host "Total entries: $tradeCount"
Write-Host ""

# Check if enough trades
if ($tradeCount -lt 5) {
    Write-Host "WARNING: Only $tradeCount trades in log" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Recommendation:" -ForegroundColor Cyan
    Write-Host "  - Minimum: 5 completed trades for analysis"
    Write-Host "  - Good: 20+ trades for basic training"
    Write-Host "  - Ideal: 50-100 trades for reliable model"
    Write-Host ""
    Write-Host "Current status: Need more trading data" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        Write-Host "Cancelled. Run the bot to accumulate more trades." -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 0
    }
} elseif ($tradeCount -lt 20) {
    Write-Host "Status: >5 trades - Can analyze patterns" -ForegroundColor Green
    Write-Host "Recommendation: 20+ trades for better results" -ForegroundColor Cyan
    Write-Host ""
} elseif ($tradeCount -lt 50) {
    Write-Host "Status: >20 trades - Good for training" -ForegroundColor Green
    Write-Host "Recommendation: 50-100 trades for best results" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "Status: >50 trades - Excellent! ✓" -ForegroundColor Green
    Write-Host "Ready for reliable model training" -ForegroundColor Green
    Write-Host ""
}

# Create learning script if it doesn't exist
if (-not (Test-Path "learn_from_trades.py")) {
    Write-Host "Creating learning script..." -ForegroundColor Yellow
    
    @"
"""
Model Learning from Trade Log
Analyzes real trading results to improve predictions
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_trades(csv_path):
    """Analyze trade log and show insights"""
    print("Loading trade data...")
    df = pd.read_csv(csv_path)
    
    print(f"Total entries: {len(df)}")
    
    # Filter completed trades
    completed = df[df['status'].str.contains('CLOSED', na=False)]
    print(f"Completed trades: {len(completed)}")
    
    if len(completed) < 5:
        print("\n⚠️  Need at least 5 completed trades for analysis")
        return
    
    # Calculate statistics
    wins = completed[completed['status'] == 'CLOSED_WIN']
    losses = completed[completed['status'] == 'CLOSED_LOSS']
    
    win_rate = len(wins) / len(completed) * 100
    total_pnl = completed['profit_loss'].sum()
    
    print(f"\n{'='*60}")
    print(f"TRADING PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Total Trades:   {len(completed)}")
    print(f"Wins:           {len(wins)} ({len(wins)/len(completed)*100:.1f}%)")
    print(f"Losses:         {len(losses)} ({len(losses)/len(completed)*100:.1f}%)")
    print(f"Total P&L:      `${total_pnl:.2f}")
    
    if total_pnl > 0:
        avg_win = wins['profit_loss'].mean()
        avg_loss = abs(losses['profit_loss'].mean())
        profit_factor = wins['profit_loss'].sum() / abs(losses['profit_loss'].sum())
        print(f"Avg Win:        `${avg_win:.2f}")
        print(f"Avg Loss:       `${avg_loss:.2f}")
        print(f"Profit Factor:  {profit_factor:.2f}")
    
    # Analyze by confidence
    print(f"\n{'='*60}")
    print(f"WIN RATE BY AI CONFIDENCE")
    print(f"{'='*60}")
    
    completed['conf_bin'] = pd.cut(completed['ai_confidence'], 
                                     bins=[0, 0.5, 0.7, 0.85, 1.0],
                                     labels=['Low (<0.5)', 'Medium (0.5-0.7)', 
                                             'High (0.7-0.85)', 'Very High (>0.85)'])
    
    for conf_level in ['Low (<0.5)', 'Medium (0.5-0.7)', 'High (0.7-0.85)', 'Very High (>0.85)']:
        subset = completed[completed['conf_bin'] == conf_level]
        if len(subset) > 0:
            subset_wins = subset[subset['status']=='CLOSED_WIN']
            wr = len(subset_wins) / len(subset) * 100
            pnl = subset['profit_loss'].sum()
            print(f"{conf_level:20s}: {wr:5.1f}% win rate  |  {len(subset):3d} trades  |  P&L: `${pnl:7.2f}")
    
    # Recommendations
    print(f"\n{'='*60}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if len(completed) < 20:
        print("📊 Accumulate 20+ trades for basic model training")
    elif len(completed) < 50:
        print("📊 Accumulate 50-100 trades for reliable model training")
    else:
        print("✅ Excellent data - Ready for advanced model training!")
    
    # Analyze confidence performance
    high_conf = completed[completed['ai_confidence'] >= 0.7]
    low_conf = completed[completed['ai_confidence'] < 0.5]
    
    if len(high_conf) > 0 and len(low_conf) > 0:
        high_wr = len(high_conf[high_conf['status']=='CLOSED_WIN']) / len(high_conf) * 100
        low_wr = len(low_conf[low_conf['status']=='CLOSED_WIN']) / len(low_conf) * 100
        
        if high_wr > low_wr + 10:
            print("✅ High confidence trades perform better - Model is calibrated!")
            print(f"   • High confidence (>70%): {high_wr:.1f}% win rate")
            print(f"   • Low confidence (<50%): {low_wr:.1f}% win rate")
        else:
            print("⚠️  Confidence not well calibrated - Consider retraining")
            print(f"   • High confidence: {high_wr:.1f}% vs Low confidence: {low_wr:.1f}%")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Continue trading to accumulate more data")
    print(f"   2. With 50+ trades, train improved confidence model")
    print(f"   3. Filter out low-probability setups")
    print(f"   4. Retrain TFT model with latest market data")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python learn_from_trades.py trade_log.csv")
        sys.exit(1)
    
    analyze_trades(sys.argv[1])
"@ | Out-File -FilePath "learn_from_trades.py" -Encoding UTF8
}

Write-Host ""
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "STARTING MODEL LEARNING" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will:" -ForegroundColor Yellow
Write-Host "  1. Analyze win/loss patterns by confidence level"
Write-Host "  2. Identify most important prediction features"
Write-Host "  3. Calculate performance metrics"
Write-Host "  4. Show recommendations for improving win rate"
Write-Host ""

Start-Sleep -Seconds 2

python learn_from_trades.py $tradeLog

Write-Host ""
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "Learning complete!" -ForegroundColor Green
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"
