# All Available Scripts - Quick Reference

## 📊 Three Main Operations

### 1. 🤖 RUN TRADING BOT (Live Trading)
**What it does**: Starts the AI bot to trade XAUUSD in real-time

**Scripts**:
- `run_trading_bot.bat` (Command Prompt)
- `run_trading_bot.ps1` (PowerShell) ⭐ Recommended

**When to use**: Daily trading operations

**Command**:
```powershell
.\run_trading_bot.ps1
```

---

### 2. 📈 LEARN FROM TRADES (Trade Log Analysis)
**What it does**: Analyzes your actual trading results to improve model

**Scripts**:
- `run_learning.bat` (Command Prompt)
- `run_learning.ps1` (PowerShell) ⭐ Recommended

**When to use**: After 20+ completed trades to see patterns

**Command**:
```powershell
.\run_learning.ps1
```

---

### 3. 🧠 TRAIN TFT MODEL (Historical Data Training)
**What it does**: Trains the base TFT model on historical XAUUSD price data

**Scripts**:
- `run_training.ps1` (PowerShell)

**When to use**: 
- Initial setup (already done)
- Monthly retraining with new historical data
- After updating XAUUSD_15M.csv

**Command**:
```powershell
.\run_training.ps1
```

---

## 🔄 Typical Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: SETUP (One Time)                                  │
└─────────────────────────────────────────────────────────────┘
  1. Download historical data → XAUUSD_15M.csv
  2. Run: .\run_training.ps1
  3. Wait 2-4 hours for training
  4. Model saved → artifacts/checkpoints/

┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: TRADING (Daily)                                   │
└─────────────────────────────────────────────────────────────┘
  1. Run: .\run_trading_bot.ps1
  2. Bot trades automatically
  3. Logs every trade → trade_log.csv
  4. Let run for weeks to accumulate data

┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: IMPROVEMENT (Weekly/Monthly)                      │
└─────────────────────────────────────────────────────────────┘
  1. After 20+ trades, run: .\run_learning.ps1
  2. Review which confidence levels win
  3. Adjust strategy based on insights
  4. Continue trading

┌─────────────────────────────────────────────────────────────┐
│  PHASE 4: RETRAINING (Monthly)                              │
└─────────────────────────────────────────────────────────────┘
  1. Update XAUUSD_15M.csv with new data
  2. Run: .\run_training.ps1
  3. New model learns latest patterns
  4. Deploy updated model to bot
```

---

## 📁 File Locations

### Scripts (in TFTmodel/)
```
run_trading_bot.bat       - CMD: Start trading bot
run_trading_bot.ps1       - PS: Start trading bot ⭐
run_learning.bat          - CMD: Analyze trades
run_learning.ps1          - PS: Analyze trades ⭐
run_training.ps1          - PS: Train TFT model
```

### Data Files
```
XAUUSD_15M.csv                              - Historical training data
trading_bot/trading_bot/trade_log.csv       - Live trading results
```

### Output/Models
```
artifacts/checkpoints/                      - Trained TFT models
artifacts/scalers/                          - Feature scalers
learn_from_trades.py                        - Auto-generated analysis script
```

---

## ⚡ Quick Commands

### Start Trading
```powershell
.\run_trading_bot.ps1
```

### Check Performance
```powershell
python trading_bot\view_trade_log.py
```

### Analyze for Learning
```powershell
.\run_learning.ps1
```

### Retrain Base Model
```powershell
.\run_training.ps1
```

### Stop Bot
```
Press Ctrl+C in the terminal
```

---

## 🎯 What Each Script Shows

### run_trading_bot.ps1
```
====================================================================
AI TRADING BOT - STARTING
====================================================================

Configuration:
  - Symbol: XAUUSD
  - Timeframe: 15M
  - Max Positions: 2 (same direction only)
  - Daily Loss Limit: 4% ($400)
  - Daily Profit Target: 5% ($500)

Start trading bot? (y/n): _
```

### run_learning.ps1
```
====================================================================
MODEL LEARNING FROM TRADE LOG
====================================================================

Found trade log: trading_bot\trading_bot\trade_log.csv
Total entries: 47

Status: >20 trades - Good for training

WIN RATE BY AI CONFIDENCE
============================================================
Low (<0.5):        50.0% win rate | 8 trades
Medium (0.5-0.7):  65.2% win rate | 23 trades
High (0.7-0.85):   75.0% win rate | 12 trades
Very High (>0.85): 100.0% win rate | 4 trades
```

### run_training.ps1
```
====================================================================
TFT MODEL TRAINING - HISTORICAL DATA
====================================================================

Training Data:
  File: XAUUSD_15M.csv
  Rows: 43,542 bars (15-minute)
  Size: 2.69 MB

Estimated training time: ~4 hours

Start TFT model training? (y/n): _
```

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| PowerShell won't run scripts | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Python not found | Add Python to PATH or use full path |
| Trade log not found | Run bot first to generate trades |
| Training data missing | Download XAUUSD_15M.csv |
| Bot stops immediately | Check MT5 connection, credentials |

---

## 💡 Pro Tips

1. **Use PowerShell (.ps1)** - Better formatting and error handling
2. **Run bot in background** - Use Windows Task Scheduler
3. **Monitor daily** - Check trade log and performance
4. **Backup trade log** - It's valuable training data
5. **Analyze weekly** - Run learning after 10-20 new trades
6. **Retrain monthly** - Update model with latest market data

---

## 📝 Script Comparison

| Feature | Trading Bot | Learning | TFT Training |
|---------|------------|----------|--------------|
| **Purpose** | Execute trades | Analyze results | Train base model |
| **Input** | Live market data | trade_log.csv | XAUUSD_15M.csv |
| **Output** | Trades + log | Performance report | TFT model |
| **Duration** | Continuous | 10 seconds | 2-4 hours |
| **Frequency** | Daily | Weekly | Monthly |
| **Required** | Yes (main operation) | Optional (improvement) | Initial + monthly |

---

## 🚀 Getting Started

**First time?** Run these in order:

1. ✅ Train model: `.\run_training.ps1` (one time, 2-4 hours)
2. ✅ Start trading: `.\run_trading_bot.ps1` (daily)
3. ✅ Wait for 20+ trades (1-2 weeks)
4. ✅ Analyze: `.\run_learning.ps1` (see what works)
5. ✅ Optimize based on results
6. ✅ Retrain monthly with new data

**That's it!** 🎯
