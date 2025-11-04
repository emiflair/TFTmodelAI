# AI Trading Bot - XAUUSD (Gold) 15M

Automated trading system using Temporal Fusion Transformer (TFT) for XAUUSD predictions on MetaTrader 5.

## 🎯 Features

### Core Trading System
- **AI Model**: Temporal Fusion Transformer (TFT) with 739K parameters
- **Timeframe**: 15-minute bars
- **Symbol**: XAUUSD (Gold)
- **Position Management**: Max 2 positions, same direction only
- **Risk Management**: Dynamic position sizing based on AI confidence
- **Market Hours**: Automatic protection (skip first 2h after open, last 1h before close)

### Smart Exit Logic
- Keep winning positions > $100 profit
- Close losing positions immediately
- Close small winners (<$100) on reversal with 65%+ confidence

### Daily Limits
- **Loss Limit**: 4% of account ($400 on $10k)
- **Profit Target**: 5% of account ($500 on $10k)

### Trade Logging
- Complete 31-column CSV logging
- Captures: Entry/exit, AI predictions, risk metrics, outcomes
- Used for model improvement and analysis

## 📊 System Architecture

```
Historical Data (XAUUSD_15M.csv)
         ↓
    TFT Model Training
         ↓
  Trained Model (739K params)
         ↓
    Trading Bot (bot.py)
         ↓
    Live Predictions
         ↓
  Risk Manager → Strategy → MT5 Execution
         ↓
   Trade Log (CSV)
         ↓
  Learning & Improvement
```

## 🚀 Quick Start

### 1. Setup

```bash
# Clone repository
git clone https://github.com/emiflair/TFTmodel.git
cd TFTmodel

# Install dependencies
pip install -r requirements.txt

# Configure MT5 connection
# Edit trading_bot/config.py with your account details
```

### 2. Run Trading Bot

**PowerShell (Recommended):**
```powershell
.\run_trading_bot.ps1
```

**Command Prompt:**
```bash
run_trading_bot.bat
```

### 3. Analyze Performance

After accumulating 20+ trades:
```powershell
.\run_learning.ps1
```

### 4. Retrain Model

Monthly or when adding new data:
```powershell
.\run_training.ps1
```

## 🧭 Train on Google Colab (GPU)

If you prefer training in Google Colab with a free GPU:

1) Upload this project folder to Google Drive, e.g., `MyDrive/TFTmodelAI` and ensure `XAUUSD_15M.csv` is in the project root.

2) Open the notebook `colab_train_tft.ipynb` in Colab and run all cells:

     - It will mount Drive, install dependencies, verify the setup, and launch training.
     - Artifacts (checkpoints, scalers, metrics, manifests) are saved under `artifacts/` in your Drive folder.

Notes:
- In Colab: Runtime → Change runtime type → Hardware accelerator: GPU
- If pip shows conflicts, restart the runtime and re-run from the install cell.
- If no GPU is detected, re-check the runtime setting.

## 📁 Project Structure

```
TFTmodel/
├── trading_bot/           # Main trading bot system
│   ├── bot.py            # Main bot orchestrator
│   ├── strategy.py       # Trading strategy logic
│   ├── risk_manager.py   # Risk & position management
│   ├── model_predictor.py # TFT model interface
│   ├── mt5_connector.py  # MetaTrader 5 API
│   ├── trade_logger.py   # CSV trade logging
│   └── config.py         # Configuration
│
├── src/                   # TFT model training
│   ├── training/         # Model training scripts
│   ├── features/         # Feature engineering
│   ├── preprocessing/    # Data preprocessing
│   └── evaluation/       # Performance metrics
│
├── artifacts/            # Trained models & scalers
│   ├── checkpoints/      # TFT model checkpoints
│   ├── scalers/          # Feature scalers
│   └── metrics/          # Training metrics
│
├── run_trading_bot.ps1   # Start trading bot
├── run_learning.ps1      # Analyze trade performance
├── run_training.ps1      # Train TFT model
├── update_training_data.py # Manage historical data
└── requirements.txt      # Python dependencies
```

## 🔧 Configuration

### Trading Bot (`trading_bot/config.py`)

```python
MT5_ACCOUNT = 52587216           # Your MT5 account
MT5_PASSWORD = "YourPassword"    # Your MT5 password
MT5_SERVER = "ICMarketsSC-Demo"  # Your MT5 server
SYMBOL = "XAUUSD"                # Trading symbol
TIMEFRAME = mt5.TIMEFRAME_M15    # 15-minute bars
```

### Risk Parameters

```python
MAX_DAILY_LOSS = 4.0      # % of balance ($400 on $10k)
MAX_DAILY_PROFIT = 5.0    # % of balance ($500 on $10k)
MAX_OPEN_POSITIONS = 2    # Maximum concurrent positions
SAME_DIRECTION_ONLY = True # Only trade one direction at a time
```

### Dynamic Risk by Confidence

| Confidence | Risk % | R:R Ratio |
|------------|--------|-----------|
| < 0.5 | 0.5% | 1:0.5 |
| 0.5 - 0.7 | 1.0% | 1:1.0 |
| 0.7 - 0.85 | 1.0-2.0% | 1:2.0 |
| > 0.85 | 2.0% | 1:3.0 |

## 📈 Performance Tracking

### Trade Log Structure (31 columns)

**Trade Details:**
- timestamp, trade_id, ticket, symbol, action
- volume, entry_price, exit_price, sl, tp

**AI Predictions:**
- ai_direction, ai_confidence, predicted_move_pct
- predicted_q10, predicted_q50, predicted_q90

**Risk Metrics:**
- risk_amount, risk_pct, rr_ratio

**Outcomes:**
- status (OPEN/CLOSED_WIN/CLOSED_LOSS)
- pips, profit_loss, profit_loss_pct
- duration_minutes, spread, slippage

**Account State:**
- balance_before, balance_after, equity

### View Statistics

```bash
python trading_bot/view_trade_log.py
```

Output:
```
Trading Statistics:
  Total Trades: 47
  Wins: 32 (68.1%)
  Losses: 15 (31.9%)
  Total P&L: $1,234.56
  Win Rate by Confidence:
    High (>0.7): 75% win rate
    Low (<0.5): 50% win rate
```

## 🧠 Model Details

### TFT Architecture
- **Type**: Temporal Fusion Transformer
- **Parameters**: 739,270
- **Hidden Size**: 160
- **Attention Heads**: 4
- **Dropout**: 0.1
- **Training**: Walk-forward validation

### Features Used
- **Price**: OHLC, returns, volatility
- **Technical**: RSI, MACD, Bollinger Bands, ATR
- **Volume**: Tick volume, volume MA
- **Time**: Hour/day cyclical encoding
- **Session**: Asian/European/US session indicators

### Training Data
- **Source**: XAUUSD 15-minute bars
- **Period**: Jan 2024 - Oct 2025 (43,542 bars)
- **Size**: 2.69 MB
- **Quality**: 80/100 (474 gaps from weekends/holidays)

## 🛡️ Risk Controls

### Position Limits
✅ Maximum 2 open positions  
✅ Same direction only (no hedging)  
✅ Auto-close opposite positions before new trade  

### Daily Limits
✅ Stop trading at 4% daily loss  
✅ Stop trading at 5% daily profit  
✅ Reset at midnight each day  

### Market Hours Protection
⛔ Skip first 2 hours after market open (Sunday 23:00 - Monday 01:00)  
⛔ Skip last 1 hour before close (Friday 21:00 - 22:00)  
⛔ No trading on weekends  

### Smart Exits
✅ Keep winners > $100 profit  
⛔ Close losers immediately  
⛔ Close small winners (<$100) on reversal with high confidence  

### Spread/Slippage Protection
✅ Max spread: 25 pips  
✅ Slippage tolerance: 10 pips  
✅ Reject trades with excessive costs  

## 📊 Learning from Trades

### Analyze Performance

After 20+ trades:
```bash
.\run_learning.ps1
```

Shows:
- Win rate by confidence level
- Most important prediction features
- Performance by market condition
- Recommendations for improvement

### Retrain with Trade Data

After 50-100 trades:
1. Analyze which confidence levels work best
2. Train improved confidence model
3. Filter out historically losing setups
4. Improve overall win rate by 10-20%

## 🔄 Maintenance

### Daily
- Monitor bot performance
- Check trade log for any issues
- Verify account balance and equity

### Weekly
- Run learning analysis (after 10-20 new trades)
- Review which setups are working
- Adjust strategy if needed

### Monthly
- Update historical data (XAUUSD_15M.csv)
- Retrain TFT model with latest data
- Review overall performance metrics

## 📝 Requirements

### Software
- Python 3.9+
- MetaTrader 5
- Windows OS (for MT5)

### Python Packages
```
pandas>=2.0.0
numpy>=1.24.0
pytorch>=2.0.0
pytorch-forecasting>=1.0.0
MetaTrader5>=5.0.0
rich>=13.0.0
scikit-learn>=1.3.0
```

Install all:
```bash
pip install -r requirements.txt
```

### Hardware
- **Minimum**: 8GB RAM, 4-core CPU
- **Recommended**: 16GB RAM, 8-core CPU
- **GPU**: Optional (speeds up training)

## 🐛 Troubleshooting

### Bot Won't Start
- Check MT5 is running
- Verify account credentials in config.py
- Ensure symbol XAUUSD is in Market Watch

### No Trades Executing
- Check if market hours are valid
- Verify daily limits not reached
- Ensure spread is acceptable (<25 pips)
- Check if predictions meet confidence threshold

### Trade Log Not Created
- Bot must execute at least one trade
- Check permissions for writing to directory
- Verify trade_logger.py is present

### Model Predictions Poor
- May need retraining with recent data
- Check if market regime has changed
- Verify feature calculations are correct

## 📄 License

This project is for educational and personal use. Use at your own risk.

## ⚠️ Disclaimer

**Trading involves substantial risk of loss.** This bot is provided as-is with no guarantees of profitability. Past performance does not guarantee future results. Always test thoroughly on demo accounts before risking real money.

## 🤝 Contributing

This is a personal project, but suggestions and bug reports are welcome via GitHub issues.

## 📧 Contact

- GitHub: [@emiflair](https://github.com/emiflair)
- Repository: [TFTmodel](https://github.com/emiflair/TFTmodel)

## 🎯 Roadmap

- [x] Basic TFT model training
- [x] MT5 integration
- [x] Risk management system
- [x] Trade logging
- [x] Market hours protection
- [x] Smart exit logic
- [ ] Multi-timeframe analysis
- [ ] Advanced trade log learning
- [ ] Automated model retraining
- [ ] Web dashboard for monitoring
- [ ] Support for multiple symbols

---

**Last Updated**: November 3, 2025  
**Version**: 2.0.0  
**Status**: Production Ready ✅
