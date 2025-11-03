# Trading Bot - AI-Powered XAUUSD Trading System

Complete automated trading system using Temporal Fusion Transformer (TFT) model for XAUUSD (Gold) prediction and execution via MetaTrader 5.

## 🎯 Overview

This trading bot:
- Uses trained TFT model for 15-minute gold price predictions
- Generates trading signals based on quantile forecasts (q10/q50/q90)
- Executes trades automatically via MetaTrader 5
- Manages risk with position sizing, daily limits, and stop losses
- Includes trailing stops and partial profit-taking

## 📦 Components

### 1. **mt5_connector.py** - MetaTrader 5 Integration
- Connection management
- Historical data fetching
- Order execution (BUY/SELL)
- Position management
- Account information retrieval

### 2. **model_predictor.py** - TFT Model Inference
- Loads trained model checkpoint
- Generates predictions with confidence scores
- Provides quantile forecasts (q10, q50, q90)
- Calculates direction and expected move percentage

### 3. **strategy.py** - Trading Strategy
- Signal generation from predictions
- Entry/exit logic with reward:risk filtering
- Position management (trailing stops, partial closes)
- Time-of-day filters
- Market regime detection

### 4. **risk_manager.py** - Risk Management
- Position sizing based on account balance
- Daily risk limit tracking
- Maximum position limits
- Trade validation against risk rules
- Performance tracking

### 5. **bot.py** - Main Orchestrator
- Coordinates all components
- Main trading loop
- Manages open positions
- Executes new trades
- Logging and monitoring

### 6. **config.py** - Configuration
- Centralized settings for all components
- Model paths and parameters
- Strategy settings
- Risk management rules
- MT5 connection details

## 🚀 Quick Start

### Prerequisites

1. **Install Dependencies**
```bash
pip install -r trading_bot_requirements.txt
```

2. **MetaTrader 5 Setup**
- Install MT5 terminal
- Open a demo/live account
- Enable automated trading (Tools → Options → Expert Advisors → Allow automated trading)

3. **Model Files**
The bot auto-detects the latest model files from:
- `artifacts/checkpoints/` - Model checkpoint (.ckpt)
- `artifacts/scalers/` - Data scaler (.pkl)
- `artifacts/manifests/` - Feature manifest (.json)

### Configuration

Edit `config.py` to customize:

```python
# Risk management (AI-driven dynamic system)
RISK_CONFIG = {
    'account_balance': 10000.0,        # Starting balance
    'max_daily_loss': 4.0,             # 4% max daily loss
    'risk_low_confidence': 0.5,        # 0.5% for low confidence
    'risk_medium_confidence': 1.0,     # 1% for medium
    'risk_high_confidence': 2.0,       # 2% for high
    'max_lot_size': 2.0,               # Max 2.0 lots
    'rr_defensive': 0.5,               # 1:0.5 for low confidence
    'rr_normal': 1.0,                  # 1:1 for medium
    'rr_strong': 2.0,                  # 1:2 for strong
    'rr_high': 3.0,                    # 1:3 for high
    'max_spread_pips': 25,             # Skip if spread > 25
}

# Strategy settings
STRATEGY_CONFIG = {
    'min_confidence': 0.30,            # Min AI confidence (lowered for dynamic)
    'min_move_pct': 0.10,              # Min predicted move
    'enable_trailing_stop': True,
    'trailing_distance_pips': 50,
}

# MT5 connection (optional - uses default if None)
MT5_CONFIG = {
    'login': None,        # Your account number
    'password': None,     # Your password
    'server': None,       # Broker server
}
```

### Running the Bot

**Option 1: Using bot.py directly**
```bash
cd trading_bot
python bot.py
```

**Option 2: Custom script**
```python
from trading_bot.bot import TradingBot

config = {
    'symbol': 'XAUUSD',
    'timeframe': '15m',
    'update_interval': 60,
    'strategy_config': {...},
    'risk_config': {...},
}

bot = TradingBot(**config)
bot.run()
```

## 📊 Trading Logic

### AI-Driven Dynamic Risk Management

The bot uses an **advanced AI-driven risk system** that adapts position sizing and reward:risk targets based on prediction confidence:

#### 1. Dynamic Position Sizing

**Risk Formula:**
```
Risk Amount = Account Balance × Risk%
Lot Size = Risk Amount / (Stop Loss in pips × Pip Value)
```

**Confidence-Based Risk:**
| AI Confidence | Risk %   | Example ($10k account) |
|--------------|----------|------------------------|
| 0.30 - 0.49  | 0.5%     | $50 risk per trade     |
| 0.50 - 0.69  | 1.0%     | $100 risk per trade    |
| 0.70 - 0.84  | 1.0-2.0% | $100-$200 (graduated)  |
| 0.85 - 1.00  | 2.0%     | $200 risk per trade    |

**Cost Adjustments for XAUUSD:**
- Pip value: $1.00 per 0.01 lot (micro lot)
- Spread cost: ~$0.50 per 0.01 lot
- Commission: $6.00 per 1.0 lot round-turn
- All costs included in total risk calculation

#### 2. Dynamic Reward:Risk Ratios

The system automatically adjusts take profit targets based on confidence:

| AI Confidence | R:R Ratio | Strategy    |
|--------------|-----------|-------------|
| 0.30 - 0.49  | 1:0.5     | Defensive   |
| 0.50 - 0.69  | 1:1       | Normal      |
| 0.70 - 0.84  | 1:2       | Strong      |
| 0.85 - 1.00  | 1:3       | Aggressive  |

**Take Profit Calculation:**
```
TP = Entry + (Stop Loss Distance × R:R Ratio)
```

#### 3. Protection Mechanisms

**Daily Loss Limit:**
- Maximum 4% of account balance per day ($400 on $10k)
- Tracks realized + floating P&L
- Auto-closes all positions when limit hit
- Stops new trades until next day

**Consecutive Loss Protection:**
- Monitors winning/losing streaks
- After 2 consecutive losses:
  - Risk reduced by 50%
  - Example: 2% risk → 1% risk
- Reset on next winning trade

**Spread Filter:**
- Skips trades if spread > 25 pips
- Protects against high volatility entry

**Position Limits:**
- Max lot size: 2.0 lots
- Max open positions: 3
- Max leverage: 10x

### Signal Generation

1. **Get Prediction**
   - Fetch latest 128 bars of 15m XAUUSD data
   - Run through TFT model
   - Get q10, q50, q90 quantile predictions

2. **Generate Signal**
   - Check prediction confidence (must be ≥ 0.90)
   - Check predicted move size (must be ≥ 0.15%)
   - Calculate reward:risk ratio (must be ≥ 1.5)
   - Apply time-of-day filter (7 AM - 8 PM UTC)

3. **Entry Logic**
   - **BUY**: Direction = UP, Entry = current price, SL = q10
   - **SELL**: Direction = DOWN, Entry = current price, SL = q90
   - **TP**: Dynamically calculated as Entry ± (SL Distance × R:R)
     - Example: 150 pip SL, R:R 1:2 → 300 pip TP

### Example Trade Flow

**Scenario:** AI predicts XAUUSD will go UP with 82% confidence

1. **Prediction Analysis:**
   - Confidence: 0.82 → "Strong" category
   - Direction: UP
   - Predicted move: 0.25%

2. **Dynamic Risk Calculation:**
   - Confidence 0.82 → 2% risk (high confidence)
   - Account balance: $10,000
   - Risk amount: $200

3. **Position Sizing:**
   - Entry price: $2000.00
   - Stop loss: $1998.50 (150 pips)
   - Lot size: $200 / (150 × $1) = **1.33 lots**
   - Spread + commission: ~$8
   - Total risk: $208

4. **Take Profit:**
   - Confidence 0.82 → R:R 1:2 (strong)
   - TP: $2000.00 + (1.50 × 2) = **$2003.00** (300 pips)
   - Potential profit: 1.33 lots × 300 pips = **$399**

5. **Trade Execution:**
   - BUY 1.33 lots XAUUSD
   - Entry: 2000.00
   - SL: 1998.50
   - TP: 2003.00
   - Risk: $208 | Reward: $399 | R:R: 1:1.92

### Position Management

1. **Trailing Stops**
   - Automatically trail stop loss when in profit
   - Default: 50 pips trailing distance

2. **Partial Close**
   - Close 50% of position at first target (TP)
   - Let remaining 50% run to extended target

3. **Exit Logic**
   - Close if prediction reverses direction
   - Close if confidence drops below 70%
   - Close at stop loss or take profit

### Risk Management

1. **Position Sizing**
   - Calculates volume based on: Risk Amount = Balance × Risk%
   - Risk distance = |Entry - Stop Loss|
   - Volume = Risk Amount / (Contract Size × Risk Distance)

2. **Daily Limits**
   - Maximum 4% of balance at risk per day ($400 on $10k)
   - Maximum 2% of balance at risk per high-confidence trade
   - Maximum 3 concurrent positions

3. **Trade Validation**
   - All trades validated against risk rules
   - Checks dynamic reward:risk ratio
   - Checks leverage limits
   - Checks daily loss limits
   - Validates spread is acceptable

4. **Adaptive Risk**
   - Reduces risk by 50% after 2 consecutive losses
   - Adjusts position size for market volatility
   - Accounts for spread and commission costs

## 📈 Performance Metrics

The model was trained with:
- **Sharpe Ratio**: 32.79
- **Win Rate**: 58.5%
- **Directional Accuracy**: 54.58%
- **Data**: XAUUSD 2020-2024 (118,289 bars)
- **Folds**: 9 training folds completed

## 🔧 Customization

### Adjust Strategy Parameters

**More Conservative** (fewer trades, higher confidence):
```python
STRATEGY_CONFIG = {
    'min_confidence': 0.95,
    'min_move_pct': 0.20,
    'min_reward_risk': 2.0,
}
```

**More Aggressive** (more trades, lower thresholds):
```python
STRATEGY_CONFIG = {
    'min_confidence': 0.85,
    'min_move_pct': 0.10,
    'min_reward_risk': 1.2,
}
```

### Adjust Risk Parameters

**Lower Risk (Conservative)**:
```python
RISK_CONFIG = {
    'max_daily_loss': 2.0,           # 2% per day
    'risk_low_confidence': 0.25,     # 0.25% per trade
    'risk_medium_confidence': 0.5,   # 0.5% per trade
    'risk_high_confidence': 1.0,     # 1% per trade
    'max_lot_size': 1.0,             # Max 1.0 lot
}
```

**Higher Risk (Aggressive)**:
```python
RISK_CONFIG = {
    'max_daily_loss': 5.0,           # 5% per day
    'risk_low_confidence': 1.0,      # 1% per trade
    'risk_medium_confidence': 2.0,   # 2% per trade
    'risk_high_confidence': 3.0,     # 3% per trade
    'max_lot_size': 5.0,             # Max 5.0 lots
}
```

### Test Risk System

Run the test script to see how the dynamic risk system works:

```bash
cd trading_bot
python test_risk_system.py
```

This demonstrates:
- Dynamic risk calculation based on confidence
- Dynamic R:R ratio selection
- Position sizing with costs
- Consecutive loss protection
- Daily loss limit protection
- Complete trade examples

## 📝 Logging

Logs are written to:
- **File**: `trading_bot/bot.log`
- **Console**: Real-time output

Log includes:
- Predictions with confidence scores
- Trade signals and executions
- Position management actions
- Risk metrics and P&L
- Errors and warnings

## ⚠️ Safety Features

1. **Paper Trading Mode**
   - Set `BOT_CONFIG['paper_trading'] = True` to test without real money
   
2. **Daily Loss Limit**
   - Automatically stops trading if daily loss exceeds limit
   
3. **Position Limits**
   - Maximum concurrent positions to limit exposure
   
4. **Kill Switch**
   - Press Ctrl+C to stop bot immediately
   - All positions remain open for manual management

## 🔍 Monitoring

Check bot status:
```python
status = bot.get_status()
print(status)
```

Returns:
- Running state
- Account balance and equity
- Open positions count
- Last prediction details
- Risk metrics
- Daily P&L

## 🐛 Troubleshooting

**Bot won't connect to MT5:**
- Ensure MT5 terminal is running
- Check automated trading is enabled
- Verify account credentials

**No trades being taken:**
- Check prediction confidence is high enough
- Verify time filter allows current hour
- Check daily risk limit not exceeded
- Ensure reward:risk ratio meets threshold

**Model prediction errors:**
- Verify model files exist in artifacts/
- Check market data has sufficient bars (128 minimum)
- Ensure feature columns match training data

## 📚 Additional Resources

- Model training: See `ENHANCED_MODEL_README.md`
- Model performance: See `artifacts/metrics/`
- Feature engineering: See `src/features/`

## ⚖️ Disclaimer

This is a trading bot for educational/research purposes. Trading carries risk of financial loss. Always test thoroughly on a demo account before live trading. Past performance does not guarantee future results.

## 🤝 Support

For issues or questions:
1. Check logs in `trading_bot/bot.log`
2. Verify configuration in `config.py`
3. Test components individually (MT5 connection, model prediction, etc.)
