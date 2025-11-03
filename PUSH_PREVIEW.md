# Files to Push to GitHub

## ✅ New Files (Will be added)

### Documentation
- README.md - Main project documentation
- ALL_SCRIPTS_GUIDE.md - Complete script reference
- RUNNER_SCRIPTS.md - Quick start guide
- MARKET_HOURS.md - Trading hours rules  
- TRAINING_DATA_GUIDE.md - Data management guide

### Scripts
- run_trading_bot.ps1 - Start trading bot (PowerShell)
- run_trading_bot.bat - Start trading bot (CMD)
- run_learning.ps1 - Analyze trades (PowerShell)
- run_learning.bat - Analyze trades (CMD)
- run_training.ps1 - Train TFT model (PowerShell)
- push_to_github.ps1 - This update script
- update_training_data.py - Manage historical data
- verify_training_setup.py - Verify setup

### Trading Bot System
- trading_bot/bot.py - Main bot orchestrator
- trading_bot/strategy.py - Trading strategy
- trading_bot/risk_manager.py - Risk management
- trading_bot/model_predictor.py - TFT predictions
- trading_bot/mt5_connector.py - MetaTrader 5 API
- trading_bot/trade_logger.py - CSV logging
- trading_bot/view_trade_log.py - View statistics
- trading_bot/config.py - Configuration
- trading_bot/__init__.py

## ❌ Files to Delete (Will be removed)

### Old Files
- ENHANCED_MODEL_README.md
- EURUSD_15M.csv (replaced with XAUUSD_15M.csv)
- check_training_status.py
- enhancement_summary.py
- training_ready_summary.py
- training_time_estimate.py

### Old Checkpoints
- artifacts/checkpoints/tft_EURUSD_*.ckpt
- artifacts/checkpoints/tft_fast.ckpt
- artifacts/checkpoints/tft_stable.ckpt
- artifacts/scalers/scaler_EURUSD_*.pkl

### Removed Modules
- src/datasets/ (not needed)
- src/evaluation/offline_test.py
- src/features/features_15m_working.py
- src/features/simplified_features_15m.py

## 🔄 Modified Files (Will be updated)

### Core System
- src/config.py - Updated for XAUUSD, production settings
- src/training/train_tft.py - Enhanced training pipeline
- src/features/features_15m.py - Optimized features
- src/pipeline.py - Updated data pipeline
- requirements.txt - Current dependencies

### Configuration
- .gitignore - Updated exclusions

## 📊 Summary

- **Files to Add**: ~30 files
- **Files to Delete**: ~20 files
- **Files to Update**: ~10 files
- **Total Changes**: ~60 file operations

## 🎯 Result

After push, GitHub repo will contain:
✅ Complete trading bot system
✅ TFT model training code
✅ All runner scripts
✅ Comprehensive documentation
✅ No old/unused files
✅ Clean project structure

Ready to push? Run:
```powershell
.\push_to_github.ps1
```
