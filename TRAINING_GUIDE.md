# Local Training Guide - XAUUSD TFT Model

## Quick Start

### 1. Train Model on CPU (Your PC)

```powershell
# Quick test (1 epoch, 1 split - ~5 minutes)
.venv\Scripts\python.exe train_local.py --fast-dev-run

# Train on 1 split only (~30-60 minutes)
.venv\Scripts\python.exe train_local.py --splits 1 --max-epochs 20

# Full training on all splits (several hours)
.venv\Scripts\python.exe train_local.py
```

### 2. Run Trading Bot

```powershell
# Bot will auto-detect the latest checkpoint
.venv\Scripts\python.exe trading_bot\bot.py
```

## What Changed

✅ **Deleted:**
- All GPU-trained checkpoints (9 files)
- Colab notebooks (no longer needed)
- Old training logs

✅ **Updated:**
- Training script now uses `accelerator='cpu'` 
- Config auto-detects latest checkpoint
- Created `train_local.py` for easy local training

✅ **Benefits:**
- Checkpoints work on any PC (CPU-compatible)
- No CUDA/GPU driver issues
- Full control over training on your machine

## Training Options

```
--fast-dev-run          Quick test (1 epoch, 1 split)
--max-epochs 10         Set max epochs (default: 30)
--splits 1              Train on N splits only (default: all)
--batch-size 64         Set batch size (default: 128)
--learning-rate 0.001   Set learning rate
--resume last           Resume from last checkpoint
```

## Expected Training Time (CPU)

- **Fast dev run**: ~5 minutes
- **1 split, 20 epochs**: ~30-60 minutes  
- **Full training (3 splits)**: ~2-4 hours

## Output Files

After training, you'll find:
- `artifacts/checkpoints/tft_XAUUSD_15m_3B_YYYYMMDD_YYYYMMDD.ckpt` - Model checkpoint
- `artifacts/scalers/scaler_XAUUSD_fold*.pkl` - Data scalers
- `artifacts/manifests/feature_manifest.json` - Feature definitions
- `artifacts/metrics/` - Training metrics and reports

## Next Steps

1. **Start with fast dev run** to test setup:
   ```powershell
   .venv\Scripts\python.exe train_local.py --fast-dev-run
   ```

2. **If successful, train on 1 split**:
   ```powershell
   .venv\Scripts\python.exe train_local.py --splits 1 --max-epochs 20
   ```

3. **Run the bot** (it auto-detects the checkpoint):
   ```powershell
   .venv\Scripts\python.exe trading_bot\bot.py
   ```

## Troubleshooting

**"No module named 'src'"**
- Run from project root directory
- Make sure you're in: `C:\Users\emife\OneDrive\Desktop\TFTmodelAI`

**Training is slow**
- Normal on CPU! GPU is 10-50x faster
- Use `--fast-dev-run` for quick tests
- Use `--splits 1` to train on less data

**Out of memory**
- Reduce batch size: `--batch-size 32`
- Close other applications
- CPU training uses RAM, not VRAM
