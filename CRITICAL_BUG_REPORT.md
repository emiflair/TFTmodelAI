# CRITICAL ISSUE FOUND AND FIXED

## Problem Summary
The model trained on Google Colab has a **CRITICAL TRAINING BUG** that makes predictions meaningless.

## Root Cause
The training script had a hardcoded bug on line 353:
```python
max_prediction_length=1  # WRONG! Should be config.data.horizon_bars (3)
```

This means:
- **Expected**: Model predicts 3 bars ahead (45 minutes on 15m timeframe)
- **Actual**: Model was trained to predict only 1 bar ahead (15 minutes)
- **Result**: Model predictions are for the wrong time horizon and produce NEUTRAL signals

## Evidence
Checked all 14 checkpoints in artifacts/checkpoints/:
```
tft_XAUUSD_15m_3B_latest.ckpt     enc=128 pred=1 (WRONG!)
last-v4.ckpt                      enc=128 pred=1 (WRONG!)
tft_fast.ckpt                     enc=128 pred=1 (WRONG!)
All Colab checkpoints             enc=128 pred=1 (WRONG!)
```

**All checkpoints have max_prediction_length=1 instead of 3**

## Fix Applied
Updated `src/training/train_tft.py` line 353:
```python
# Before (BUG):
max_prediction_length=1

# After (FIXED):
max_prediction_length = config.data.horizon_bars  # Will use 3 from config
```

## What This Means
1. ❌ Current model from Colab is **BROKEN** and cannot be used
2. ✅ Training script is now **FIXED**  
3. ⚠️ Model must be **RETRAINED** on Google Colab with fixed script

## Action Required
1. **Commit and push** the fixed training script to GitHub
2. **Retrain the model** on Google Colab using the fixed script
3. **Download** the new trained_model.zip
4. **Replace** the current checkpoints

## Additional Issues Found

### Issue #1: Old Checkpoint Being Loaded
- Bot was auto-detecting the OLDEST checkpoint (256 lookback) instead of newest
- **Fixed**: Added explicit checkpoint path in bot config
- **Fixed**: Improved auto-detection to check each parameter individually

### Issue #2: Configuration Mismatch
- Model trained with lookback_bars=128 but bot was configured for 256 bars
- **Fixed**: Updated bot to use 128 bars (fetch 200, check for 128 after features)

## Why The Model Predicted NEUTRAL

The model consistently predicted NEUTRAL with 0% moves because:

1. **Wrong prediction horizon**: Trained to predict 1 bar ahead (15 min) instead of 3 bars (45 min)
   - 1-bar ahead predictions have much smaller price movements
   - Model learned to predict the mean (current price) for safety
   
2. **Only 5 epochs**: train_metadata.json shows max_epochs=5 which is extremely short
   - Model didn't have enough training time to learn meaningful patterns
   - Needs at least 30-50 epochs for proper convergence

3. **Wrong target calculation**: Since max_prediction_length=1, the target (return_3) calculation was mismatched
   - Target was 3-bar return but model expected 1-bar return
   - This confusion led to predicting the mean

## Next Steps

1. **Push fixed code to GitHub**:
   ```bash
   git add src/training/train_tft.py
   git commit -m "FIX: Use horizon_bars for max_prediction_length instead of hardcoded 1"
   git push
   ```

2. **Retrain on Google Colab**:
   - Pull latest code
   - Run training with more epochs (30-50 recommended)
   - Verify checkpoint has max_prediction_length=3
   - Download trained_model.zip

3. **Deploy new model**:
   - Extract to artifacts/checkpoints/
   - Restart trading bot
   - Verify predictions show directional moves (not NEUTRAL)

## Expected Behavior After Fix

With the correctly trained model:
- **Predictions**: Should show UP/DOWN directions with non-zero move percentages
- **Confidence**: Variable based on market conditions (not always 100%)
- **Range**: q10, q50, q90 should differ (not all equal to current price)
- **Signals**: BUY/SELL signals should trigger when conditions met

## Technical Details

### Correct Training Configuration
```python
data:
  lookback_bars: 128        # Encoder length
  horizon_bars: 3           # Prediction length (45 min on 15m timeframe)
  
training:
  max_epochs: 30-50         # Need more epochs for convergence
  batch_size: 64
  learning_rate: 0.0002
```

### Correct Checkpoint Parameters
```python
dataset_parameters:
  max_encoder_length: 128
  max_prediction_length: 3   # MUST be 3, not 1!
  min_encoder_length: 128
  min_prediction_length: 3
```

### Bot Configuration
```python
- Fetch: 200 bars raw data
- After features: Need 128+ bars (drops ~70 bars due to indicators)
- Model input: 128 encoder + 3 prediction = 131 total rows required
```

## Verification Checklist

After retraining, verify:
- [ ] Checkpoint max_prediction_length = 3 (not 1)
- [ ] Model trained for 30+ epochs (not 5)
- [ ] Bot loads correct checkpoint (tft_XAUUSD_15m_3B_latest.ckpt)
- [ ] Predictions show UP/DOWN (not always NEUTRAL)
- [ ] Move percentages are non-zero (not 0.00%)
- [ ] Confidence varies (not always 100%)
- [ ] q10 ≠ q50 ≠ q90 (not all equal)

## Summary

**The model from Colab cannot work** because it was trained with the wrong prediction horizon (1 bar instead of 3 bars). The training script has been fixed. **You must retrain the model on Colab** with the corrected script to get a working trading bot.
