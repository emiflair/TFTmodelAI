"""
CLEAN TRAINING SCRIPT FOR TFT MODEL
====================================

This script trains a Temporal Fusion Transformer model for XAUUSD 15-minute predictions.

Key Parameters:
- Lookback: 128 bars (32 hours of 15-min data)
- Horizon: 3 bars (45 minutes ahead)
- Target: 3-bar forward return
- Quantiles: [0.1, 0.5, 0.9] for risk-aware predictions

Usage:
    python train_model_clean.py
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import pytorch_forecasting as pf
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data parameters
    'csv_path': 'XAUUSD_15M.csv',
    'lookback_bars': 128,        # Encoder length: 128 bars = 32 hours
    'horizon_bars': 3,           # Prediction length: 3 bars = 45 minutes
    
    # Train/val/test split
    'train_pct': 0.70,           # 70% for training
    'val_pct': 0.15,             # 15% for validation
    'test_pct': 0.15,            # 15% for testing
    
    # Model architecture
    'hidden_size': 160,          # Hidden layer size
    'attention_head_size': 4,    # Number of attention heads
    'dropout': 0.15,             # Dropout rate
    'hidden_continuous_size': 8, # Size of hidden continuous layers
    
    # Training parameters
    'batch_size': 64,
    'learning_rate': 0.0003,
    'max_epochs': 30,            # Train for 30 epochs
    'gradient_clip_val': 0.1,
    'early_stop_patience': 8,
    
    # Loss function
    'quantiles': [0.1, 0.5, 0.9],  # Predict 10th, 50th, 90th percentiles
    
    # Output paths
    'checkpoint_dir': 'artifacts/checkpoints',
    'logs_dir': 'lightning_logs',
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def set_seed(seed=42):
    """Set random seeds for reproducibility (torch + numpy + lightning)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic for reproducible attention mask creation on GPU
    pl.seed_everything(seed, workers=True)
    torch.use_deterministic_algorithms(False)

def load_and_prepare_data(csv_path):
    """Load CSV and standardize columns, robust to different time/header casings."""
    print(f"\n📊 Loading data from {csv_path}...")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df):,} rows")
    print(f"   Raw columns: {list(df.columns)}")

    # Normalize column names (lowercase strip)
    canon_map = {c: c.strip().lower() for c in df.columns}

    # Possible time column variants
    time_candidates = [
        'time (utc)', 'time_utc', 'timestamp', 'time', 'date', 'datetime'
    ]
    found_time_col = None
    for col, lc in canon_map.items():
        if lc in time_candidates:
            found_time_col = col
            break

    if found_time_col is None:
        raise KeyError(f"No time column found. Expected one of {time_candidates}. Got: {list(df.columns)}")

    # Rename to standard schema (case-insensitive matching)
    rename_rules = {}
    # Core OHLCV mapping
    for original in df.columns:
        lc = original.lower()
        if original == found_time_col:
            rename_rules[original] = 'timestamp'
        elif lc == 'open':
            rename_rules[original] = 'open'
        elif lc == 'high':
            rename_rules[original] = 'high'
        elif lc == 'low':
            rename_rules[original] = 'low'
        elif lc == 'close':
            rename_rules[original] = 'close'
        elif lc == 'volume':
            rename_rules[original] = 'volume'

    df = df.rename(columns=rename_rules)

    required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after rename: {missing}. Present: {list(df.columns)}")

    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"   Using time column: {found_time_col} -> 'timestamp'")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Close price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    return df

def create_features(df, lookback, horizon):
    """Create features for TFT model"""
    print(f"\n🔧 Creating features...")
    
    df = df.copy()
    
    # Time index (monotonically increasing)
    df['time_idx'] = np.arange(len(df))
    
    # Series ID (required by TFT)
    df['series_id'] = 'XAUUSD'
    
    # Basic price features
    df['return_1'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility features
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['atr_proxy'] = df['high_low_range'].rolling(14, min_periods=1).mean()
    
    # Trend features
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['close'].rolling(period, min_periods=1).mean()
        df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
    
    # Momentum features
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['close'].pct_change(period)
        df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
    
    # Volume features
    df['volume_ma_5'] = df['volume'].rolling(5, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_5']
    
    # Hour of day (cyclical encoding)
    df['hour'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day of week (cyclical encoding)
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # TARGET: N-bar forward return
    # This is what the model will predict
    df['target'] = df['close'].pct_change(horizon).shift(-horizon)
    
    # Drop NaN rows (from feature calculation and target)
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    
    print(f"   Created {len([c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} features")
    print(f"   Dropped {dropped_rows} rows with NaN values")
    print(f"   Final dataset: {len(df):,} rows")
    
    return df

def split_data(df, train_pct, val_pct, test_pct):
    """Split data into train/val/test sets"""
    print(f"\n✂️  Splitting data: {train_pct:.0%} train / {val_pct:.0%} val / {test_pct:.0%} test")
    
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"   Train: {len(train_df):,} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"   Val:   {len(val_df):,} rows ({val_df['timestamp'].min()} to {val_df['timestamp'].max()})")
    print(f"   Test:  {len(test_df):,} rows ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
    
    return train_df, val_df, test_df

def create_dataset(train_df, val_df, config):
    """Create TimeSeriesDataSet for TFT"""
    print(f"\n🎯 Creating TimeSeriesDataSet...")
    print(f"   Encoder length: {config['lookback_bars']} bars")
    print(f"   Prediction length: {config['horizon_bars']} bars")
    print(f"   Total required: {config['lookback_bars'] + config['horizon_bars']} bars per sample")
    
    # Define feature groups
    time_varying_known = []  # Features known in advance (e.g., time)
    time_varying_unknown = [  # Features not known in advance
        'return_1', 'log_return', 'high_low_range', 'atr_proxy',
        'price_to_sma_5', 'price_to_sma_10', 'price_to_sma_20', 'price_to_sma_50',
        'momentum_5', 'momentum_10', 'momentum_20',
        'roc_5', 'roc_10', 'roc_20',
        'volume_ratio',
        'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos'
    ]
    
    # Create training dataset
    training = TimeSeriesDataSet(
        train_df,
        time_idx='time_idx',
        target='target',
        group_ids=['series_id'],
        min_encoder_length=config['lookback_bars'],
        max_encoder_length=config['lookback_bars'],
        min_prediction_length=config['horizon_bars'],
        max_prediction_length=config['horizon_bars'],  # CRITICAL: Must match horizon_bars!
        time_varying_known_reals=time_varying_known,
        time_varying_unknown_reals=time_varying_unknown,
        static_categoricals=['series_id'],
        target_normalizer=GroupNormalizer(groups=['series_id'], transformation='softplus'),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )
    
    # Create validation dataset
    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)
    
    print(f"   Training samples: {len(training):,}")
    print(f"   Validation samples: {len(validation):,}")
    
    return training, validation

def create_model(dataset, config):
    """Create TFT model"""
    print(f"\n🤖 Creating TFT model...")
    print(f"   Hidden size: {config['hidden_size']}")
    print(f"   Attention heads: {config['attention_head_size']}")
    print(f"   Dropout: {config['dropout']}")
    print(f"   Learning rate: {config['learning_rate']}")
    
    model = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=config['learning_rate'],
        hidden_size=config['hidden_size'],
        attention_head_size=config['attention_head_size'],
        dropout=config['dropout'],
        hidden_continuous_size=config['hidden_continuous_size'],
        loss=QuantileLoss(quantiles=config['quantiles']),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return model

def train_model(model, train_dataset, val_dataset, config):
    """Train the TFT model"""
    print(f"\n🚀 Starting training...")
    print(f"   Max epochs: {config['max_epochs']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Early stopping patience: {config['early_stop_patience']}")
    
    # Create data loaders
    train_dataloader = train_dataset.to_dataloader(
        train=True,
        batch_size=config['batch_size'],
        num_workers=2 if torch.cuda.is_available() else 0,
        persistent_workers=torch.cuda.is_available(),
        pin_memory=torch.cuda.is_available(),
    )
    val_dataloader = val_dataset.to_dataloader(
        train=False,
        batch_size=config['batch_size'] * 2,
        num_workers=2 if torch.cuda.is_available() else 0,
        persistent_workers=torch.cuda.is_available(),
        pin_memory=torch.cuda.is_available(),
    )
    
    # Create callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-4,
        patience=config['early_stop_patience'],
        verbose=True,
        mode='min'
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['checkpoint_dir'],
        filename='tft_XAUUSD_15m_best',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
    )
    
    lr_logger = LearningRateMonitor(logging_interval='epoch')
    
    # Create logger (optional; disable if tensorboard unavailable locally)
    try:
        from lightning.pytorch.loggers import TensorBoardLogger as TBL
        logger = TBL(save_dir=config['logs_dir'], name='tft_training')
    except (ImportError, ModuleNotFoundError):
        logger = None  # Train without TensorBoard logger
    
    # Detect accelerator
    if torch.cuda.is_available():
        accelerator = 'auto'
        devices = 1
        print(f"   Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        accelerator = 'cpu'
        devices = 1
        print(f"   Using CPU (training will be slower)")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator=accelerator,
        devices=devices,
        precision='32-true',  # Use FP32 to avoid FP16 overflow in attention mask
        gradient_clip_val=config['gradient_clip_val'],
        callbacks=[early_stop_callback, checkpoint_callback, lr_logger],
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )
    
    # Train!
    print(f"\n{'='*70}")
    print(f"TRAINING STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # Load best model
    best_model_path = checkpoint_callback.best_model_path
    print(f"✅ Best model saved to: {best_model_path}")
    print(f"   Best validation loss: {checkpoint_callback.best_model_score:.6f}")
    
    return trainer, best_model_path

def evaluate_model(model, test_df, train_dataset, config):
    """Evaluate model on test set"""
    print(f"\n📈 Evaluating model on test set...")
    
    # Create test dataset
    test_dataset = TimeSeriesDataSet.from_dataset(train_dataset, test_df, predict=False, stop_randomization=True)
    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=config['batch_size'] * 2, num_workers=0)
    
    # Get predictions
    predictions = model.predict(test_dataloader, return_x=True, return_y=True)
    
    # Calculate metrics
    actuals = predictions.y[0].cpu().numpy()
    pred_output = predictions.output['prediction'].cpu().numpy()
    
    # Handle different output shapes (batch, horizon, quantiles) or (batch, quantiles)
    if pred_output.ndim == 3:
        preds_q50 = pred_output[:, :, 1]  # Median prediction across horizon
    elif pred_output.ndim == 2:
        preds_q50 = pred_output[:, 1]  # Median prediction (single horizon)
    else:
        preds_q50 = pred_output  # Fallback
    
    # Flatten if needed
    if actuals.ndim > 1 and actuals.shape[1] == 1:
        actuals = actuals[:, 0]
    if preds_q50.ndim > 1 and preds_q50.shape[1] == 1:
        preds_q50 = preds_q50[:, 0]
    
    # Calculate errors
    mae = np.mean(np.abs(actuals - preds_q50))
    rmse = np.sqrt(np.mean((actuals - preds_q50) ** 2))
    mape = np.mean(np.abs((actuals - preds_q50) / (actuals + 1e-8))) * 100
    
    # Directional accuracy
    actual_direction = np.sign(actuals)
    pred_direction = np.sign(preds_q50)
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    print(f"\n   📊 Test Set Metrics:")
    print(f"   MAE:  {mae:.6f}")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   Directional Accuracy: {directional_accuracy:.2f}%")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'directional_accuracy': directional_accuracy
    }

def verify_checkpoint(checkpoint_path):
    """Verify the trained checkpoint has correct parameters"""
    print(f"\n🔍 Verifying checkpoint...")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'dataset_parameters' in checkpoint:
        params = checkpoint['dataset_parameters']
        encoder_len = params.get('max_encoder_length')
        pred_len = params.get('max_prediction_length')
        
        print(f"   ✓ Checkpoint parameters:")
        print(f"     - max_encoder_length: {encoder_len}")
        print(f"     - max_prediction_length: {pred_len}")
        
        if pred_len != CONFIG['horizon_bars']:
            print(f"   ⚠️  WARNING: max_prediction_length ({pred_len}) != horizon_bars ({CONFIG['horizon_bars']})")
            return False
        else:
            print(f"   ✅ Checkpoint parameters are CORRECT!")
            return True
    else:
        print(f"   ❌ No dataset_parameters found in checkpoint!")
        return False

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print("  TFT MODEL TRAINING - CLEAN VERSION")
    print("="*70)
    
    # Set seed
    set_seed(42)
    print("✓ Random seed set to 42")
    
    # Create output directories
    Path(CONFIG['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(CONFIG['logs_dir']).mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_and_prepare_data(CONFIG['csv_path'])
    
    # Create features
    df = create_features(df, CONFIG['lookback_bars'], CONFIG['horizon_bars'])
    
    # Split data
    train_df, val_df, test_df = split_data(df, CONFIG['train_pct'], CONFIG['val_pct'], CONFIG['test_pct'])
    
    # Create datasets
    train_dataset, val_dataset = create_dataset(train_df, val_df, CONFIG)
    
    # Create model
    model = create_model(train_dataset, CONFIG)
    
    # Train model
    trainer, best_model_path = train_model(model, train_dataset, val_dataset, CONFIG)
    
    # Load best model for evaluation
    # ensure map to GPU if available
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    if torch.cuda.is_available():
        best_model = best_model.to('cuda')
    
    # Evaluate on test set
    metrics = evaluate_model(best_model, test_df, train_dataset, CONFIG)
    
    # Verify checkpoint
    checkpoint_valid = verify_checkpoint(best_model_path)
    
    # Summary
    print(f"\n{'='*70}")
    print("  TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"\n✅ Training completed successfully!")
    print(f"\n📁 Model saved to: {best_model_path}")
    print(f"\n📊 Test Metrics:")
    print(f"   - Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
    print(f"   - MAPE: {metrics['mape']:.2f}%")
    print(f"   - MAE: {metrics['mae']:.6f}")
    print(f"   - RMSE: {metrics['rmse']:.6f}")
    print(f"\n✓ Checkpoint verification: {'PASSED' if checkpoint_valid else 'FAILED'}")
    print(f"\n{'='*70}\n")
    
    # Save final checkpoint with clear name
    final_path = Path(CONFIG['checkpoint_dir']) / f"tft_XAUUSD_15m_{CONFIG['lookback_bars']}x{CONFIG['horizon_bars']}_final.ckpt"
    import shutil
    shutil.copy(best_model_path, final_path)
    print(f"📦 Final checkpoint: {final_path}")
    
    return best_model, metrics

if __name__ == "__main__":
    try:
        model, metrics = main()
        print("\n✅ All done! Model is ready for deployment.")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
