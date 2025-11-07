"""
Simple Local Training Script for CPU
Train TFT model on your PC without GPU
"""

import os
import sys

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Run training
from src.training.train_tft import main

if __name__ == "__main__":
    print("="*80)
    print("LOCAL CPU TRAINING - XAUUSD TFT MODEL")
    print("="*80)
    print("\nTraining configuration:")
    print("  - Accelerator: CPU")
    print("  - Data: XAUUSD 15-minute bars")
    print("  - Horizon: 3 bars ahead")
    print("  - Lookback: 256 bars")
    print("\nNote: CPU training is slower than GPU but produces")
    print("      CPU-compatible checkpoints that work on any PC.")
    print("="*80)
    print()
    
    # Add fast-dev-run flag for quick testing
    if len(sys.argv) == 1:
        print("Usage:")
        print("  python train_local.py                    # Full training")
        print("  python train_local.py --fast-dev-run     # Quick test (1 epoch, 1 split)")
        print("  python train_local.py --max-epochs 10    # Custom epochs")
        print("  python train_local.py --splits 1         # Train on 1 split only")
        print()
    
    # Run the main training function
    sys.exit(main())
