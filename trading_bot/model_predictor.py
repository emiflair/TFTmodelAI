"""
Model Prediction Service
Wrapper for TFT model inference with live market data
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import logging

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.api import predict_r3_quantiles

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Handles model predictions for trading bot"""
    
    def __init__(
        self,
        checkpoint_path: str = None,
        scaler_path: str = None,
        manifest_path: str = None
    ):
        """
        Initialize model predictor
        
        Args:
            checkpoint_path: Path to trained model checkpoint (.ckpt) - None = auto-detect
            scaler_path: Path to data scaler (.pkl) - None = auto-detect
            manifest_path: Path to feature manifest (.json) - None = auto-detect
        """
        # Auto-detect model files if not provided
        if checkpoint_path is None or scaler_path is None or manifest_path is None:
            logger.info("Auto-detecting model files...")
            detected = get_latest_model()
            
            if checkpoint_path is None:
                checkpoint_path = detected['checkpoint']
            if scaler_path is None:
                scaler_path = detected['scaler']
            if manifest_path is None:
                manifest_path = detected['manifest']
            
            logger.info(f"Found checkpoint: {checkpoint_path}")
            logger.info(f"Found scaler: {scaler_path}")
            logger.info(f"Found manifest: {manifest_path}")
        
        self.checkpoint_path = Path(checkpoint_path)
        self.scaler_path = Path(scaler_path)
        self.manifest_path = Path(manifest_path)
        
        # Validate paths
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        logger.info(f"Model predictor initialized")
        logger.info(f"  Checkpoint: {self.checkpoint_path.name}")
        logger.info(f"  Scaler: {self.scaler_path.name}")
    
    def predict(self, market_data: pd.DataFrame) -> Optional[Dict]:
        """
        Make prediction using TFT model
        
        Args:
            market_data: DataFrame with columns [time, open, high, low, close, volume, spread]
                        Must have at least 256 bars (model lookback requirement)
        
        Returns:
            Dict with prediction results:
            {
                'q10': float,      # 10th percentile (lower bound)
                'q50': float,      # 50th percentile (median/expected)
                'q90': float,      # 90th percentile (upper bound)
                'current_price': float,
                'predicted_move': float,  # Expected price change
                'move_pct': float,        # Expected % change
                'confidence': float,      # Prediction confidence (0-1)
                'direction': str,         # 'UP', 'DOWN', or 'NEUTRAL'
                'ok': bool,
                'reason': str
            }
        """
        try:
            # Validate input data (API expects timestamp and tick_volume)
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'tick_volume']
            missing_cols = [col for col in required_cols if col not in market_data.columns]
            if missing_cols:
                return {
                    'ok': False,
                    'reason': f"Missing columns: {missing_cols}"
                }
            
            if len(market_data) < 256:
                return {
                    'ok': False,
                    'reason': f"Insufficient data: {len(market_data)} bars (need 256)"
                }
            
            # Get model prediction
            logger.info(f"Making prediction with {len(market_data)} bars")
            logger.info(f"Market data columns: {list(market_data.columns)}")
            logger.info(f"First timestamp: {market_data['timestamp'].iloc[0]}")
            logger.info(f"Last timestamp: {market_data['timestamp'].iloc[-1]}")
            # Check timestamp intervals
            deltas = market_data['timestamp'].diff().dropna()
            logger.info(f"Timestamp intervals (minutes): {(deltas.dt.total_seconds() / 60).unique()}")
            
            result = predict_r3_quantiles(
                latest_bars_df=market_data,
                checkpoint_path=self.checkpoint_path,
                scaler_path=self.scaler_path,
                manifest_path=self.manifest_path
            )
            
            if not result['ok']:
                logger.error(f"Prediction failed: {result['reason']}")
                return result
            
            # Extract predictions
            q10 = result['q10']
            q50 = result['q50']
            q90 = result['q90']
            
            # Get current price
            current_price = market_data['close'].iloc[-1]
            
            # Calculate prediction metrics
            predicted_move = q50 - current_price
            move_pct = (predicted_move / current_price) * 100
            
            # Calculate confidence (tighter range = higher confidence)
            prediction_range = q90 - q10
            confidence = max(0.0, 1.0 - (prediction_range / current_price))
            
            # Determine direction
            if move_pct > 0.1:  # More than 0.1% up
                direction = 'UP'
            elif move_pct < -0.1:  # More than 0.1% down
                direction = 'DOWN'
            else:
                direction = 'NEUTRAL'
            
            # Compile result
            prediction = {
                'q10': q10,
                'q50': q50,
                'q90': q90,
                'current_price': current_price,
                'predicted_move': predicted_move,
                'move_pct': move_pct,
                'confidence': confidence,
                'direction': direction,
                'prediction_range': prediction_range,
                'ok': True,
                'reason': None
            }
            
            logger.info(f"Prediction: {direction} | Move: {move_pct:.3f}% | Confidence: {confidence:.2f}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return {
                'ok': False,
                'reason': str(e)
            }
    
    def get_trading_signal(
        self,
        market_data: pd.DataFrame,
        min_confidence: float = 0.90,
        min_move_pct: float = 0.15
    ) -> Dict:
        """
        Generate trading signal with filters
        
        Args:
            market_data: Market data for prediction
            min_confidence: Minimum confidence threshold (0-1)
            min_move_pct: Minimum expected move % to trade
        
        Returns:
            Dict with trading signal:
            {
                'signal': 'BUY', 'SELL', or 'HOLD',
                'confidence': float,
                'entry_price': float,
                'stop_loss': float,    # Based on q10 for BUY, q90 for SELL
                'take_profit': float,  # Based on q90 for BUY, q10 for SELL
                'expected_move': float,
                'prediction': dict,    # Full prediction details
                'ok': bool,
                'reason': str
            }
        """
        # Get prediction
        prediction = self.predict(market_data)
        
        if not prediction['ok']:
            return {
                'signal': 'HOLD',
                'ok': False,
                'reason': prediction['reason'],
                'prediction': prediction
            }
        
        # Extract values
        direction = prediction['direction']
        confidence = prediction['confidence']
        move_pct = abs(prediction['move_pct'])
        current_price = prediction['current_price']
        q10 = prediction['q10']
        q50 = prediction['q50']
        q90 = prediction['q90']
        
        # Check confidence threshold
        if confidence < min_confidence:
            return {
                'signal': 'HOLD',
                'ok': True,
                'reason': f"Low confidence: {confidence:.2f} < {min_confidence}",
                'confidence': confidence,
                'prediction': prediction
            }
        
        # Check minimum move threshold
        if move_pct < min_move_pct:
            return {
                'signal': 'HOLD',
                'ok': True,
                'reason': f"Small expected move: {move_pct:.2f}% < {min_move_pct}%",
                'confidence': confidence,
                'prediction': prediction
            }
        
        # Generate signal
        if direction == 'UP':
            signal = 'BUY'
            entry_price = current_price
            stop_loss = q10  # Lower bound
            take_profit = q90  # Upper bound
        elif direction == 'DOWN':
            signal = 'SELL'
            entry_price = current_price
            stop_loss = q90  # Upper bound (for short)
            take_profit = q10  # Lower bound (for short)
        else:
            signal = 'HOLD'
            entry_price = current_price
            stop_loss = None
            take_profit = None
        
        # Calculate expected reward:risk ratio
        if signal in ['BUY', 'SELL'] and stop_loss and take_profit:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            reward_risk_ratio = reward / risk if risk > 0 else 0
        else:
            reward_risk_ratio = 0
        
        result = {
            'signal': signal,
            'confidence': confidence,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'expected_move': prediction['predicted_move'],
            'move_pct': prediction['move_pct'],
            'reward_risk_ratio': reward_risk_ratio,
            'prediction': prediction,
            'ok': True,
            'reason': None
        }
        
        logger.info(f"Trading signal: {signal} | Confidence: {confidence:.2f} | R:R: {reward_risk_ratio:.2f}")
        
        return result


def get_latest_model() -> Dict:
    """
    Find latest trained model files in artifacts directory
    
    Returns:
        Dict with checkpoint, scaler, and manifest paths
    """
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    
    # Find latest checkpoint
    checkpoints = sorted(
        artifacts_dir.glob("checkpoints/*.ckpt"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    if not checkpoints:
        raise FileNotFoundError("No model checkpoints found")
    
    checkpoint = checkpoints[0]
    
    # Find corresponding scaler (use latest)
    scalers = sorted(
        artifacts_dir.glob("scalers/scaler_*.pkl"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    if not scalers:
        raise FileNotFoundError("No scaler files found")
    
    scaler = scalers[0]
    
    # Get manifest
    manifest = artifacts_dir / "manifests" / "feature_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError("Feature manifest not found")
    
    logger.info(f"Using latest model:")
    logger.info(f"  Checkpoint: {checkpoint.name}")
    logger.info(f"  Scaler: {scaler.name}")
    
    return {
        'checkpoint': str(checkpoint),
        'scaler': str(scaler),
        'manifest': str(manifest)
    }
