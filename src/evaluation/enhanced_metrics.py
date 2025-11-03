"""Enhanced evaluation metrics for TFT trading performance analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TradingPerformanceMetrics:
    """Comprehensive trading performance metrics."""
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_return: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float


@dataclass
class RegimeAnalysis:
    """Performance analysis by market regime."""
    low_volatility: Dict[str, float]
    normal_volatility: Dict[str, float] 
    high_volatility: Dict[str, float]
    trending_market: Dict[str, float]
    ranging_market: Dict[str, float]


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio of returns."""
    if len(returns) == 0:
        return float('nan')
    
    excess_returns = returns - risk_free_rate
    if np.std(returns) == 0:
        return float('nan')
    
    # Annualize for 15-minute returns (35040 bars per year)
    annualization_factor = np.sqrt(35040)
    return float(np.mean(excess_returns) / np.std(returns) * annualization_factor)


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown from returns series."""
    if len(returns) == 0:
        return float('nan')
    
    cumulative_returns = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    
    return float(np.min(drawdowns))


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio (downside deviation instead of total volatility)."""
    if len(returns) == 0:
        return float('nan')
    
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf') if np.mean(excess_returns) > 0 else float('nan')
    
    downside_deviation = np.std(downside_returns)
    if downside_deviation == 0:
        return float('nan')
    
    annualization_factor = np.sqrt(35040)
    return float(np.mean(excess_returns) / downside_deviation * annualization_factor)


def calculate_calmar_ratio(returns: np.ndarray) -> float:
    """Calculate Calmar ratio (annual return / max drawdown)."""
    if len(returns) == 0:
        return float('nan')
    
    annual_return = (np.prod(1 + returns) ** (35040 / len(returns))) - 1
    max_dd = abs(calculate_max_drawdown(returns))
    
    if max_dd == 0:
        return float('inf') if annual_return > 0 else float('nan')
    
    return float(annual_return / max_dd)


def calculate_hit_rate_by_regime(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate hit rates segmented by market regimes."""
    results = {}
    
    # Volatility regimes
    for regime in [0, 1, 2]:  # Low, Normal, High volatility
        mask = df.get('volatility_regime', pd.Series([1]*len(df))) == regime
        regime_data = df[mask]
        if len(regime_data) > 0:
            hit_rate = directional_hit_rate(
                regime_data['target'].values,
                regime_data['q50'].values
            )
            regime_name = ['low_vol', 'normal_vol', 'high_vol'][regime]
            results[regime_name] = hit_rate
    
    # Market phases
    for phase in [0, 1, 2]:  # Ranging, Trending, Breakout
        mask = df.get('market_phase', pd.Series([0]*len(df))) == phase
        phase_data = df[mask]
        if len(phase_data) > 0:
            hit_rate = directional_hit_rate(
                phase_data['target'].values,
                phase_data['q50'].values
            )
            phase_name = ['ranging', 'trending', 'breakout'][phase]
            results[phase_name] = hit_rate
    
    # Session analysis
    for session in ['Asia', 'London', 'NY', 'Other']:
        mask = df.get('session_label', pd.Series(['Other']*len(df))) == session
        session_data = df[mask]
        if len(session_data) > 0:
            hit_rate = directional_hit_rate(
                session_data['target'].values,
                session_data['q50'].values
            )
            results[f'session_{session.lower()}'] = hit_rate
    
    return results


def calculate_prediction_consistency(df: pd.DataFrame, window: int = 100) -> Dict[str, float]:
    """Calculate prediction consistency metrics."""
    results = {}
    
    if len(df) < window:
        return results
    
    # Rolling hit rate consistency
    rolling_hit_rates = []
    for i in range(window, len(df)):
        window_data = df.iloc[i-window:i]
        hit_rate = directional_hit_rate(
            window_data['target'].values,
            window_data['q50'].values
        )
        rolling_hit_rates.append(hit_rate)
    
    if len(rolling_hit_rates) > 0:
        results['hit_rate_stability'] = float(np.std(rolling_hit_rates))
        results['hit_rate_trend'] = float(np.corrcoef(
            range(len(rolling_hit_rates)), 
            rolling_hit_rates
        )[0, 1]) if len(rolling_hit_rates) > 1 else 0.0
    
    # Prediction magnitude consistency
    pred_magnitudes = np.abs(df['q50'].values)
    if len(pred_magnitudes) > 0:
        results['prediction_magnitude_std'] = float(np.std(pred_magnitudes))
    
    # Confidence interval consistency
    if 'q10' in df.columns and 'q90' in df.columns:
        interval_widths = df['q90'].values - df['q10'].values
        results['confidence_width_std'] = float(np.std(interval_widths))
        results['confidence_width_mean'] = float(np.mean(interval_widths))
    
    return results


def calculate_risk_adjusted_returns(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate various risk-adjusted return metrics."""
    results = {}
    
    if 'target' not in df.columns or len(df) == 0:
        return results
    
    returns = df['target'].values
    predictions = df.get('q50', pd.Series(0, index=df.index)).values
    
    # Basic return metrics
    results['total_return'] = float(np.sum(returns))
    results['mean_return'] = float(np.mean(returns))
    results['return_volatility'] = float(np.std(returns))
    
    # Risk metrics
    results['sharpe_ratio'] = calculate_sharpe_ratio(returns)
    results['max_drawdown'] = calculate_max_drawdown(returns)
    results['sortino_ratio'] = calculate_sortino_ratio(returns)
    results['calmar_ratio'] = calculate_calmar_ratio(returns)
    
    # Prediction-based returns (if following predictions)
    if len(predictions) > 0:
        prediction_returns = returns * np.sign(predictions)  # Trade in direction of prediction
        valid_trades = np.sign(predictions) != 0
        
        if valid_trades.sum() > 0:
            results['strategy_return'] = float(np.sum(prediction_returns[valid_trades]))
            results['strategy_sharpe'] = calculate_sharpe_ratio(prediction_returns[valid_trades])
            results['strategy_max_dd'] = calculate_max_drawdown(prediction_returns[valid_trades])
            results['trade_count'] = int(valid_trades.sum())
    
    # Tail risk metrics
    if len(returns) > 0:
        results['var_95'] = float(np.percentile(returns, 5))  # Value at Risk (95%)
        results['var_99'] = float(np.percentile(returns, 1))  # Value at Risk (99%)
        results['cvar_95'] = float(np.mean(returns[returns <= np.percentile(returns, 5)]))  # Conditional VaR
    
    return results


def trading_performance_summary(df: pd.DataFrame, 
                              lower_q: str = 'q10', 
                              median_q: str = 'q50', 
                              upper_q: str = 'q90') -> TradingPerformanceMetrics:
    """Generate comprehensive trading performance summary."""
    
    if len(df) == 0 or 'target' not in df.columns:
        return TradingPerformanceMetrics(
            sharpe_ratio=float('nan'),
            max_drawdown=float('nan'),
            win_rate=float('nan'),
            profit_factor=float('nan'),
            avg_win=float('nan'),
            avg_loss=float('nan'),
            total_return=float('nan'),
            volatility=float('nan'),
            calmar_ratio=float('nan'),
            sortino_ratio=float('nan')
        )
    
    returns = df['target'].values
    predictions = df.get(median_q, pd.Series(0, index=df.index)).values
    
    # Generate trading signals based on predictions and confidence
    if lower_q in df.columns and upper_q in df.columns:
        confidence_width = df[upper_q].values - df[lower_q].values
        high_confidence = confidence_width < np.percentile(confidence_width, 30)  # Top 30% confidence
        strong_signal = np.abs(predictions) > np.percentile(np.abs(predictions), 70)  # Top 30% signal strength
        
        trade_mask = high_confidence & strong_signal
    else:
        trade_mask = np.abs(predictions) > 0
    
    # Calculate strategy returns
    strategy_returns = returns * np.sign(predictions)
    trading_returns = strategy_returns[trade_mask]
    
    # Basic metrics
    total_return = float(np.sum(trading_returns)) if len(trading_returns) > 0 else 0.0
    volatility = float(np.std(trading_returns)) if len(trading_returns) > 0 else 0.0
    
    # Win/Loss analysis
    wins = trading_returns[trading_returns > 0]
    losses = trading_returns[trading_returns < 0]
    
    win_rate = len(wins) / len(trading_returns) if len(trading_returns) > 0 else 0.0
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    
    # Profit factor
    total_wins = np.sum(wins) if len(wins) > 0 else 0.0
    total_losses = abs(np.sum(losses)) if len(losses) > 0 else 0.0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0.0
    
    # Risk metrics
    sharpe_ratio = calculate_sharpe_ratio(trading_returns) if len(trading_returns) > 0 else float('nan')
    max_drawdown = calculate_max_drawdown(trading_returns) if len(trading_returns) > 0 else float('nan')
    calmar_ratio = calculate_calmar_ratio(trading_returns) if len(trading_returns) > 0 else float('nan')
    sortino_ratio = calculate_sortino_ratio(trading_returns) if len(trading_returns) > 0 else float('nan')
    
    return TradingPerformanceMetrics(
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        avg_win=avg_win,
        avg_loss=avg_loss,
        total_return=total_return,
        volatility=volatility,
        calmar_ratio=calmar_ratio,
        sortino_ratio=sortino_ratio
    )


def directional_hit_rate(y_true: np.ndarray, median_pred: np.ndarray) -> float:
    """Calculate directional hit rate (imported from main metrics for consistency)."""
    eligible = (median_pred != 0) & (y_true != 0)
    if eligible.sum() == 0:
        return float("nan")
    hits = np.sign(median_pred[eligible]) == np.sign(y_true[eligible])
    return float(np.mean(hits))


def regime_performance_analysis(df: pd.DataFrame) -> RegimeAnalysis:
    """Comprehensive regime-based performance analysis."""
    
    # Initialize result dictionaries
    low_vol = {}
    normal_vol = {}
    high_vol = {}
    trending = {}
    ranging = {}
    
    # Volatility regime analysis
    for regime_val, regime_dict in [(0, low_vol), (1, normal_vol), (2, high_vol)]:
        mask = df.get('volatility_regime', pd.Series([1]*len(df))) == regime_val
        regime_data = df[mask]
        if len(regime_data) > 0:
            performance = calculate_risk_adjusted_returns(regime_data)
            hit_rate = directional_hit_rate(
                regime_data['target'].values,
                regime_data.get('q50', pd.Series(0, index=regime_data.index)).values
            )
            regime_dict.update(performance)
            regime_dict['hit_rate'] = hit_rate
    
    # Market phase analysis
    for phase_val, phase_dict in [(1, trending), (0, ranging)]:
        mask = df.get('market_phase', pd.Series([0]*len(df))) == phase_val
        phase_data = df[mask]
        if len(phase_data) > 0:
            performance = calculate_risk_adjusted_returns(phase_data)
            hit_rate = directional_hit_rate(
                phase_data['target'].values,
                phase_data.get('q50', pd.Series(0, index=phase_data.index)).values
            )
            phase_dict.update(performance)
            phase_dict['hit_rate'] = hit_rate
    
    return RegimeAnalysis(
        low_volatility=low_vol,
        normal_volatility=normal_vol,
        high_volatility=high_vol,
        trending_market=trending,
        ranging_market=ranging
    )