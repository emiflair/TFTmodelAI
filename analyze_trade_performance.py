"""
Analyze Trade Log for Model Performance
Uses logged trades to evaluate model accuracy and generate training insights
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def analyze_prediction_accuracy(log_file: str = "trading_bot/trade_log.csv"):
    """
    Analyze how accurate the AI predictions were
    
    Returns insights for model improvement:
    - Directional accuracy (% of correct up/down predictions)
    - Confidence calibration (high confidence = better results?)
    - Overconfident vs underconfident predictions
    - Which market conditions work best
    """
    
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"❌ Trade log not found: {log_file}")
        return
    
    # Load trade log
    df = pd.read_csv(log_path)
    
    # Filter closed trades only
    closed = df[df['status'].isin(['CLOSED_WIN', 'CLOSED_LOSS', 'CLOSED_BREAKEVEN'])].copy()
    
    if len(closed) == 0:
        print("❌ No closed trades found in log")
        return
    
    print("="*80)
    print("📊 TRADE PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"\nTotal Trades: {len(closed)}")
    print(f"Date Range: {closed['timestamp'].min()} to {closed['timestamp'].max()}")
    
    # 1. Overall Performance
    wins = closed[closed['status'] == 'CLOSED_WIN']
    losses = closed[closed['status'] == 'CLOSED_LOSS']
    
    win_rate = len(wins) / len(closed) * 100 if len(closed) > 0 else 0
    avg_win = wins['profit_loss'].mean() if len(wins) > 0 else 0
    avg_loss = losses['profit_loss'].mean() if len(losses) > 0 else 0
    
    total_profit = closed['profit_loss'].sum()
    profit_factor = abs(wins['profit_loss'].sum() / losses['profit_loss'].sum()) if len(losses) > 0 and losses['profit_loss'].sum() != 0 else 0
    
    print(f"\n📈 OVERALL PERFORMANCE")
    print(f"  Win Rate: {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"  Avg Win: ${avg_win:.2f}")
    print(f"  Avg Loss: ${avg_loss:.2f}")
    print(f"  Total Profit: ${total_profit:.2f}")
    print(f"  Profit Factor: {profit_factor:.2f}")
    
    # 2. Directional Accuracy
    print(f"\n🎯 AI PREDICTION ACCURACY")
    
    # Determine if prediction direction matched outcome
    closed['actual_direction'] = closed['profit_loss'].apply(
        lambda x: 'UP' if x > 0 else ('DOWN' if x < 0 else 'NEUTRAL')
    )
    closed['prediction_correct'] = (closed['ai_direction'] == closed['actual_direction']) | (closed['actual_direction'] == 'NEUTRAL')
    
    directional_accuracy = closed['prediction_correct'].mean() * 100
    print(f"  Directional Accuracy: {directional_accuracy:.1f}%")
    
    # By action type
    for action in ['BUY', 'SELL']:
        action_trades = closed[closed['action'] == action]
        if len(action_trades) > 0:
            action_correct = action_trades['prediction_correct'].mean() * 100
            action_winrate = (action_trades['status'] == 'CLOSED_WIN').mean() * 100
            print(f"  {action} Trades: {action_correct:.1f}% accuracy, {action_winrate:.1f}% win rate")
    
    # 3. Confidence Calibration
    print(f"\n🔍 CONFIDENCE ANALYSIS")
    
    # Group by confidence levels
    closed['confidence_bucket'] = pd.cut(
        closed['ai_confidence'], 
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['Low (0-0.3)', 'Medium (0.3-0.5)', 'High (0.5-0.7)', 'Very High (0.7-1.0)']
    )
    
    for bucket in closed['confidence_bucket'].dropna().unique():
        bucket_trades = closed[closed['confidence_bucket'] == bucket]
        bucket_winrate = (bucket_trades['status'] == 'CLOSED_WIN').mean() * 100
        bucket_accuracy = bucket_trades['prediction_correct'].mean() * 100
        bucket_profit = bucket_trades['profit_loss'].sum()
        
        print(f"  {bucket}:")
        print(f"    Trades: {len(bucket_trades)}")
        print(f"    Win Rate: {bucket_winrate:.1f}%")
        print(f"    Accuracy: {bucket_accuracy:.1f}%")
        print(f"    Total P/L: ${bucket_profit:.2f}")
    
    # 4. Prediction Error Analysis
    print(f"\n📉 PREDICTION ERROR ANALYSIS")
    
    # Calculate actual move vs predicted move (only for trades with complete data)
    closed_complete = closed.dropna(subset=['profit_loss_pct', 'predicted_move_pct'])
    
    if len(closed_complete) > 0:
        closed_complete['actual_move_pct'] = closed_complete['profit_loss_pct']
        closed_complete['prediction_error'] = abs(closed_complete['actual_move_pct'] - closed_complete['predicted_move_pct'])
        
        avg_error = closed_complete['prediction_error'].mean()
        print(f"  Average Prediction Error: {avg_error:.2f}%")
        print(f"  Median Prediction Error: {closed_complete['prediction_error'].median():.2f}%")
        
        # Best and worst predictions
        best_pred = closed_complete.loc[closed_complete['prediction_error'].idxmin()]
        worst_pred = closed_complete.loc[closed_complete['prediction_error'].idxmax()]
        
        print(f"\n  Best Prediction:")
        print(f"    Predicted: {best_pred['predicted_move_pct']:.2f}% | Actual: {best_pred['actual_move_pct']:.2f}%")
        print(f"    Confidence: {best_pred['ai_confidence']:.2f} | Result: {best_pred['status']}")
        
        print(f"\n  Worst Prediction:")
        print(f"    Predicted: {worst_pred['predicted_move_pct']:.2f}% | Actual: {worst_pred['actual_move_pct']:.2f}%")
        print(f"    Confidence: {worst_pred['ai_confidence']:.2f} | Result: {worst_pred['status']}")
    else:
        print(f"  ⚠️  No complete prediction data available")
    
    # 5. Market Conditions
    print(f"\n🌍 MARKET CONDITIONS")
    
    spread_median = closed['spread'].median()
    low_spread = closed[closed['spread'] <= spread_median]
    high_spread = closed[closed['spread'] > spread_median]
    
    if len(low_spread) > 0 and len(high_spread) > 0:
        print(f"  Low Spread (<= {spread_median:.1f} pips):")
        print(f"    Win Rate: {(low_spread['status'] == 'CLOSED_WIN').mean() * 100:.1f}%")
        print(f"    Avg Profit: ${low_spread['profit_loss'].mean():.2f}")
        
        print(f"  High Spread (> {spread_median:.1f} pips):")
        print(f"    Win Rate: {(high_spread['status'] == 'CLOSED_WIN').mean() * 100:.1f}%")
        print(f"    Avg Profit: ${high_spread['profit_loss'].mean():.2f}")
    
    # 6. Risk Management Effectiveness
    print(f"\n⚖️ RISK MANAGEMENT")
    
    avg_risk_pct = closed['risk_pct'].mean()
    avg_rr_ratio = closed['rr_ratio'].mean()
    
    print(f"  Average Risk: {avg_risk_pct:.2f}%")
    print(f"  Average R:R Ratio: {avg_rr_ratio:.2f}")
    
    # Actual vs target risk
    closed['actual_rr'] = abs(closed['profit_loss'] / closed['risk_amount'])
    achieved_rr = closed[closed['status'] == 'CLOSED_WIN']['actual_rr'].mean()
    
    print(f"  Achieved R:R on Wins: {achieved_rr:.2f}x")
    
    # 7. Generate Training Recommendations
    print(f"\n💡 RECOMMENDATIONS FOR MODEL IMPROVEMENT")
    
    recommendations = []
    
    if directional_accuracy < 60:
        recommendations.append(f"⚠️  Low directional accuracy ({directional_accuracy:.1f}%) - Consider retraining with more recent data")
    
    if len(closed) < 30:
        recommendations.append(f"⚠️  Limited data ({len(closed)} trades) - Collect more trades before major model changes")
    
    # Check if high confidence predictions are actually better
    high_conf = closed[closed['ai_confidence'] > 0.6]
    low_conf = closed[closed['ai_confidence'] <= 0.6]
    
    if len(high_conf) > 0 and len(low_conf) > 0:
        high_conf_winrate = (high_conf['status'] == 'CLOSED_WIN').mean()
        low_conf_winrate = (low_conf['status'] == 'CLOSED_WIN').mean()
        
        if high_conf_winrate < low_conf_winrate:
            recommendations.append("⚠️  High confidence predictions performing worse - Model may be overconfident")
    
    if len(closed_complete) > 0 and avg_error > 2.0:
        recommendations.append(f"⚠️  High prediction error ({avg_error:.2f}%) - Consider adding more features or regularization")
    
    if win_rate >= 55 and profit_factor >= 1.5:
        recommendations.append("✅ Model performing well - Continue collecting data for validation")
    
    if len(recommendations) == 0:
        recommendations.append("✅ No major issues detected - Continue monitoring")
    
    for rec in recommendations:
        print(f"  {rec}")
    
    # 8. Export for Training
    print(f"\n📁 EXPORT FOR RETRAINING")
    
    export_data = closed[[
        'timestamp', 'ai_direction', 'ai_confidence', 'predicted_move_pct',
        'predicted_q10', 'predicted_q50', 'predicted_q90',
        'actual_direction', 'actual_move_pct', 'prediction_correct',
        'status', 'profit_loss', 'spread', 'duration_minutes'
    ]].copy()
    
    export_file = "artifacts/trade_performance_analysis.csv"
    Path(export_file).parent.mkdir(parents=True, exist_ok=True)
    export_data.to_csv(export_file, index=False)
    
    print(f"  Exported analysis to: {export_file}")
    print(f"  Use this data to:")
    print(f"    - Identify weak prediction patterns")
    print(f"    - Retrain model with recent market data")
    print(f"    - Adjust confidence thresholds")
    
    print("\n" + "="*80)
    
    return closed


if __name__ == "__main__":
    analyze_prediction_accuracy()
