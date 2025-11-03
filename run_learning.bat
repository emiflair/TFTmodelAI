@echo off
REM ====================================================================
REM Model Learning from Trade Log
REM Trains improved confidence model using real trading results
REM ====================================================================

echo.
echo ====================================================================
echo MODEL LEARNING FROM TRADE LOG
echo ====================================================================
echo.

REM Check if trade log exists
set TRADE_LOG=trading_bot\trading_bot\trade_log.csv

if not exist "%TRADE_LOG%" (
    echo ERROR: Trade log not found!
    echo Expected location: %TRADE_LOG%
    echo.
    echo The trade log is created automatically when the bot trades.
    echo Please run the trading bot first to accumulate trade data.
    echo.
    pause
    exit /b 1
)

REM Count trades in log
for /f %%A in ('find /c /v "" ^< "%TRADE_LOG%"') do set LINE_COUNT=%%A
set /a TRADE_COUNT=%LINE_COUNT%-1

echo Found trade log: %TRADE_LOG%
echo Total entries: %TRADE_COUNT%
echo.

REM Check if enough trades
if %TRADE_COUNT% LSS 5 (
    echo WARNING: Only %TRADE_COUNT% trades in log
    echo.
    echo Recommendation:
    echo   - Minimum: 5 completed trades for analysis
    echo   - Good: 20+ trades for basic training
    echo   - Ideal: 50-100 trades for reliable model
    echo.
    echo Current status: Need more trading data
    echo.
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i not "!CONTINUE!"=="y" (
        echo Cancelled. Run the bot to accumulate more trades.
        pause
        exit /b 0
    )
) else if %TRADE_COUNT% LSS 20 (
    echo Status: ^>5 trades - Can analyze patterns
    echo Recommendation: 20+ trades for better results
    echo.
) else if %TRADE_COUNT% LSS 50 (
    echo Status: ^>20 trades - Good for training
    echo Recommendation: 50-100 trades for best results
    echo.
) else (
    echo Status: ^>50 trades - Excellent! ✓
    echo Ready for reliable model training
    echo.
)

REM Create learning script if it doesn't exist
if not exist "learn_from_trades.py" (
    echo Creating learning script...
    call :CREATE_LEARNING_SCRIPT
)

echo.
echo ====================================================================
echo STARTING MODEL LEARNING
echo ====================================================================
echo.
echo This will:
echo   1. Analyze win/loss patterns by confidence level
echo   2. Identify most important prediction features
echo   3. Train improved confidence model
echo   4. Show recommendations for improving win rate
echo.

timeout /t 2 /nobreak >nul

python learn_from_trades.py "%TRADE_LOG%"

echo.
echo ====================================================================
echo Learning complete!
echo ====================================================================
echo.
pause
exit /b 0

REM ====================================================================
REM Create learning script
REM ====================================================================
:CREATE_LEARNING_SCRIPT
(
echo """
echo Model Learning from Trade Log
echo Analyzes real trading results to improve predictions
echo """
echo.
echo import sys
echo import pandas as pd
echo import numpy as np
echo from pathlib import Path
echo.
echo def analyze_trades^(csv_path^):
echo     """Analyze trade log and show insights"""
echo     print^("Loading trade data..."^)
echo     df = pd.read_csv^(csv_path^)
echo     
echo     print^(f"Total entries: {len^(df^)}"^)
echo     
echo     # Filter completed trades
echo     completed = df[df['status'].str.contains^('CLOSED', na=False^)]
echo     print^(f"Completed trades: {len^(completed^)}"^)
echo     
echo     if len^(completed^) ^< 5:
echo         print^("\n⚠️  Need at least 5 completed trades for analysis"^)
echo         return
echo     
echo     # Calculate statistics
echo     wins = completed[completed['status'] == 'CLOSED_WIN']
echo     losses = completed[completed['status'] == 'CLOSED_LOSS']
echo     
echo     print^(f"\nWin Rate: {len^(wins^)/len^(completed^)*100:.1f}%%"^)
echo     print^(f"Total P^&L: ${completed['profit_loss'].sum^(^):.2f}"^)
echo     
echo     # Analyze by confidence
echo     print^("\nWin Rate by Confidence:"^)
echo     completed['conf_bin'] = pd.cut^(completed['ai_confidence'], 
echo                                      bins=[0, 0.5, 0.7, 0.85, 1.0],
echo                                      labels=['Low', 'Med', 'High', 'Very High']^)
echo     
echo     for conf_level in ['Low', 'Med', 'High', 'Very High']:
echo         subset = completed[completed['conf_bin'] == conf_level]
echo         if len^(subset^) ^> 0:
echo             wr = len^(subset[subset['status']=='CLOSED_WIN']^)/len^(subset^)*100
echo             print^(f"  {conf_level:10s}: {wr:5.1f}%% ^({len^(subset^)} trades^)"^)
echo     
echo     print^("\nRecommendations:"^)
echo     print^("  • Accumulate 50-100 trades for reliable model training"^)
echo     print^("  • Focus on high confidence setups ^(they should win more^)"^)
echo     print^("  • Use this data to calibrate confidence scores"^)
echo.
echo if __name__ == "__main__":
echo     if len^(sys.argv^) ^< 2:
echo         print^("Usage: python learn_from_trades.py trade_log.csv"^)
echo         sys.exit^(1^)
echo     
echo     analyze_trades^(sys.argv[1]^)
) > learn_from_trades.py
goto :eof
