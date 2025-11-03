@echo off
REM ====================================================================
REM AI Trading Bot Runner
REM Starts the automated trading bot with TFT model predictions
REM ====================================================================

echo.
echo ====================================================================
echo AI TRADING BOT - STARTING
echo ====================================================================
echo.

REM Check if in correct directory
if not exist "trading_bot\bot.py" (
    echo ERROR: trading_bot\bot.py not found!
    echo Please run this script from the TFTmodel directory
    pause
    exit /b 1
)

REM Display configuration
echo Configuration:
echo   - Symbol: XAUUSD
echo   - Timeframe: 15M
echo   - Max Positions: 2 (same direction only)
echo   - Daily Loss Limit: 4%% ($400)
echo   - Daily Profit Target: 5%% ($500)
echo   - Market Hours Protection: Active
echo.

REM Ask for confirmation
set /p CONFIRM="Start trading bot? (y/n): "
if /i not "%CONFIRM%"=="y" (
    echo Bot start cancelled.
    pause
    exit /b 0
)

echo.
echo Starting bot in 3 seconds...
timeout /t 3 /nobreak >nul

echo.
echo ====================================================================
echo BOT IS RUNNING - Press Ctrl+C to stop
echo ====================================================================
echo.

REM Change to trading_bot directory and run
cd trading_bot
python bot.py

REM If bot stops, show message
echo.
echo ====================================================================
echo Bot has stopped.
echo ====================================================================
echo.
pause
