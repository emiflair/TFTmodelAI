# ====================================================================
# AI Trading Bot Runner (PowerShell)
# Starts the automated trading bot with TFT model predictions
# ====================================================================

Write-Host ""
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "AI TRADING BOT - STARTING" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if in correct directory
if (-not (Test-Path "trading_bot\bot.py")) {
    Write-Host "ERROR: trading_bot\bot.py not found!" -ForegroundColor Red
    Write-Host "Please run this script from the TFTmodel directory" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Display configuration
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  - Symbol: XAUUSD"
Write-Host "  - Timeframe: 15M"
Write-Host "  - Max Positions: 2 (same direction only)"
Write-Host "  - Daily Loss Limit: 4% (`$400)"
Write-Host "  - Daily Profit Target: 5% (`$500)"
Write-Host "  - Market Hours Protection: Active"
Write-Host ""

# Ask for confirmation
$confirm = Read-Host "Start trading bot? (y/n)"
if ($confirm -ne "y" -and $confirm -ne "Y") {
    Write-Host "Bot start cancelled." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 0
}

Write-Host ""
Write-Host "Starting bot in 3 seconds..." -ForegroundColor Green
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "BOT IS RUNNING - Press Ctrl+C to stop" -ForegroundColor Green
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""

# Change to trading_bot directory and run
Push-Location trading_bot
try {
    python bot.py
}
finally {
    Pop-Location
}

# If bot stops, show message
Write-Host ""
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "Bot has stopped." -ForegroundColor Yellow
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"
