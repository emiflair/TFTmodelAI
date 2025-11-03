# ====================================================================
# GitHub Repository Update Script
# Cleans repo and pushes current system setup
# ====================================================================

Write-Host ""
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "GITHUB REPOSITORY UPDATE" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "This will:" -ForegroundColor Yellow
Write-Host "  1. Remove all old files from GitHub repo" -ForegroundColor Red
Write-Host "  2. Add current system files" -ForegroundColor Green
Write-Host "  3. Commit changes" -ForegroundColor Green
Write-Host "  4. Push to GitHub (origin/main)" -ForegroundColor Green
Write-Host ""

Write-Host "Repository: TFTmodel (emiflair)" -ForegroundColor Cyan
Write-Host "Branch: main" -ForegroundColor Cyan
Write-Host ""

$confirm = Read-Host "⚠️  This will DELETE all old files from GitHub. Continue? (yes/no)"
if ($confirm -ne "yes") {
    Write-Host "Operation cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "STEP 1: Staging All Changes" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""

# Stage all deletions and modifications
Write-Host "Staging deleted and modified files..." -ForegroundColor Yellow
git add -A

Write-Host "✓ Changes staged" -ForegroundColor Green
Write-Host ""

# Show what will be committed
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "STEP 2: Review Changes" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""

$status = git status --short
Write-Host $status
Write-Host ""

$fileCount = ($status | Measure-Object -Line).Lines
Write-Host "Total files changed: $fileCount" -ForegroundColor Yellow
Write-Host ""

$confirm2 = Read-Host "Commit these changes? (y/n)"
if ($confirm2 -ne "y" -and $confirm2 -ne "Y") {
    Write-Host "Operation cancelled." -ForegroundColor Yellow
    git reset
    exit 0
}

Write-Host ""
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "STEP 3: Committing Changes" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""

$commitMessage = @"
Complete system update - AI Trading Bot

Major Changes:
- Added complete trading bot system (bot.py, strategy, risk management)
- Implemented CSV trade logging (31 columns)
- Added market hours protection (skip first 2h, last 1h)
- Smart exit logic (keep winners >$100, cut losers)
- Daily profit/loss limits (4% loss, 5% profit)
- Same-direction trading (max 2 positions)
- Created runner scripts (PowerShell + BAT)
- Added trade log analysis tools
- Updated training data management
- Cleaned up old EURUSD files
- Switched to XAUUSD (Gold) trading

Trading Bot Features:
- TFT model predictions (15M timeframe)
- Dynamic position sizing by confidence
- Adaptive R:R ratios (0.5-3.0)
- Trade log for model improvement
- Market hours validation
- Spread/slippage protection

Documentation:
- ALL_SCRIPTS_GUIDE.md - Complete script reference
- RUNNER_SCRIPTS.md - Quick start guide
- MARKET_HOURS.md - Trading hours rules
- TRAINING_DATA_GUIDE.md - Data management

Scripts:
- run_trading_bot.ps1/bat - Start trading bot
- run_learning.ps1/bat - Analyze trade performance
- run_training.ps1 - Train TFT model
- update_training_data.py - Manage historical data

Ready for production trading!
"@

git commit -m $commitMessage

Write-Host "✓ Changes committed" -ForegroundColor Green
Write-Host ""

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "STEP 4: Pushing to GitHub" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Pushing to origin/main..." -ForegroundColor Yellow
git push origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "====================================================================" -ForegroundColor Green
    Write-Host "✓ SUCCESS! Repository Updated" -ForegroundColor Green
    Write-Host "====================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Repository URL: https://github.com/emiflair/TFTmodel" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Changes pushed to GitHub successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "====================================================================" -ForegroundColor Red
    Write-Host "❌ PUSH FAILED" -ForegroundColor Red
    Write-Host "====================================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please check your internet connection and GitHub credentials." -ForegroundColor Yellow
    Write-Host ""
}

Write-Host ""
Read-Host "Press Enter to exit"
