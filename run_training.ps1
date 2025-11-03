# ====================================================================
# TFT Model Training Script
# Trains the Temporal Fusion Transformer on historical XAUUSD data
# This is DIFFERENT from learning from trade log
# ====================================================================

Write-Host ""
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "TFT MODEL TRAINING - HISTORICAL DATA" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if training data exists
if (-not (Test-Path "XAUUSD_15M.csv")) {
    Write-Host "ERROR: Training data not found!" -ForegroundColor Red
    Write-Host "Expected: XAUUSD_15M.csv" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please ensure XAUUSD_15M.csv is in the TFTmodel directory" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Get data info
$lineCount = (Get-Content "XAUUSD_15M.csv" | Measure-Object -Line).Lines
$dataRows = $lineCount - 1
$fileSize = (Get-Item "XAUUSD_15M.csv").Length / 1MB

Write-Host "Training Data:" -ForegroundColor Yellow
Write-Host "  File: XAUUSD_15M.csv"
Write-Host "  Rows: $dataRows bars (15-minute)"
Write-Host "  Size: $($fileSize.ToString('N2')) MB"
Write-Host ""

Write-Host "Training Configuration:" -ForegroundColor Yellow
Write-Host "  - Model: Temporal Fusion Transformer (TFT)"
Write-Host "  - Hidden Size: 160"
Write-Host "  - Attention Heads: 4"
Write-Host "  - Max Epochs: 30"
Write-Host "  - Batch Size: 128"
Write-Host ""

Write-Host "NOTE: This trains the TFT model on HISTORICAL price data" -ForegroundColor Cyan
Write-Host "      For learning from TRADE RESULTS, use: run_learning.ps1" -ForegroundColor Cyan
Write-Host ""

# Estimate training time
$estimatedHours = [Math]::Ceiling($dataRows / 10000)
Write-Host "Estimated training time: ~$estimatedHours hours" -ForegroundColor Yellow
Write-Host ""

$confirm = Read-Host "Start TFT model training? (y/n)"
if ($confirm -ne "y" -and $confirm -ne "Y") {
    Write-Host "Training cancelled." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 0
}

Write-Host ""
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "TRAINING STARTED" -ForegroundColor Green
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""

# Set encoding and run training
$env:PYTHONIOENCODING = 'utf-8'

try {
    python -m src.training.train_tft
    
    Write-Host ""
    Write-Host "====================================================================" -ForegroundColor Cyan
    Write-Host "TRAINING COMPLETE!" -ForegroundColor Green
    Write-Host "====================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Model saved to: artifacts/checkpoints/" -ForegroundColor Green
    Write-Host ""
}
catch {
    Write-Host ""
    Write-Host "====================================================================" -ForegroundColor Red
    Write-Host "TRAINING FAILED" -ForegroundColor Red
    Write-Host "====================================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host ""
}

Read-Host "Press Enter to exit"
