#!/usr/bin/env pwsh
# Start Flask backend for Tax Fraud GNN

Write-Host "🚀 Starting Tax Fraud GNN Backend..." -ForegroundColor Green
Write-Host "📁 Working directory: $PWD" -ForegroundColor Cyan

# Check if we're in the right directory
if (-not (Test-Path "app.py")) {
    Write-Host "❌ Error: app.py not found in current directory" -ForegroundColor Red
    Write-Host "Please run this script from the tax-fraud-gnn directory" -ForegroundColor Yellow
    exit 1
}

# Check if data files exist
$dataFiles = @(
    "data/processed/companies_processed.csv",
    "data/processed/invoices_processed.csv",
    "data/raw/companies.csv",
    "data/raw/invoices.csv"
)

$foundData = $false
foreach ($file in $dataFiles) {
    if (Test-Path $file) {
        Write-Host "✓ Found: $file" -ForegroundColor Green
        $foundData = $true
    }
}

if (-not $foundData) {
    Write-Host "⚠️  Warning: Could not find data files" -ForegroundColor Yellow
}

# Start the Flask app
Write-Host "`n🔥 Starting Flask server on http://localhost:5000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server`n" -ForegroundColor Cyan

python app.py
