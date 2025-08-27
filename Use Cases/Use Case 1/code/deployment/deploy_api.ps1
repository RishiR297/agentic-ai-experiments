# LangGraph Medical Assistant API Deployment Script (Windows)
# This script sets up and runs the API server for team testing

Write-Host "🏥 LangGraph Medical Assistant API Deployment" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python is not installed. Please install Python 3.11+ and try again." -ForegroundColor Red
    exit 1
}

# Check if pip is available
try {
    $pipVersion = pip --version 2>&1
    Write-Host "✅ pip found: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ pip is not installed. Please install pip and try again." -ForegroundColor Red
    exit 1
}

# Install dependencies
Write-Host ""
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install dependencies. Please check requirements.txt" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host ""
    Write-Host "⚠️  .env file not found. Please create .env with the following variables:" -ForegroundColor Yellow
    Write-Host "   AZURE_OPENAI_ENDPOINT=your_endpoint" -ForegroundColor White
    Write-Host "   AZURE_OPENAI_API_KEY=your_key" -ForegroundColor White
    Write-Host "   AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your_deployment" -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter to continue when .env is ready"
}

Write-Host "✅ Environment configuration found" -ForegroundColor Green

# Get local IP for LAN access
$localIP = (Get-NetIPConfiguration | Where-Object {$_.IPv4DefaultGateway -ne $null}).IPv4Address.IPAddress

# Start the API server
Write-Host ""
Write-Host "🚀 Starting LangGraph Medical Assistant API Server..." -ForegroundColor Cyan
Write-Host "   Local URL: http://localhost:8502" -ForegroundColor White
Write-Host "   Swagger UI: http://localhost:8502/docs" -ForegroundColor White
Write-Host "   ReDoc: http://localhost:8502/redoc" -ForegroundColor White
Write-Host ""
Write-Host "📡 Server will be accessible on LAN at: http://${localIP}:8502" -ForegroundColor Yellow
Write-Host ""
Write-Host "🛑 Press Ctrl+C to stop the server" -ForegroundColor Red
Write-Host ""

# Start uvicorn server
Write-Host "💡 For public access (team sharing):" -ForegroundColor Cyan
Write-Host "   1. Keep this server running" -ForegroundColor White
Write-Host "   2. Open another terminal and run: ngrok http 8502" -ForegroundColor Yellow
Write-Host "   3. Share the ngrok https URL with your team" -ForegroundColor Yellow
Write-Host ""
python -m uvicorn src.api.langgraph_server:app --host 0.0.0.0 --port 8502 --reload
