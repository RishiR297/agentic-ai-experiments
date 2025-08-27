#!/bin/bash

# LangGraph Medical Assistant API Deployment Script
# This script sets up and runs the API server for team testing

echo "🏥 LangGraph Medical Assistant API Deployment"
echo "=============================================="
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.11+ and try again."
    exit 1
fi

echo "✅ Python found: $(python --version)"

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip and try again."
    exit 1
fi

echo "✅ pip found: $(pip --version)"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies. Please check requirements.txt"
    exit 1
fi

echo "✅ Dependencies installed successfully"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  .env file not found. Please create .env with the following variables:"
    echo "   AZURE_OPENAI_ENDPOINT=your_endpoint"
    echo "   AZURE_OPENAI_API_KEY=your_key"
    echo "   AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your_deployment"
    echo ""
    read -p "Press Enter to continue when .env is ready..."
fi

echo "✅ Environment configuration found"

# Start the API server
echo ""
echo "🚀 Starting LangGraph Medical Assistant API Server..."
echo "   Local URL: http://localhost:8502"
echo "   Swagger UI: http://localhost:8502/docs"
echo "   ReDoc: http://localhost:8502/redoc"
echo ""
echo "📡 Server will be accessible on LAN at: http://$(hostname -I | awk '{print $1}'):8502"
echo ""
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Start uvicorn server
python -m uvicorn src.api.langgraph_server:app --host 0.0.0.0 --port 8502 --reload
