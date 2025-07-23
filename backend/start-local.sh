#!/bin/bash

# iQore Chatbot Backend - Local Development Start Script

echo "🚀 Starting iQore Chatbot Backend..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.11+ first."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if port 8080 is available
if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 8080 is already in use. Please stop the service using port 8080 first."
    exit 1
fi

echo "✅ Starting FastAPI server on http://localhost:8080"
echo "📖 API Documentation: http://localhost:8080/docs"
echo "🔍 Health Check: http://localhost:8080/"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server with uvicorn directly (matching production)
uvicorn main:app --host 0.0.0.0 --port 8080 --reload 