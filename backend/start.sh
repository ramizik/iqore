#!/bin/bash

# Set locale for UTF-8 support (fixes Click encoding issues)
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONIOENCODING=utf-8

# Print environment info for debugging
echo "🚀 Starting iQore Multi-Agent Chatbot Backend"
echo "📊 Python version: $(python --version)"
echo "🌐 Locale: $LC_ALL"
echo "🔧 Port: ${PORT:-8080}"

# Start the FastAPI app with uvicorn
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} 