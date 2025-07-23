#!/bin/bash

# iQore Chatbot Backend Startup Script
echo "🚀 Starting iQore Chatbot Backend..."
echo "📊 Environment: ${GAE_ENV:-local}"
echo "🔧 Port: ${PORT:-8080}"
echo "🏠 Host: 0.0.0.0"

# Start the application using uvicorn directly for maximum compatibility
exec python -m uvicorn main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8080} \
    --log-level info \
    --access-log \
    --timeout-keep-alive 30 