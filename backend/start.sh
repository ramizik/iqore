#!/bin/bash

# iQore Chatbot Backend Production Startup Script
echo "🚀 Starting iQore Chatbot Backend..."
echo "📊 Environment: ${GAE_ENV:-production}"
echo "🔧 Port: ${PORT:-8080}"
echo "🏠 Host: 0.0.0.0"

# Start the application using uvicorn directly (matching vocal AI approach)
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8080} \
    --log-level info \
    --access-log 