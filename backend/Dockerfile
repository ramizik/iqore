# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Accept build arguments for environment variables
ARG OPENAI_API_KEY
ARG MONGODB_URI
ARG MONGODB_DATABASE

# Set environment variables from build args
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV MONGODB_URI=${MONGODB_URI}
ENV MONGODB_DATABASE=${MONGODB_DATABASE}
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Cloud Run will override this)
EXPOSE 8080

# Set environment variable for port (Cloud Run will provide PORT=8080)
ENV PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# Run the application directly with uvicorn (matching vocal AI approach)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"] 