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

# Fix locale/encoding issues for Click/Uvicorn
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONIOENCODING=utf-8

# Install system dependencies including locales
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    build-essential \
    locales \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen C.UTF-8

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

# Health check (matches our /health endpoint)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Make start script executable and use it to run the app (ensures proper locale)
RUN chmod +x start.sh
CMD ["./start.sh"] 