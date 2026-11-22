# ==============================================================================
# Dockerfile for Central-Agent: AI-Powered CI/CD Failure Triage Service
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing bytecode and enable immediate log streaming
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    WEBHOOK_PORT=8000 \
    WEBHOOK_HOST=0.0.0.0 \
    CHROMA_PERSIST_DIR=/app/chroma_data

WORKDIR /app

# Install system dependencies (curl for health check, build-essential for C-extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies list first to maximize layer caching
COPY requirements.txt .

# Install lightweight CPU-optimized PyTorch first, then install remaining dependencies
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Create non-root user and setup directories with appropriate permissions
RUN mkdir -p /app/chroma_data && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user for security best practices
USER appuser

# Expose Webhook server port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI Webhook Listener
CMD ["python", "webhook_server.py"]
